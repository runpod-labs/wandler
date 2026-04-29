import { TextStreamer } from "@huggingface/transformers";
import type { LoadedModels } from "../models/manager.js";
import type { ChatMessage, GenerationOptions, GenerationProfile, Tool } from "../types/openai.js";
import { formatChat } from "../models/tokenizer.js";
import {
  elapsedMs,
  estimateAttentionScoresMb,
  estimateFullLogitsMb,
  GenerationExecutionError,
  logGenerationProfile,
  memorySnapshot,
  nowMs,
  serializedToolsChars,
} from "./profile.js";

/**
 * Generate tokens with a callback for each token.
 * Framework-agnostic — works with Hono streamSSE, raw http, or anything else.
 */
export async function generateStreamTokens(
  models: LoadedModels,
  modelId: string,
  messages: ChatMessage[],
  genOpts: GenerationOptions,
  onToken: (token: string) => void | Promise<void>,
  tools?: Tool[],
): Promise<{ promptTokens: number; completionTokens: number; profile: GenerationProfile }> {
  const started = nowMs();
  const memoryBefore = memorySnapshot();
  const formatStart = nowMs();
  const prompt = formatChat(models.tokenizer!, messages, modelId, tools, models.chatTemplate);
  const formatMs = elapsedMs(formatStart);
  const tokenizeStart = nowMs();
  const inputs = models.tokenizer!(prompt, { return_tensors: "pt" });
  const tokenizeMs = elapsedMs(tokenizeStart);
  const memoryAfterTokenize = memorySnapshot();
  const promptTokens = inputs.input_ids.dims[1]!;
  let completionTokens = 0;
  if (models.maxContextLength && promptTokens >= models.maxContextLength) {
    const message = (
      `Prompt has ${promptTokens} tokens, but the model context is ${models.maxContextLength} tokens. ` +
      "Reduce the prompt/tools."
    );
    const profile: GenerationProfile = {
      path: "stream",
      promptChars: prompt.length,
      toolsCount: tools?.length ?? 0,
      toolsChars: serializedToolsChars(tools),
      promptTokens,
      completionTokens: 0,
      formatMs,
      tokenizeMs,
      generateMs: 0,
      decodeMs: 0,
      totalMs: elapsedMs(started),
      memoryBefore,
      memoryAfterTokenize,
      memoryAfterGenerate: memoryAfterTokenize,
      memoryAfterDecode: memoryAfterTokenize,
      estimatedFullLogitsMb: estimateFullLogitsMb(models, promptTokens),
      estimatedAttentionScoresMb: estimateAttentionScoresMb(models, promptTokens),
      numLogitsToKeepInput: models.generationDiagnostics.numLogitsToKeepInput,
      numLogitsToKeepPatchedSessions: models.generationDiagnostics.numLogitsToKeepPatchedSessions,
      failedStage: "tokenize",
      errorMessage: message,
    };
    logGenerationProfile(profile);
    throw new GenerationExecutionError(new Error(message), profile, 400);
  }
  const effectiveGenOpts = models.maxContextLength
    ? {
        ...genOpts,
        max_new_tokens: Math.min(
          genOpts.max_new_tokens,
          Math.max(1, models.maxContextLength - promptTokens),
        ),
      }
    : genOpts;

  const streamer = new TextStreamer(
    models.tokenizer as unknown as ConstructorParameters<typeof TextStreamer>[0],
    {
      skip_prompt: true,
      callback_function: (token: string) => {
        completionTokens++;
        onToken(token);
      },
    },
  );

  const generateStart = nowMs();
  try {
    await models.model!.generate({ ...inputs, ...effectiveGenOpts, streamer });
  } catch (error) {
    if (error instanceof GenerationExecutionError) throw error;
    const memoryAfterGenerate = memorySnapshot();
    const profile: GenerationProfile = {
      path: "stream",
      promptChars: prompt.length,
      toolsCount: tools?.length ?? 0,
      toolsChars: serializedToolsChars(tools),
      promptTokens,
      completionTokens,
      formatMs,
      tokenizeMs,
      generateMs: elapsedMs(generateStart),
      decodeMs: 0,
      totalMs: elapsedMs(started),
      memoryBefore,
      memoryAfterTokenize,
      memoryAfterGenerate,
      memoryAfterDecode: memoryAfterGenerate,
      estimatedFullLogitsMb: estimateFullLogitsMb(models, promptTokens),
      estimatedAttentionScoresMb: estimateAttentionScoresMb(models, promptTokens),
      numLogitsToKeepInput: models.generationDiagnostics.numLogitsToKeepInput,
      numLogitsToKeepPatchedSessions: models.generationDiagnostics.numLogitsToKeepPatchedSessions,
      failedStage: "generate",
      errorMessage: error instanceof Error ? error.message : String(error),
    };
    logGenerationProfile(profile);
    throw new GenerationExecutionError(error, profile);
  }
  const generateMs = elapsedMs(generateStart);
  const memoryAfterGenerate = memorySnapshot();

  const profile: GenerationProfile = {
    path: "stream",
    promptChars: prompt.length,
    toolsCount: tools?.length ?? 0,
    toolsChars: serializedToolsChars(tools),
    promptTokens,
    completionTokens,
    formatMs,
    tokenizeMs,
    generateMs,
    decodeMs: 0,
    totalMs: elapsedMs(started),
    memoryBefore,
    memoryAfterTokenize,
    memoryAfterGenerate,
    memoryAfterDecode: memoryAfterGenerate,
    estimatedFullLogitsMb: estimateFullLogitsMb(models, promptTokens),
    estimatedAttentionScoresMb: estimateAttentionScoresMb(models, promptTokens),
    numLogitsToKeepInput: models.generationDiagnostics.numLogitsToKeepInput,
    numLogitsToKeepPatchedSessions: models.generationDiagnostics.numLogitsToKeepPatchedSessions,
  };
  logGenerationProfile(profile);
  return { promptTokens, completionTokens, profile };
}
