import { TextStreamer } from "@huggingface/transformers";
import type { LoadedModels } from "../models/manager.js";
import { formatChat } from "../models/tokenizer.js";
import type { ChatMessage, GenerationOptions, GenerationProfile, Tool } from "../types/openai.js";
import { parseToolCalls } from "../tools/parser.js";
import { stripInternalGenOpts } from "../generation/options.js";
import {
  elapsedMs,
  estimateAttentionScoresMb,
  estimateFullLogitsMb,
  GenerationExecutionError,
  logGenerationProfile,
  memorySnapshot,
  nowMs,
  serializedToolsChars,
} from "../generation/profile.js";
import type { LLMBackend, StreamToolHandlers } from "./types.js";

function promptTooLongErrorMessage(promptTokens: number, maxContextLength: number): string | null {
  if (promptTokens < maxContextLength) return null;
  return (
    `Prompt has ${promptTokens} tokens, but the model context is ${maxContextLength} tokens. ` +
    "Reduce the prompt/tools."
  );
}

function capGenOpts(models: LoadedModels, promptTokens: number, genOpts: GenerationOptions): GenerationOptions {
  return models.maxContextLength
    ? {
        ...genOpts,
        max_new_tokens: Math.min(
          genOpts.max_new_tokens,
          Math.max(1, models.maxContextLength - promptTokens),
        ),
      }
    : genOpts;
}

export class TransformersJsBackend implements LLMBackend {
  readonly name = "transformersjs" as const;

  constructor(readonly models: LoadedModels) {}

  async generateChat(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    tools?: Tool[],
  ) {
    const started = nowMs();
    const memoryBefore = memorySnapshot();
    const toolsCount = tools?.length ?? 0;
    const toolsChars = serializedToolsChars(tools);
    const formatStart = nowMs();
    const prompt = formatChat(this.models.tokenizer!, messages, modelId, tools, this.models.chatTemplate);
    const formatMs = elapsedMs(formatStart);
    const tokenizeStart = nowMs();
    const inputs = this.models.tokenizer!(prompt, { return_tensors: "pt" });
    const tokenizeMs = elapsedMs(tokenizeStart);
    const memoryAfterTokenize = memorySnapshot();
    const promptTokens = inputs.input_ids.dims[1]!;
    const contextError = this.models.maxContextLength
      ? promptTooLongErrorMessage(promptTokens, this.models.maxContextLength)
      : null;
    if (contextError) {
      const profile: GenerationProfile = {
        path: "text",
        promptChars: prompt.length,
        toolsCount,
        toolsChars,
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
        estimatedFullLogitsMb: estimateFullLogitsMb(this.models, promptTokens),
        estimatedAttentionScoresMb: estimateAttentionScoresMb(this.models, promptTokens),
        numLogitsToKeepInput: this.models.generationDiagnostics.numLogitsToKeepInput,
        numLogitsToKeepPatchedSessions: this.models.generationDiagnostics.numLogitsToKeepPatchedSessions,
        failedStage: "tokenize",
        errorMessage: contextError,
      };
      logGenerationProfile(profile);
      throw new GenerationExecutionError(new Error(contextError), profile, 400);
    }

    const generateStart = nowMs();
    let outputIds: { dims: number[]; slice(...args: unknown[]): unknown };
    try {
      outputIds = await this.models.model!.generate({
        ...inputs,
        ...stripInternalGenOpts(capGenOpts(this.models, promptTokens, genOpts)),
      });
    } catch (error) {
      const memoryAfterGenerate = memorySnapshot();
      const profile: GenerationProfile = {
        path: "text",
        promptChars: prompt.length,
        toolsCount,
        toolsChars,
        promptTokens,
        completionTokens: 0,
        formatMs,
        tokenizeMs,
        generateMs: elapsedMs(generateStart),
        decodeMs: 0,
        totalMs: elapsedMs(started),
        memoryBefore,
        memoryAfterTokenize,
        memoryAfterGenerate,
        memoryAfterDecode: memoryAfterGenerate,
        estimatedFullLogitsMb: estimateFullLogitsMb(this.models, promptTokens),
        estimatedAttentionScoresMb: estimateAttentionScoresMb(this.models, promptTokens),
        numLogitsToKeepInput: this.models.generationDiagnostics.numLogitsToKeepInput,
        numLogitsToKeepPatchedSessions: this.models.generationDiagnostics.numLogitsToKeepPatchedSessions,
        failedStage: "generate",
        errorMessage: error instanceof Error ? error.message : String(error),
      };
      logGenerationProfile(profile);
      throw new GenerationExecutionError(error, profile);
    }

    const generateMs = elapsedMs(generateStart);
    const memoryAfterGenerate = memorySnapshot();
    const completionTokens = outputIds.dims[1]! - promptTokens;
    const decodeStart = nowMs();
    const newIds = outputIds.slice(null, [promptTokens, null]);
    const text = this.models.tokenizer!.batch_decode(newIds, { skip_special_tokens: true })[0]!;
    const decodeMs = elapsedMs(decodeStart);
    const profile: GenerationProfile = {
      path: "text",
      promptChars: prompt.length,
      toolsCount,
      toolsChars,
      promptTokens,
      completionTokens,
      formatMs,
      tokenizeMs,
      generateMs,
      decodeMs,
      totalMs: elapsedMs(started),
      memoryBefore,
      memoryAfterTokenize,
      memoryAfterGenerate,
      memoryAfterDecode: memorySnapshot(),
      estimatedFullLogitsMb: estimateFullLogitsMb(this.models, promptTokens),
      estimatedAttentionScoresMb: estimateAttentionScoresMb(this.models, promptTokens),
      numLogitsToKeepInput: this.models.generationDiagnostics.numLogitsToKeepInput,
      numLogitsToKeepPatchedSessions: this.models.generationDiagnostics.numLogitsToKeepPatchedSessions,
    };
    logGenerationProfile(profile);
    return { text, promptTokens, completionTokens, profile };
  }

  async streamChat(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    onToken: (token: string) => void | Promise<void>,
    tools?: Tool[],
  ): Promise<{ promptTokens: number; completionTokens: number; profile?: GenerationProfile }> {
    const started = nowMs();
    const memoryBefore = memorySnapshot();
    const formatStart = nowMs();
    const prompt = formatChat(this.models.tokenizer!, messages, modelId, tools, this.models.chatTemplate);
    const formatMs = elapsedMs(formatStart);
    const tokenizeStart = nowMs();
    const inputs = this.models.tokenizer!(prompt, { return_tensors: "pt" });
    const tokenizeMs = elapsedMs(tokenizeStart);
    const memoryAfterTokenize = memorySnapshot();
    const promptTokens = inputs.input_ids.dims[1]!;
    let completionTokens = 0;

    const streamer = new TextStreamer(
      this.models.tokenizer as unknown as ConstructorParameters<typeof TextStreamer>[0],
      {
        skip_prompt: true,
        callback_function: (token: string) => {
          completionTokens++;
          onToken(token);
        },
      },
    );

    const generateStart = nowMs();
    await this.models.model!.generate({
      ...inputs,
      ...stripInternalGenOpts(capGenOpts(this.models, promptTokens, genOpts)),
      streamer,
    });
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
      estimatedFullLogitsMb: estimateFullLogitsMb(this.models, promptTokens),
      estimatedAttentionScoresMb: estimateAttentionScoresMb(this.models, promptTokens),
      numLogitsToKeepInput: this.models.generationDiagnostics.numLogitsToKeepInput,
      numLogitsToKeepPatchedSessions: this.models.generationDiagnostics.numLogitsToKeepPatchedSessions,
    };
    logGenerationProfile(profile);
    return { promptTokens, completionTokens, profile };
  }

  async streamChatWithTools(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    tools: Tool[] | undefined,
    handlers: StreamToolHandlers,
  ) {
    let buffer = "";
    const result = await this.streamChat(modelId, messages, genOpts, (token) => {
      buffer += token;
    }, tools);
    const calls = parseToolCalls(buffer);
    if (calls?.length) {
      await handlers.onToolCalls(calls);
    } else if (buffer) {
      await handlers.onContent(buffer);
    }
    return result;
  }
}
