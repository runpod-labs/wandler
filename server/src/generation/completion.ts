import { TextStreamer } from "@huggingface/transformers";
import type { LoadedModels } from "../models/manager.js";
import type { GenerationOptions, GenerationProfile, GenerationResult } from "../types/openai.js";
import { stripInternalGenOpts } from "./options.js";
import { preparePrefill, type TensorLike } from "./prefill.js";
import {
  elapsedMs,
  estimateAttentionScoresMb,
  estimateFullLogitsMb,
  GenerationExecutionError,
  logGenerationProfile,
  memorySnapshot,
  nowMs,
} from "./profile.js";

function promptTooLongErrorMessage(promptTokens: number, maxContextLength: number): string | null {
  if (promptTokens < maxContextLength) return null;
  return (
    `Prompt has ${promptTokens} tokens, but the model context is ${maxContextLength} tokens. ` +
    "Reduce the prompt."
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

export async function generateCompletion(
  models: LoadedModels,
  prompt: string,
  genOpts: GenerationOptions,
): Promise<GenerationResult> {
  const started = nowMs();
  const memoryBefore = memorySnapshot();
  const tokenizeStart = nowMs();
  const inputs = models.tokenizer!(prompt, { return_tensors: "pt" });
  const tokenizeMs = elapsedMs(tokenizeStart);
  const memoryAfterTokenize = memorySnapshot();
  const promptTokens = inputs.input_ids.dims[1]!;
  const contextError = models.maxContextLength
    ? promptTooLongErrorMessage(promptTokens, models.maxContextLength)
    : null;
  if (contextError) {
    const profile: GenerationProfile = {
      path: "text",
      promptChars: prompt.length,
      toolsCount: 0,
      toolsChars: 0,
      promptTokens,
      completionTokens: 0,
      formatMs: 0,
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
      errorMessage: contextError,
    };
    logGenerationProfile(profile);
    throw new GenerationExecutionError(new Error(contextError), profile, 400);
  }

  const effectiveGenOpts = capGenOpts(models, promptTokens, genOpts);
  const transformersGenOpts = stripInternalGenOpts(effectiveGenOpts);
  const generateStart = nowMs();
  let outputIds: { dims: number[]; slice(...args: unknown[]): unknown };
  let prefill = await preparePrefill(models, inputs.input_ids as TensorLike, promptTokens, effectiveGenOpts);
  try {
    if (prefill.pastKeyValues) {
      outputIds = await models.model!.generate({
        input_ids: prefill.inputIds,
        past_key_values: prefill.pastKeyValues,
        ...transformersGenOpts,
      });
    } else {
      outputIds = await models.model!.generate({ ...inputs, ...transformersGenOpts });
    }
  } catch (error) {
    await prefill.cleanup();
    const memoryAfterGenerate = memorySnapshot();
    const profile: GenerationProfile = {
      path: "text",
      promptChars: prompt.length,
      toolsCount: 0,
      toolsChars: 0,
      promptTokens,
      completionTokens: 0,
      formatMs: 0,
      tokenizeMs,
      generateMs: elapsedMs(generateStart),
      decodeMs: 0,
      totalMs: elapsedMs(started),
      prefillChunkSize: prefill.prefillChunkSize,
      prefillChunks: prefill.prefillChunks,
      prefillMs: prefill.prefillMs,
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

  const promptOffset = prefill.prefillChunkSize ? 1 : promptTokens;
  const completionTokens = outputIds.dims[1]! - promptOffset;
  const decodeStart = nowMs();
  const newIds = outputIds.slice(null, [promptOffset, null]);
  const text = models.tokenizer!.batch_decode(newIds, { skip_special_tokens: true })[0]!;
  const decodeMs = elapsedMs(decodeStart);
  await prefill.cleanup();

  const profile: GenerationProfile = {
    path: "text",
    promptChars: prompt.length,
    toolsCount: 0,
    toolsChars: 0,
    promptTokens,
    completionTokens,
    formatMs: 0,
    tokenizeMs,
    generateMs,
    decodeMs,
    totalMs: elapsedMs(started),
    prefillChunkSize: prefill.prefillChunkSize,
    prefillChunks: prefill.prefillChunks,
    prefillMs: prefill.prefillMs,
    memoryBefore,
    memoryAfterTokenize,
    memoryAfterGenerate,
    memoryAfterDecode: memorySnapshot(),
    estimatedFullLogitsMb: estimateFullLogitsMb(models, promptTokens),
    estimatedAttentionScoresMb: estimateAttentionScoresMb(models, promptTokens),
    numLogitsToKeepInput: models.generationDiagnostics.numLogitsToKeepInput,
    numLogitsToKeepPatchedSessions: models.generationDiagnostics.numLogitsToKeepPatchedSessions,
  };
  logGenerationProfile(profile);
  return { text, promptTokens, completionTokens, profile };
}

export async function streamCompletionTokens(
  models: LoadedModels,
  prompt: string,
  genOpts: GenerationOptions,
  onToken: (token: string) => void | Promise<void>,
): Promise<{ promptTokens: number; completionTokens: number; profile: GenerationProfile }> {
  const started = nowMs();
  const memoryBefore = memorySnapshot();
  const tokenizeStart = nowMs();
  const inputs = models.tokenizer!(prompt, { return_tensors: "pt" });
  const tokenizeMs = elapsedMs(tokenizeStart);
  const memoryAfterTokenize = memorySnapshot();
  const promptTokens = inputs.input_ids.dims[1]!;
  const effectiveGenOpts = capGenOpts(models, promptTokens, genOpts);
  const transformersGenOpts = stripInternalGenOpts(effectiveGenOpts);
  let completionTokens = 0;
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
  const prefill = await preparePrefill(models, inputs.input_ids as TensorLike, promptTokens, effectiveGenOpts);
  try {
    if (prefill.pastKeyValues) {
      await models.model!.generate({
        input_ids: prefill.inputIds,
        past_key_values: prefill.pastKeyValues,
        ...transformersGenOpts,
        streamer,
      });
    } else {
      await models.model!.generate({ ...inputs, ...transformersGenOpts, streamer });
    }
  } finally {
    await prefill.cleanup();
  }
  const memoryAfterGenerate = memorySnapshot();
  const profile: GenerationProfile = {
    path: "stream",
    promptChars: prompt.length,
    toolsCount: 0,
    toolsChars: 0,
    promptTokens,
    completionTokens,
    formatMs: 0,
    tokenizeMs,
    generateMs: elapsedMs(generateStart),
    decodeMs: 0,
    totalMs: elapsedMs(started),
    prefillChunkSize: prefill.prefillChunkSize,
    prefillChunks: prefill.prefillChunks,
    prefillMs: prefill.prefillMs,
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
