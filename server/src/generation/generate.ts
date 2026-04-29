import type { LoadedModels } from "../models/manager.js";
import { loadImage } from "../models/manager.js";
import type {
  ChatMessage,
  GenerationOptions,
  GenerationResult,
  Tool,
} from "../types/openai.js";
import { formatChat } from "../models/tokenizer.js";
import { stripInternalGenOpts } from "./options.js";
import { buildPrefixCandidate, preparePrefill, type TensorLike } from "./prefill.js";
import { getImageUrls } from "../utils/content.js";
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

function promptTooLongErrorMessage(promptTokens: number, maxContextLength: number): string | null {
  if (promptTokens < maxContextLength) return null;
  return (
    `Prompt has ${promptTokens} tokens, but the model context is ${maxContextLength} tokens. ` +
    "Reduce the prompt/tools."
  );
}

/**
 * Prepare inputs for vision models: process text + images through the processor.
 */
async function prepareVisionInputs(
  models: LoadedModels,
  messages: ChatMessage[],
  _modelId: string,
  tools?: Tool[],
) {
  const proc = models.processor as {
    apply_chat_template: (msgs: unknown[], opts: Record<string, unknown>) => string;
    (textOrImages: unknown, imagesOrOpts?: unknown): Promise<Record<string, unknown>>;
    batch_decode: (ids: unknown, opts: Record<string, unknown>) => string[];
  };

  // Convert OpenAI message format to transformers.js format
  // { type: "image_url", image_url: { url } } -> { type: "image" }
  const templateMessages = messages.map((m) => {
    if (typeof m.content === "string" || m.content == null) return m;
    return {
      ...m,
      content: m.content.map((part) =>
        part.type === "image_url" ? { type: "image" as const } : part,
      ),
    };
  });

  const opts: Record<string, unknown> = {
    tokenize: false,
    add_generation_prompt: true,
  };
  if (tools?.length) opts.tools = tools;
  if (models.chatTemplate) opts.chat_template = models.chatTemplate;

  const text = proc.apply_chat_template(templateMessages, opts);

  // Collect all images from messages
  const imageUrls: string[] = [];
  for (const m of messages) {
    imageUrls.push(...getImageUrls(m.content));
  }

  const images = await Promise.all(imageUrls.map(loadImage));

  // Process text + images through the processor
  // Most processors: processor(text, images), LLaVA: processor(images, text)
  const inputs = images.length > 0
    ? await proc(text, images.length === 1 ? images[0] : images)
    : await proc(text);

  return { inputs, batchDecode: proc.batch_decode.bind(proc) };
}

export async function generate(
  models: LoadedModels,
  modelId: string,
  messages: ChatMessage[],
  genOpts: GenerationOptions,
  tools?: Tool[],
): Promise<GenerationResult> {
  const started = nowMs();
  const memoryBefore = memorySnapshot();
  const toolsCount = tools?.length ?? 0;
  const toolsChars = serializedToolsChars(tools);

  // Vision model path: use processor for text+image inputs
  if (models.isVision && models.processor) {
    const imageUrls = messages.flatMap((m) => getImageUrls(m.content));
    if (imageUrls.length > 0) {
      const formatStart = nowMs();
      const { inputs, batchDecode } = await prepareVisionInputs(models, messages, modelId, tools);
      const formatMs = elapsedMs(formatStart);
      const inputIds = inputs.input_ids as { dims: number[] };
      const memoryAfterTokenize = memorySnapshot();
      const generateStart = nowMs();
      let outputIds: { dims: number[]; slice(...args: unknown[]): unknown };
      try {
        outputIds = await models.model!.generate({ ...inputs, ...stripInternalGenOpts(genOpts) });
      } catch (error) {
        const promptTokens = inputIds.dims[1]!;
        const memoryAfterGenerate = memorySnapshot();
        const profile = {
          path: "vision" as const,
          promptChars: 0,
          toolsCount,
          toolsChars,
          promptTokens,
          completionTokens: 0,
          formatMs,
          tokenizeMs: 0,
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
          failedStage: "generate" as const,
          errorMessage: error instanceof Error ? error.message : String(error),
        };
        logGenerationProfile(profile);
        throw new GenerationExecutionError(error, profile);
      }
      const generateMs = elapsedMs(generateStart);
      const memoryAfterGenerate = memorySnapshot();

      const promptTokens = inputIds.dims[1]!;
      const completionTokens = outputIds.dims[1]! - promptTokens;
      const decodeStart = nowMs();
      const newIds = outputIds.slice(null, [promptTokens, null]);
      const text = batchDecode(newIds, { skip_special_tokens: true })[0]!;
      const decodeMs = elapsedMs(decodeStart);
      const profile = {
        path: "vision" as const,
        promptChars: 0,
        toolsCount,
        toolsChars,
        promptTokens,
        completionTokens,
        formatMs,
        tokenizeMs: 0,
        generateMs,
        decodeMs,
        totalMs: elapsedMs(started),
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
  }

  // Text-only path (default)
  const formatStart = nowMs();
  const prompt = formatChat(models.tokenizer!, messages, modelId, tools, models.chatTemplate);
  const formatMs = elapsedMs(formatStart);
  const tokenizeStart = nowMs();
  const inputs = models.tokenizer!(prompt, { return_tensors: "pt" });
  const tokenizeMs = elapsedMs(tokenizeStart);
  const memoryAfterTokenize = memorySnapshot();
  const generateStart = nowMs();
  let outputIds: { dims: number[]; slice(...args: unknown[]): unknown };
  let prefillChunkSize: number | undefined;
  let prefillChunks: number | undefined;
  let prefillMs: number | undefined;
  let prefixCacheHit: boolean | undefined;
  let prefixCacheTokens: number | undefined;
  let prefillCleanup: (() => Promise<void>) | null = null;
  try {
    const promptTokens = inputs.input_ids.dims[1]!;
    const contextError = models.maxContextLength
      ? promptTooLongErrorMessage(promptTokens, models.maxContextLength)
      : null;
    if (contextError) {
      const profile = {
        path: "text" as const,
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
        estimatedFullLogitsMb: estimateFullLogitsMb(models, promptTokens),
        estimatedAttentionScoresMb: estimateAttentionScoresMb(models, promptTokens),
        numLogitsToKeepInput: models.generationDiagnostics.numLogitsToKeepInput,
        numLogitsToKeepPatchedSessions: models.generationDiagnostics.numLogitsToKeepPatchedSessions,
        failedStage: "tokenize" as const,
        errorMessage: contextError,
      };
      logGenerationProfile(profile);
      throw new GenerationExecutionError(new Error(contextError), profile, 400);
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
    const transformersGenOpts = stripInternalGenOpts(effectiveGenOpts);
    const prefixCandidate = buildPrefixCandidate(
      models.tokenizer!,
      messages,
      modelId,
      tools,
      models.chatTemplate,
      prompt,
    );
    const prefill = await preparePrefill(
      models,
      inputs.input_ids as TensorLike,
      promptTokens,
      effectiveGenOpts,
      prompt,
      prefixCandidate,
    );
    prefillCleanup = prefill.cleanup;
    prefillChunkSize = prefill.prefillChunkSize;
    prefillChunks = prefill.prefillChunks;
    prefillMs = prefill.prefillMs;
    prefixCacheHit = prefill.prefixCacheHit;
    prefixCacheTokens = prefill.prefixCacheTokens;

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
    if (error instanceof GenerationExecutionError) {
      await prefillCleanup?.();
      throw error;
    }
    const promptTokens = inputs.input_ids.dims[1]!;
    const memoryAfterGenerate = memorySnapshot();
    const profile = {
      path: "text" as const,
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
      prefillChunkSize,
      prefillChunks,
      prefillMs,
      prefixCacheHit,
      prefixCacheTokens,
      memoryBefore,
      memoryAfterTokenize,
      memoryAfterGenerate,
      memoryAfterDecode: memoryAfterGenerate,
      estimatedFullLogitsMb: estimateFullLogitsMb(models, promptTokens),
      estimatedAttentionScoresMb: estimateAttentionScoresMb(models, promptTokens),
      numLogitsToKeepInput: models.generationDiagnostics.numLogitsToKeepInput,
      numLogitsToKeepPatchedSessions: models.generationDiagnostics.numLogitsToKeepPatchedSessions,
      failedStage: "generate" as const,
      errorMessage: error instanceof Error ? error.message : String(error),
    };
    logGenerationProfile(profile);
    await prefillCleanup?.();
    throw new GenerationExecutionError(error, profile);
  }
  const generateMs = elapsedMs(generateStart);
  const memoryAfterGenerate = memorySnapshot();

  const promptTokens = inputs.input_ids.dims[1]!;
  const chunkedPromptOffset = prefillChunkSize ? 1 : promptTokens;
  const completionTokens = outputIds.dims[1]! - chunkedPromptOffset;
  const decodeStart = nowMs();
  const newIds = outputIds.slice(null, [chunkedPromptOffset, null]);
  const text = models.tokenizer!.batch_decode(newIds, { skip_special_tokens: true })[0]!;
  const decodeMs = elapsedMs(decodeStart);
  await prefillCleanup?.();
  const profile = {
    path: "text" as const,
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
    prefillChunkSize,
    prefillChunks,
    prefillMs,
    prefixCacheHit,
    prefixCacheTokens,
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
