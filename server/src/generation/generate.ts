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

type TensorLike = {
  dims: number[];
  location?: string;
  slice(...args: unknown[]): TensorLike;
  dispose?(): Promise<void>;
};

type CacheTensor = TensorLike;

class WandlerDynamicCache {
  [key: string]: CacheTensor | unknown;

  constructor(entries?: Record<string, CacheTensor>) {
    if (!entries) return;
    this.update(entries);
  }

  get_seq_length(): number {
    for (const [name, tensor] of Object.entries(this)) {
      if (name.startsWith("past_key_values.") && isTensorLike(tensor)) {
        return tensor.dims.at(-2) ?? 0;
      }
    }
    return 0;
  }

  update(entries: Record<string, CacheTensor>): void {
    for (const [name, tensor] of Object.entries(entries)) {
      const old = this[name];
      if (isTensorLike(old) && old !== tensor && old.location === "gpu-buffer") {
        void old.dispose?.();
      }
      this[name] = tensor;
    }
  }

  async dispose(): Promise<void> {
    await Promise.all(
      Object.values(this)
        .filter(isTensorLike)
        .filter((tensor) => tensor.location === "gpu-buffer")
        .map((tensor) => tensor.dispose?.() ?? Promise.resolve()),
    );
  }
}

function isTensorLike(value: unknown): value is TensorLike {
  return Boolean(
    value &&
    typeof value === "object" &&
    "dims" in value &&
    Array.isArray((value as { dims?: unknown }).dims),
  );
}

function getPastKeyValues(outputs: Record<string, unknown>, cache: WandlerDynamicCache | null): WandlerDynamicCache {
  const entries: Record<string, CacheTensor> = Object.create(null);
  for (const [name, value] of Object.entries(outputs)) {
    if (!name.startsWith("present") || !isTensorLike(value)) continue;
    const newName = name
      .replace("present_ssm", "past_ssm")
      .replace("present_conv", "past_conv")
      .replace("present_recurrent", "past_recurrent")
      .replace("present", "past_key_values");
    entries[newName] = value;
  }
  if (cache) {
    cache.update(entries);
    return cache;
  }
  return new WandlerDynamicCache(entries);
}

async function disposeUnusedOutputs(outputs: Record<string, unknown>, cache: WandlerDynamicCache): Promise<void> {
  const cached = new Set(Object.values(cache));
  await Promise.all(
    Object.values(outputs)
      .filter(isTensorLike)
      .filter((tensor) => tensor.location === "gpu-buffer" && !cached.has(tensor))
      .map((tensor) => tensor.dispose?.() ?? Promise.resolve()),
  );
}

function readPrefillChunkSize(promptTokens: number, raw = process.env.WANDLER_PREFILL_CHUNK_SIZE ?? "1024"): number | null {
  if (["0", "false", "off", "no"].includes(raw.toLowerCase())) return null;
  const chunkSize = Number.parseInt(raw, 10);
  if (!Number.isFinite(chunkSize) || chunkSize < 2 || chunkSize >= promptTokens) return null;
  return chunkSize;
}

function promptTooLongErrorMessage(promptTokens: number, maxContextLength: number): string | null {
  if (promptTokens < maxContextLength) return null;
  return (
    `Prompt has ${promptTokens} tokens, but the model context is ${maxContextLength} tokens. ` +
    "Reduce the prompt/tools."
  );
}

async function prefillPromptCache(
  model: LoadedModels["model"],
  inputIds: TensorLike,
  promptTokens: number,
  chunkSize: number,
): Promise<{ cache: WandlerDynamicCache | null; lastTokenInputIds: TensorLike; chunks: number; prefillMs: number }> {
  const m = model as unknown as {
    prepare_inputs_for_generation(inputIds: bigint[][], modelInputs: Record<string, unknown>, generationConfig: Record<string, unknown>): Record<string, unknown>;
    forward(modelInputs: Record<string, unknown>): Promise<Record<string, unknown>>;
  };

  const started = nowMs();
  let cache: WandlerDynamicCache | null = null;
  let chunks = 0;
  const prefillEnd = promptTokens - 1;

  for (let start = 0; start < prefillEnd; start += chunkSize) {
    const end = Math.min(start + chunkSize, prefillEnd);
    const chunkInputIds = inputIds.slice(null, [start, end]);
    let modelInputs: Record<string, unknown> = {
      input_ids: chunkInputIds,
      past_key_values: cache,
    };
    modelInputs = m.prepare_inputs_for_generation([], modelInputs, {});
    const outputs = await m.forward(modelInputs);
    cache = getPastKeyValues(outputs, cache);
    await disposeUnusedOutputs(outputs, cache);
    chunks++;
  }

  return {
    cache,
    lastTokenInputIds: inputIds.slice(null, [promptTokens - 1, promptTokens]),
    chunks,
    prefillMs: elapsedMs(started),
  };
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
  let chunkedCache: WandlerDynamicCache | null = null;
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
    const chunkSize = readPrefillChunkSize(promptTokens, effectiveGenOpts.prefill_chunk_size);
    const transformersGenOpts = stripInternalGenOpts(effectiveGenOpts);
    if (chunkSize) {
      const prefill = await prefillPromptCache(
        models.model,
        inputs.input_ids as TensorLike,
        promptTokens,
        chunkSize,
      );
      chunkedCache = prefill.cache;
      prefillChunkSize = chunkSize;
      prefillChunks = prefill.chunks;
      prefillMs = prefill.prefillMs;
      outputIds = await models.model!.generate({
        input_ids: prefill.lastTokenInputIds,
        past_key_values: chunkedCache,
        ...transformersGenOpts,
      });
    } else {
      outputIds = await models.model!.generate({ ...inputs, ...transformersGenOpts });
    }
  } catch (error) {
    if (error instanceof GenerationExecutionError) {
      await chunkedCache?.dispose();
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
    await chunkedCache?.dispose();
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
  await chunkedCache?.dispose();
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
