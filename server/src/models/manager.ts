import {
  AutoTokenizer,
  AutoModelForCausalLM,
  AutoProcessor,
  RawImage,
  pipeline,
  env as transformersEnv,
} from "@huggingface/transformers";
import type { ServerConfig } from "../config.js";
import type { Tokenizer } from "./tokenizer.js";

// ── Model manager — loads and holds references to models ────────────────────

export interface LoadedModels {
  tokenizer: Tokenizer | null;
  chatTemplate: string | null;
  processor: unknown | null;
  isVision: boolean;
  model: {
    generate(opts: Record<string, unknown>): Promise<{ dims: number[]; slice(...args: unknown[]): unknown }>;
    dispose?(): Promise<void>;
  } | null;
  transcriber: ((input: Float32Array) => Promise<{ text: string }>) | null;
  embedder: ((input: string, opts: Record<string, unknown>) => Promise<{ data: Float32Array }>) | null;
  /**
   * Max context length (`max_position_embeddings`) read from the loaded LLM's
   * config at startup. Used as the default max-tokens ceiling when the
   * operator hasn't set `--max-tokens` explicitly. `null` when no LLM is
   * loaded or the model doesn't expose the field.
   */
  maxContextLength: number | null;
}

/**
 * Read `max_position_embeddings` from the loaded model's config. Vision/
 * multimodal models nest it under `text_config`; text-only models expose it
 * at the top level. Returns `null` when the field is missing or non-numeric.
 */
function readMaxContextLength(
  model: unknown,
): number | null {
  const config = (model as { config?: Record<string, unknown> }).config;
  if (!config) return null;
  const textConfig = (config.text_config as Record<string, unknown> | undefined) ?? config;
  const mpe = textConfig.max_position_embeddings;
  return typeof mpe === "number" && mpe > 0 ? mpe : null;
}

/**
 * Try to load chat_template.jinja from the model repo if the tokenizer
 * doesn't have a built-in chat template.
 */
async function loadChatTemplate(
  tokenizer: Tokenizer,
  modelId: string,
): Promise<string | null> {
  const tok = tokenizer as unknown as { chat_template?: string | null };
  if (tok.chat_template) return null;

  try {
    const url = `https://huggingface.co/${modelId}/resolve/main/chat_template.jinja`;
    const res = await fetch(url);
    if (res.ok) {
      const template = await res.text();
      console.log(`[wandler] Loaded chat_template.jinja for ${modelId}`);
      return template;
    }
  } catch {
    // Failed to fetch
  }

  return null;
}

/**
 * Load an image from a URL or base64 data URI.
 */
export async function loadImage(url: string): Promise<unknown> {
  if (url.startsWith("data:")) {
    // base64 data URI: data:image/jpeg;base64,/9j/4AAQ...
    const base64Match = url.match(/^data:[^;]+;base64,(.+)$/);
    if (!base64Match) throw new Error("Invalid base64 data URI");
    const buffer = Buffer.from(base64Match[1]!, "base64");
    const blob = new Blob([buffer]);
    return RawImage.fromBlob(blob);
  }
  // HTTP(S) URL
  return RawImage.read(url);
}

type DeviceType = "auto" | "cpu" | "cuda" | "coreml" | "dml" | "webgpu" | "wasm";
type DtypeType = "q4" | "q8" | "fp16" | "fp32";

// All device types in preference order (best perf first, cpu last).
// onnxruntime may crash instead of falling back between providers
// (e.g. CUDA missing libcudnn, Vulkan unavailable in Docker), so when
// device="auto" fails we walk this list ourselves.  Devices that don't
// apply to the current platform fail instantly via transformers.js
// ("Unsupported device") — no model loading, no GPU probing.
//
// Ideally we'd import this from @huggingface/transformers, but
// deviceToExecutionProviders / supportedDevices aren't exported.
// See: https://github.com/huggingface/transformers.js/issues/1645
// Source: node_modules/@huggingface/transformers/src/backends/onnx.js
const DEVICE_FALLBACK_ORDER: DeviceType[] = ["cuda", "coreml", "dml", "webgpu", "cpu"];

/**
 * Load the model, walking the device fallback chain when needed.
 * For device="auto": try each device individually until one works.
 * For an explicit device: try it, then fall back to cpu.
 */
async function loadLLM(
  modelId: string,
  dtype: string,
  device: string,
): Promise<{ model: LoadedModels["model"]; processor: unknown | null; isVision: boolean }> {
  // Explicit device (not "auto") — try it, fall back to cpu.
  if (device !== "auto") {
    try {
      return await loadLLMWithDevice(modelId, dtype, device as DeviceType);
    } catch (err) {
      if (device !== "cpu") {
        console.warn(
          `[wandler] device=${device} failed: ${err instanceof Error ? err.message : err}`,
        );
        console.warn(`[wandler] Falling back to device=cpu`);
        return await loadLLMWithDevice(modelId, dtype, "cpu");
      }
      throw err;
    }
  }

  // device="auto" — walk the fallback chain.
  for (let i = 0; i < DEVICE_FALLBACK_ORDER.length; i++) {
    const dev = DEVICE_FALLBACK_ORDER[i]!;
    try {
      return await loadLLMWithDevice(modelId, dtype, dev);
    } catch (err) {
      const isLast = i === DEVICE_FALLBACK_ORDER.length - 1;
      if (isLast) throw err;
      console.warn(
        `[wandler] device=${dev} failed: ${err instanceof Error ? err.message : err}`,
      );
    }
  }

  // Unreachable — chain always ends with "cpu", but satisfy TS.
  throw new Error("No usable device found");
}

async function loadLLMWithDevice(
  modelId: string,
  dtype: string,
  device: DeviceType,
): Promise<{ model: LoadedModels["model"]; processor: unknown | null; isVision: boolean }> {
  // Try loading as a vision model first (AutoModelForImageTextToText),
  // fall back to text-only (AutoModelForCausalLM)
  try {
    const { AutoModelForImageTextToText: VisionModel } = await import("@huggingface/transformers");
    const model = await VisionModel.from_pretrained(modelId, {
      dtype: dtype as DtypeType,
      device,
    }) as unknown as LoadedModels["model"];

    const processor = await AutoProcessor.from_pretrained(modelId);
    console.log(`[wandler] Loaded as vision model (device=${device})`);
    return { model, processor, isVision: true };
  } catch {
    // Not a vision model or vision files not available — load as text-only
    const model = await AutoModelForCausalLM.from_pretrained(modelId, {
      dtype: dtype as DtypeType,
      device,
    }) as unknown as LoadedModels["model"];
    return { model, processor: null, isVision: false };
  }
}

export async function loadModels(config: ServerConfig): Promise<LoadedModels> {
  // Configure transformers.js environment
  if (config.cacheDir) {
    transformersEnv.cacheDir = config.cacheDir;
  }
  if (config.hfToken) {
    process.env.HF_TOKEN = config.hfToken;
  }

  const device = config.device || "auto";

  let model: LoadedModels["model"] = null;
  let tokenizer: Tokenizer | null = null;
  let processor: unknown | null = null;
  let isVision = false;
  let chatTemplate: string | null = null;
  let maxContextLength: number | null = null;

  if (config.modelId) {
    console.log(`[wandler] Loading LLM: ${config.modelId} (${config.modelDtype}, device=${device})`);
    const t0 = Date.now();

    const result = await loadLLM(config.modelId, config.modelDtype, device);
    model = result.model;
    processor = result.processor;
    isVision = result.isVision;

    tokenizer = isVision && processor
      ? (processor as { tokenizer: Tokenizer }).tokenizer ?? await AutoTokenizer.from_pretrained(config.modelId) as unknown as Tokenizer
      : await AutoTokenizer.from_pretrained(config.modelId) as unknown as Tokenizer;

    chatTemplate = await loadChatTemplate(tokenizer, config.modelId);
    maxContextLength = readMaxContextLength(model);
    if (maxContextLength) {
      console.log(`[wandler] Model context: ${maxContextLength} tokens`);
    }
    console.log(`[wandler] LLM ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  }

  let transcriber: LoadedModels["transcriber"] = null;
  if (config.sttModelId) {
    console.log(`[wandler] Loading STT: ${config.sttModelId} (${config.sttDtype})`);
    const t1 = Date.now();
    const sttPipeline = await pipeline("automatic-speech-recognition", config.sttModelId, {
      dtype: config.sttDtype as "q4" | "q8" | "fp16" | "fp32",
    });
    transcriber = (input: Float32Array) =>
      sttPipeline(input) as Promise<{ text: string }>;
    console.log(`[wandler] STT ready in ${((Date.now() - t1) / 1000).toFixed(1)}s`);
  }

  let embedder: LoadedModels["embedder"] = null;
  if (config.embeddingModelId) {
    console.log(`[wandler] Loading embeddings: ${config.embeddingModelId} (${config.embeddingDtype})`);
    const t2 = Date.now();
    const embPipeline = await pipeline("feature-extraction", config.embeddingModelId, {
      dtype: config.embeddingDtype as "q4" | "q8" | "fp16" | "fp32",
    });
    embedder = (input: string, opts: Record<string, unknown>) =>
      embPipeline(input, opts) as Promise<{ data: Float32Array }>;
    console.log(`[wandler] Embeddings ready in ${((Date.now() - t2) / 1000).toFixed(1)}s`);
  }

  return {
    tokenizer,
    chatTemplate,
    processor,
    isVision,
    model,
    transcriber,
    embedder,
    maxContextLength,
  };
}
