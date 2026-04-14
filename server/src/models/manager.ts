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
  tokenizer: Tokenizer;
  chatTemplate: string | null;
  processor: unknown | null;
  isVision: boolean;
  model: {
    generate(opts: Record<string, unknown>): Promise<{ dims: number[]; slice(...args: unknown[]): unknown }>;
    dispose?(): Promise<void>;
  };
  transcriber: ((input: Float32Array) => Promise<{ text: string }>) | null;
  embedder: ((input: string, opts: Record<string, unknown>) => Promise<{ data: Float32Array }>) | null;
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

export async function loadModels(config: ServerConfig): Promise<LoadedModels> {
  // Configure transformers.js environment
  if (config.cacheDir) {
    transformersEnv.cacheDir = config.cacheDir;
  }
  if (config.hfToken) {
    process.env.HF_TOKEN = config.hfToken;
  }

  const device = config.device || "auto";

  console.log(`[wandler] Loading LLM: ${config.modelId} (${config.modelDtype}, device=${device})`);
  const t0 = Date.now();

  // Try loading as a vision model first (AutoModelForImageTextToText),
  // fall back to text-only (AutoModelForCausalLM)
  let model: LoadedModels["model"];
  let processor: unknown | null = null;
  let isVision = false;

  try {
    const { AutoModelForImageTextToText: VisionModel } = await import("@huggingface/transformers");
    model = await VisionModel.from_pretrained(config.modelId, {
      dtype: config.modelDtype as "q4" | "q8" | "fp16" | "fp32",
      device,
    }) as unknown as LoadedModels["model"];

    // If vision model loaded, also load the processor
    processor = await AutoProcessor.from_pretrained(config.modelId);
    isVision = true;
    console.log(`[wandler] Loaded as vision model`);
  } catch {
    // Not a vision model or vision files not available — load as text-only
    model = await AutoModelForCausalLM.from_pretrained(config.modelId, {
      dtype: config.modelDtype as "q4" | "q8" | "fp16" | "fp32",
      device,
    }) as unknown as LoadedModels["model"];
  }

  const tokenizer = isVision && processor
    ? (processor as { tokenizer: Tokenizer }).tokenizer ?? await AutoTokenizer.from_pretrained(config.modelId) as unknown as Tokenizer
    : await AutoTokenizer.from_pretrained(config.modelId) as unknown as Tokenizer;

  const chatTemplate = await loadChatTemplate(tokenizer, config.modelId);
  console.log(`[wandler] LLM ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

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
  };
}
