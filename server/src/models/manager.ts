import {
  AutoTokenizer,
  AutoModelForCausalLM,
  pipeline,
  env as transformersEnv,
} from "@huggingface/transformers";
import type { ServerConfig } from "../config.js";
import type { Tokenizer } from "./tokenizer.js";

// ── Model manager — loads and holds references to models ────────────────────

export interface LoadedModels {
  tokenizer: Tokenizer;
  model: {
    generate(opts: Record<string, unknown>): Promise<{ dims: number[]; slice(...args: unknown[]): unknown }>;
    dispose?(): Promise<void>;
  };
  transcriber: ((input: Float32Array) => Promise<{ text: string }>) | null;
  embedder: ((input: string, opts: Record<string, unknown>) => Promise<{ data: Float32Array }>) | null;
}

type DeviceType = "webgpu" | "cpu" | "wasm";

/**
 * Resolve "auto" device to the best available: webgpu > cpu.
 * If a specific device is requested, use it as-is.
 */
async function resolveDevice(requested: string): Promise<DeviceType> {
  if (requested !== "auto") return requested as DeviceType;

  try {
    // Try to get a WebGPU adapter — if it works, WebGPU is available
    const gpu = (globalThis as Record<string, unknown>).gpu as { requestAdapter?: () => Promise<unknown> } | undefined;
    if (gpu?.requestAdapter) {
      const adapter = await gpu.requestAdapter();
      if (adapter) return "webgpu";
    }
  } catch {
    // WebGPU not available
  }

  console.log("[wandler] WebGPU not available, falling back to cpu");
  return "cpu";
}

export async function loadModels(config: ServerConfig): Promise<LoadedModels> {
  // Configure transformers.js environment
  if (config.cacheDir) {
    transformersEnv.cacheDir = config.cacheDir;
  }
  // HF_TOKEN is read automatically by @huggingface/hub (used internally by transformers.js)
  if (config.hfToken) {
    process.env.HF_TOKEN = config.hfToken;
  }

  const device = await resolveDevice(config.device);
  // Update config so other parts of the server know the resolved device
  config.device = device;

  console.log(`[wandler] Loading LLM: ${config.modelId} (${config.modelDtype}, ${device})`);
  const t0 = Date.now();
  const tokenizer = await AutoTokenizer.from_pretrained(config.modelId) as unknown as Tokenizer;
  const model = await AutoModelForCausalLM.from_pretrained(config.modelId, {
    dtype: config.modelDtype as "q4" | "q8" | "fp16" | "fp32",
    device,
  });
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
    model: model as unknown as LoadedModels["model"],
    transcriber,
    embedder,
  };
}
