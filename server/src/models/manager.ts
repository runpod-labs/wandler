import {
  AutoTokenizer,
  AutoModelForCausalLM,
  AutoProcessor,
  LogLevel,
  ModelRegistry,
  RawImage,
  pipeline,
  env as transformersEnv,
} from "@huggingface/transformers";
import type { ServerConfig } from "../config.js";
import { logInfo, logWarn } from "../utils/logging.js";
import type { Tokenizer } from "./tokenizer.js";

// ── Model manager — loads and holds references to models ────────────────────

export interface LoadedModels {
  device: string | null;
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
  /**
   * Vocabulary size read from the loaded LLM config. Used for diagnostics to
   * estimate how large full-prompt logits would be for long contexts.
   */
  vocabSize: number | null;
  generationDiagnostics: {
    numLogitsToKeepInput: boolean;
    numLogitsToKeepPatchedSessions: string[];
  };
  attentionHeads: number | null;
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

function readVocabSize(model: unknown): number | null {
  const config = (model as { config?: Record<string, unknown> }).config;
  if (!config) return null;
  const textConfig = (config.text_config as Record<string, unknown> | undefined) ?? config;
  const vocabSize = textConfig.vocab_size;
  return typeof vocabSize === "number" && vocabSize > 0 ? vocabSize : null;
}

function readAttentionHeads(model: unknown): number | null {
  const config = (model as { config?: Record<string, unknown> }).config;
  if (!config) return null;
  const textConfig = (config.text_config as Record<string, unknown> | undefined) ?? config;
  const heads = textConfig.num_attention_heads;
  return typeof heads === "number" && heads > 0 ? heads : null;
}

type SessionLike = {
  inputNames?: string[];
  run?: (feeds: Record<string, unknown>, ...args: unknown[]) => Promise<unknown>;
  __wandlerRunPatch?: boolean;
};

type OnnxRunOptions = {
  extra?: {
    memory?: {
      enable_memory_arena_shrinkage?: "0" | "1";
    };
  };
};

function shouldShrinkOnnxMemoryArena(): boolean {
  const raw = process.env.WANDLER_ONNX_MEMORY_ARENA_SHRINKAGE;
  if (raw == null) return true;
  return !["0", "false", "off", "no"].includes(raw.toLowerCase());
}

function mergeRunOptions(args: unknown[], runOptions: OnnxRunOptions): unknown[] {
  if (args.length === 0) return [runOptions];

  // session.run(feeds, options)
  if (args.length === 1 && isPlainObject(args[0]) && !looksLikeFetches(args[0])) {
    return [mergePlainObjects(args[0], runOptions)];
  }

  // session.run(feeds, fetches, options)
  if (args.length >= 2) {
    const next = [...args];
    const existing = isPlainObject(next[1]) && !looksLikeFetches(next[1]) ? next[1] : {};
    next[1] = mergePlainObjects(existing, runOptions);
    return next;
  }

  return args;
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === "object" && !Array.isArray(value));
}

function looksLikeFetches(value: unknown): boolean {
  if (!isPlainObject(value)) return false;
  return Object.values(value).some((entry) => entry === true || isOnnxValueLike(entry));
}

function isOnnxValueLike(value: unknown): boolean {
  return Boolean(
    value &&
    typeof value === "object" &&
    "dims" in value &&
    Array.isArray((value as { dims?: unknown }).dims),
  );
}

function mergePlainObjects(base: Record<string, unknown>, patch: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...base };
  for (const [key, value] of Object.entries(patch)) {
    out[key] = isPlainObject(value) && isPlainObject(out[key])
      ? mergePlainObjects(out[key], value)
      : value;
  }
  return out;
}

function setScalarTensorValue(tensor: unknown, value: bigint): boolean {
  const t = tensor as { data?: unknown; cpuData?: unknown };
  const data = t.data ?? t.cpuData;
  if (
    data &&
    typeof data === "object" &&
    "length" in data &&
    typeof (data as { length: unknown }).length === "number" &&
    (data as { length: number }).length > 0
  ) {
    const array = data as { [index: number]: bigint | number };
    const current = array[0];
    if (current === 0n) {
      array[0] = value;
      return true;
    }
    if (current === 0) {
      array[0] = Number(value);
      return true;
    }
  }
  return false;
}

/**
 * transformers.js sets `num_logits_to_keep=1` during generation, but the
 * current Gemma3n/Gemma4 forward path drops that kwarg before `decoder_forward`.
 * If it reaches ONNX as 0, Gemma computes logits for the full prompt
 * (`prompt_tokens * vocab_size`) and long prompts can allocate tens of GB.
 *
 * Wandler only uses `generate()`, where last-token logits are sufficient, so
 * this ONNX-feed guard restores the intended generation behavior without
 * changing user prompts, messages, or tool schemas.
 */
function patchOnnxSessionRuns(model: LoadedModels["model"]): string[] {
  const sessions = (model as unknown as { sessions?: Record<string, SessionLike> })?.sessions;
  if (!sessions) return [];

  const patched: string[] = [];
  const shrinkMemoryArena = shouldShrinkOnnxMemoryArena();
  const shrinkRunOptions: OnnxRunOptions = {
    extra: { memory: { enable_memory_arena_shrinkage: "1" } },
  };

  for (const [name, session] of Object.entries(sessions)) {
    if (!session.run || session.__wandlerRunPatch) {
      continue;
    }

    const originalRun = session.run.bind(session);
    session.run = async (feeds: Record<string, unknown>, ...args: unknown[]) => {
      const valueWasPatched = session.inputNames?.includes("num_logits_to_keep")
        ? setScalarTensorValue(feeds.num_logits_to_keep, 1n)
        : false;
      if (valueWasPatched && process.env.WANDLER_LOG_LEVEL === "debug") {
        console.debug(`[wandler] Forced num_logits_to_keep=1 for session=${name}`);
      }
      const runArgs = shrinkMemoryArena ? mergeRunOptions(args, shrinkRunOptions) : args;
      return await originalRun(feeds, ...runArgs);
    };
    session.__wandlerRunPatch = true;
    if (session.inputNames?.includes("num_logits_to_keep")) {
      patched.push(name);
    }
  }

  return patched;
}

function hasNumLogitsToKeepInput(model: LoadedModels["model"]): boolean {
  const sessions = (model as unknown as { sessions?: Record<string, SessionLike> })?.sessions;
  if (!sessions) return false;
  return Object.values(sessions).some((session) => session.inputNames?.includes("num_logits_to_keep"));
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
      logInfo(`[wandler] Loaded chat_template.jinja for ${modelId}`);
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
// Full set of transformers.js v4 dtypes, including 1-bit (BitNet) and 2-bit
// formats added in v4.1.0. "auto" defers selection to the model's own config.
// Source: @huggingface/transformers DATA_TYPES constant.
export type DtypeType =
  | "auto"
  | "fp32"
  | "fp16"
  | "q8"
  | "int8"
  | "uint8"
  | "q4"
  | "bnb4"
  | "q4f16"
  | "q2"
  | "q2f16"
  | "q1"
  | "q1f16";

export const SUPPORTED_DTYPES: readonly DtypeType[] = [
  "auto",
  "fp32",
  "fp16",
  "q8",
  "int8",
  "uint8",
  "q4",
  "bnb4",
  "q4f16",
  "q2",
  "q2f16",
  "q1",
  "q1f16",
] as const;

/**
 * Dtypes that are only reliable on CPU/WebGPU execution providers today.
 * CUDA / CoreML / DML lack the custom ternary + 2-bit ONNX kernels, so we
 * warn the user instead of silently falling back and producing garbage.
 */
const LOW_BIT_DTYPES: readonly DtypeType[] = ["q1", "q1f16", "q2", "q2f16"] as const;

function buildSessionOptions(device: DeviceType): Record<string, unknown> | undefined {
  if (device !== "cuda") return undefined;

  const cudaProvider: Record<string, unknown> = {
    name: "cuda",
    // ONNX Runtime's default grows the CUDA arena by powers of two. That is
    // fast, but long prompts can leave most of the GPU reserved after a run.
    arena_extend_strategy: process.env.WANDLER_CUDA_ARENA_EXTEND_STRATEGY ?? "kSameAsRequested",
  };

  const gpuMemLimitMb = process.env.WANDLER_CUDA_GPU_MEM_LIMIT_MB;
  if (gpuMemLimitMb) {
    const mb = Number.parseInt(gpuMemLimitMb, 10);
    if (Number.isFinite(mb) && mb > 0) {
      cudaProvider.gpu_mem_limit = String(mb * 1024 * 1024);
    }
  }

  return {
    executionProviders: [cudaProvider, "cpu"],
  };
}

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
 * When the user passes dtype "auto" at the wandler level we resolve it to
 * the smallest available ONNX export for the model. Preference: tiniest
 * memory footprint first, falling through to fp32 as a last resort.
 */
const AUTO_DTYPE_PREFERENCE: DtypeType[] = [
  "q1", "q1f16", "q2", "q2f16", "q4", "q4f16", "bnb4", "q8", "int8", "uint8", "fp16", "fp32",
];

/**
 * Pre-flight dtype check via `ModelRegistry.get_available_dtypes`. Resolves
 * "auto" to the smallest available variant and validates explicit dtypes
 * against what the HF repo actually ships, so users see a useful error
 * instead of a cryptic 404 from the ONNX fetcher.
 *
 * Returns the (possibly resolved) dtype. Never throws on probe failure —
 * falls back to the requested dtype and lets the real loader surface the
 * error. This keeps offline / private-model flows working.
 */
async function resolveDtype(modelId: string, requested: string): Promise<string> {
  let available: string[];
  try {
    available = await ModelRegistry.get_available_dtypes(modelId);
  } catch (err) {
    logWarn(
      `[wandler] Could not probe dtypes for ${modelId} (${err instanceof Error ? err.message : err}); proceeding with ${requested}`,
    );
    return requested;
  }

  if (!available.length) return requested;

  if (requested === "auto") {
    const picked = AUTO_DTYPE_PREFERENCE.find((d) => available.includes(d));
    if (!picked) {
      logWarn(`[wandler] dtype=auto: no recognised dtype in ${JSON.stringify(available)}; using ${available[0]}`);
      return available[0]!;
    }
    logInfo(`[wandler] dtype=auto → ${picked} (available: ${available.join(", ")})`);
    return picked;
  }

  if (!available.includes(requested)) {
    throw new Error(
      `dtype "${requested}" is not available for ${modelId}. ` +
      `Available on the Hub: [${available.join(", ")}]. ` +
      `Retry with one of those, or use :auto to let wandler pick.`,
    );
  }
  return requested;
}

/**
 * Warn when the user pairs a 1-bit / 2-bit dtype with a device whose ONNX
 * Runtime build lacks the ternary kernels — today that's cuda/coreml/dml.
 * Our device fallback chain will silently slide to CPU in that case, which
 * is correct but surprising. A single log line flags the cost.
 */
function warnIfLowBitMismatch(dtype: string, device: string): void {
  if (!LOW_BIT_DTYPES.includes(dtype as DtypeType)) return;
  const ok = device === "auto" || device === "cpu" || device === "webgpu" || device === "wasm";
  if (ok) return;
  logWarn(
    `[wandler] dtype=${dtype} (1-bit/2-bit) is CPU/WebGPU-only today. ` +
    `device=${device} will likely fall back to cpu.`,
  );
}

/**
 * Load the model, walking the device fallback chain when needed.
 * For device="auto": try each device individually until one works.
 * For an explicit device: try it, then fall back to cpu.
 */
async function loadLLM(
  modelId: string,
  dtype: string,
  device: string,
): Promise<{ model: LoadedModels["model"]; processor: unknown | null; isVision: boolean; device: DeviceType }> {
  // Explicit device (not "auto") — try it, fall back to cpu.
  if (device !== "auto") {
    return await loadLLMWithDevice(modelId, dtype, device as DeviceType);
  }

  // device="auto" — walk the fallback chain.
  for (let i = 0; i < DEVICE_FALLBACK_ORDER.length; i++) {
    const dev = DEVICE_FALLBACK_ORDER[i]!;
    try {
      return await loadLLMWithDevice(modelId, dtype, dev);
    } catch (err) {
      const isLast = i === DEVICE_FALLBACK_ORDER.length - 1;
      if (isLast) throw err;
      logWarn(
        `[wandler] device=${dev} failed: ${err instanceof Error ? err.message : err}`,
      );
    }
  }

  // Unreachable — chain always ends with "cpu", but satisfy TS.
  throw new Error("No usable device found");
}

/**
 * Cheap probe: fetch `config.json` once and decide whether to try loading
 * this as a vision model. Returns `true` when the config carries tell-tale
 * vision fields (a nested `vision_config`, an image token id, or an
 * architecture name ending in `*ForConditionalGeneration` /
 * `*ForImageTextToText`). Any fetch failure returns `null`, which keeps
 * the downstream fallback loop intact for private / offline loads.
 */
async function isLikelyVisionModel(modelId: string): Promise<boolean | null> {
  try {
    const url = `https://huggingface.co/${modelId}/resolve/main/config.json`;
    const res = await fetch(url, {
      headers: process.env.HF_TOKEN ? { Authorization: `Bearer ${process.env.HF_TOKEN}` } : undefined,
    });
    if (!res.ok) return null;
    const config = await res.json() as {
      architectures?: string[];
      vision_config?: unknown;
      image_token_id?: unknown;
      image_token_index?: unknown;
    };
    if (config.vision_config || config.image_token_id != null || config.image_token_index != null) {
      return true;
    }
    const arch = config.architectures?.[0] ?? "";
    if (/(ForConditionalGeneration|ForImageTextToText|VLM|VisionText|Vision2Seq)$/i.test(arch)) {
      return true;
    }
    return false;
  } catch {
    return null;
  }
}

async function loadLLMWithDevice(
  modelId: string,
  dtype: string,
  device: DeviceType,
): Promise<{ model: LoadedModels["model"]; processor: unknown | null; isVision: boolean; device: DeviceType }> {
  // Probe config.json so we don't try the vision path on a 20B text-only
  // model (wasted round-trip) or skip it on a real vision model. A null
  // probe result means "unknown" and we fall through to the old try/catch
  // behaviour for safety.
  const probe = await isLikelyVisionModel(modelId);

  if (probe === false) {
    const model = await AutoModelForCausalLM.from_pretrained(modelId, {
      dtype: dtype as DtypeType,
      device,
      session_options: buildSessionOptions(device),
    }) as unknown as LoadedModels["model"];
    return { model, processor: null, isVision: false, device };
  }

  // probe === true OR probe === null (unknown): try vision first, fall back.
  try {
    const { AutoModelForImageTextToText: VisionModel } = await import("@huggingface/transformers");
    const model = await VisionModel.from_pretrained(modelId, {
      dtype: dtype as DtypeType,
      device,
      session_options: buildSessionOptions(device),
    }) as unknown as LoadedModels["model"];

    const processor = await AutoProcessor.from_pretrained(modelId);
    logInfo(`[wandler] Loaded as vision model (device=${device})`);
    return { model, processor, isVision: true, device };
  } catch {
    // Not a vision model or vision files not available — load as text-only
    const model = await AutoModelForCausalLM.from_pretrained(modelId, {
      dtype: dtype as DtypeType,
      device,
      session_options: buildSessionOptions(device),
    }) as unknown as LoadedModels["model"];
    return { model, processor: null, isVision: false, device };
  }
}

/**
 * Map a user-provided log level string ("debug" / "info" / "warning" /
 * "error" / "silent") onto the transformers.js `LogLevel` numeric enum.
 * Unknown values fall back to INFO.
 */
function toTransformersLogLevel(level: string): number {
  switch (level.toLowerCase()) {
    case "debug": return LogLevel.DEBUG;
    case "info": return LogLevel.INFO;
    case "warn":
    case "warning": return LogLevel.WARNING;
    case "error": return LogLevel.ERROR;
    case "silent":
    case "none":
    case "off": return LogLevel.NONE;
    default: return LogLevel.INFO;
  }
}

export async function loadModels(config: ServerConfig): Promise<LoadedModels> {
  // Configure transformers.js environment
  transformersEnv.cacheDir = config.cacheDir;
  if (config.hfToken) {
    process.env.HF_TOKEN = config.hfToken;
  }
  // Quiet the noisy ONNX Runtime WebGPU warnings (v4.0+). We map the user's
  // configured level instead of always silencing, so debug builds still see
  // the good stuff.
  transformersEnv.logLevel = toTransformersLogLevel(config.logLevel);
  // Cache compiled WASM kernels across runs — cuts cold start on Node.
  transformersEnv.useWasmCache = true;

  const device = config.device || "auto";

  let model: LoadedModels["model"] = null;
  let tokenizer: Tokenizer | null = null;
  let processor: unknown | null = null;
  let isVision = false;
  let loadedDevice: string | null = config.modelId ? device : null;
  let chatTemplate: string | null = null;
  let maxContextLength: number | null = null;
  let vocabSize: number | null = null;
  let attentionHeads: number | null = null;
  let generationDiagnostics: LoadedModels["generationDiagnostics"] = {
    numLogitsToKeepInput: false,
    numLogitsToKeepPatchedSessions: [],
  };

  if (config.modelId) {
    logInfo(`[wandler] Loading LLM: ${config.modelId} (${config.modelDtype}, device=${device})`);
    const t0 = Date.now();

    const resolvedDtype = await resolveDtype(config.modelId, config.modelDtype);
    warnIfLowBitMismatch(resolvedDtype, device);

    const result = await loadLLM(config.modelId, resolvedDtype, device);
    model = result.model;
    processor = result.processor;
    isVision = result.isVision;
    loadedDevice = result.device;

    tokenizer = isVision && processor
      ? (processor as { tokenizer: Tokenizer }).tokenizer ?? await AutoTokenizer.from_pretrained(config.modelId) as unknown as Tokenizer
      : await AutoTokenizer.from_pretrained(config.modelId) as unknown as Tokenizer;

    chatTemplate = await loadChatTemplate(tokenizer, config.modelId);
    maxContextLength = readMaxContextLength(model);
    vocabSize = readVocabSize(model);
    attentionHeads = readAttentionHeads(model);
    generationDiagnostics = {
      numLogitsToKeepInput: hasNumLogitsToKeepInput(model),
      numLogitsToKeepPatchedSessions: patchOnnxSessionRuns(model),
    };
    if (maxContextLength) {
      logInfo(`[wandler] Model context: ${maxContextLength} tokens`);
    }
    if (vocabSize) {
      logInfo(`[wandler] Model vocab: ${vocabSize} tokens`);
    }
    if (attentionHeads) {
      logInfo(`[wandler] Model attention heads: ${attentionHeads}`);
    }
    if (generationDiagnostics.numLogitsToKeepInput) {
      const sessions = generationDiagnostics.numLogitsToKeepPatchedSessions.join(", ") || "none";
      logInfo(`[wandler] num_logits_to_keep input detected; patched sessions: ${sessions}`);
    }
    logInfo(`[wandler] LLM ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  }

  let transcriber: LoadedModels["transcriber"] = null;
  if (config.sttModelId) {
    logInfo(`[wandler] Loading STT: ${config.sttModelId} (${config.sttDtype})`);
    const t1 = Date.now();
    const sttPipeline = await pipeline("automatic-speech-recognition", config.sttModelId, {
      dtype: config.sttDtype as DtypeType,
    });
    transcriber = (input: Float32Array) =>
      sttPipeline(input) as Promise<{ text: string }>;
    logInfo(`[wandler] STT ready in ${((Date.now() - t1) / 1000).toFixed(1)}s`);
  }

  let embedder: LoadedModels["embedder"] = null;
  if (config.embeddingModelId) {
    logInfo(`[wandler] Loading embeddings: ${config.embeddingModelId} (${config.embeddingDtype})`);
    const t2 = Date.now();
    const embPipeline = await pipeline("feature-extraction", config.embeddingModelId, {
      dtype: config.embeddingDtype as DtypeType,
    });
    embedder = (input: string, opts: Record<string, unknown>) =>
      embPipeline(input, opts) as Promise<{ data: Float32Array }>;
    logInfo(`[wandler] Embeddings ready in ${((Date.now() - t2) / 1000).toFixed(1)}s`);
  }

  return {
    device: loadedDevice,
    tokenizer,
    chatTemplate,
    processor,
    isVision,
    model,
    transcriber,
    embedder,
    maxContextLength,
    vocabSize,
    generationDiagnostics,
    attentionHeads,
  };
}
