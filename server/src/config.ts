import { homedir } from "node:os";
import { join } from "node:path";

// ── Server configuration from environment variables ─────────────────────────

export interface ServerConfig {
  port: number;
  host: string;
  modelId: string;
  modelDtype: string;
  device: string;
  sttModelId: string;
  sttDtype: string;
  embeddingModelId: string;
  embeddingDtype: string;
  apiKey: string;
  corsOrigin: string;
  /**
   * Optional hard cap on `max_new_tokens` per request.
   * When `null` (the default), the effective cap is derived from the loaded
   * model's `max_position_embeddings`. Set `--max-tokens <n>` on the CLI /
   * `WANDLER_MAX_TOKENS=<n>` in the env to opt into an explicit cap — useful
   * for shared deployments where you want to protect the host.
   */
  maxTokens: number | null;
  maxConcurrent: number;
  timeout: number;
  logLevel: string;
  hfToken: string;
  cacheDir: string;
  prefillChunkSize: string;
  warmupTokens: number;
  warmupMaxNewTokens: number;
}

/**
 * Parse a model reference like "org/repo:q4" into { id, dtype }.
 * If no :suffix is present, falls back to the provided default dtype.
 *
 * Supported dtype suffixes (transformers.js v4.2+):
 *   - Full precision: fp32, fp16
 *   - 8-bit:          q8, int8, uint8
 *   - 4-bit:          q4, bnb4, q4f16
 *   - 2-bit:          q2, q2f16       (v4.1+, CPU/WebGPU only)
 *   - 1-bit (BitNet): q1, q1f16       (v4.1+, CPU/WebGPU only)
 *   - Auto:           auto            (defers to the model's config dtype)
 *
 * Example: "onnx-community/Qwen3-0.6B-ONNX:q4" or "some-org/bitnet:q1".
 */
export function parseModelRef(ref: string, defaultDtype: string): { id: string; dtype: string } {
  const lastColon = ref.lastIndexOf(":");
  const slashIdx = ref.indexOf("/");
  // Only split on colon if there's a slash (org/repo format)
  // and the colon comes after the slash
  if (lastColon > 0 && slashIdx >= 0 && slashIdx < lastColon) {
    return { id: ref.slice(0, lastColon), dtype: ref.slice(lastColon + 1) };
  }
  return { id: ref, dtype: defaultDtype };
}

/**
 * Resolve the default model cache directory following the standard HuggingFace
 * cache hierarchy so downloaded models are shared with vLLM, the Python
 * `huggingface_hub` package, and other HF-ecosystem tools:
 *
 *   WANDLER_CACHE_DIR  >  HF_HOME  >  XDG_CACHE_HOME/huggingface  >  ~/.cache/huggingface
 *
 * This avoids the transformers.js default of caching inside `node_modules/`,
 * which gets wiped on every `npm install` or when running via `npx`.
 */
function defaultHfCacheDir(env: Record<string, string | undefined>): string {
  if (env.HF_HOME) return env.HF_HOME;
  if (env.XDG_CACHE_HOME) return join(env.XDG_CACHE_HOME, "huggingface");
  return join(homedir(), ".cache", "huggingface");
}

export function loadConfig(env: Record<string, string | undefined> = process.env): ServerConfig {
  const DEFAULT_DTYPE = "q4";

  const llmRaw = env.WANDLER_LLM ?? env.MODEL_ID ?? "";
  const sttRaw = env.WANDLER_STT ?? env.STT_MODEL_ID ?? "";
  const embRaw = env.WANDLER_EMBEDDING ?? env.EMBEDDING_MODEL_ID ?? "";

  const llm = parseModelRef(llmRaw, env.DTYPE || DEFAULT_DTYPE);
  const stt = sttRaw ? parseModelRef(sttRaw, env.STT_DTYPE || DEFAULT_DTYPE) : { id: "", dtype: DEFAULT_DTYPE };
  const emb = embRaw ? parseModelRef(embRaw, env.EMBEDDING_DTYPE || "q8") : { id: "", dtype: "q8" };

  return {
    port: parseInt(env.WANDLER_PORT || env.PORT || "8000", 10),
    host: env.WANDLER_HOST || "127.0.0.1",
    modelId: llm.id,
    modelDtype: llm.dtype,
    device: env.WANDLER_DEVICE || env.DEVICE || "auto",
    sttModelId: stt.id,
    sttDtype: stt.dtype,
    embeddingModelId: emb.id,
    embeddingDtype: emb.dtype,
    apiKey: env.WANDLER_API_KEY || "",
    corsOrigin: env.WANDLER_CORS_ORIGIN || "*",
    maxTokens: env.WANDLER_MAX_TOKENS ? parseInt(env.WANDLER_MAX_TOKENS, 10) : null,
    maxConcurrent: parseInt(env.WANDLER_MAX_CONCURRENT || "1", 10),
    timeout: parseInt(env.WANDLER_TIMEOUT || "120000", 10),
    logLevel: env.WANDLER_LOG_LEVEL || "info",
    hfToken: env.HF_TOKEN || env.WANDLER_HF_TOKEN || "",
    cacheDir: env.WANDLER_CACHE_DIR || defaultHfCacheDir(env),
    prefillChunkSize: env.WANDLER_PREFILL_CHUNK_SIZE || "2048",
    warmupTokens: parseNonNegativeInt(env.WANDLER_WARMUP_TOKENS, 0),
    warmupMaxNewTokens: parsePositiveInt(env.WANDLER_WARMUP_MAX_NEW_TOKENS, 8),
  };
}

function parseNonNegativeInt(raw: string | undefined, fallback: number): number {
  if (raw == null || raw === "") return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function parsePositiveInt(raw: string | undefined, fallback: number): number {
  if (raw == null || raw === "") return fallback;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}
