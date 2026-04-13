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
  maxTokens: number;
  maxConcurrent: number;
  timeout: number;
  logLevel: string;
  hfToken: string;
  cacheDir: string;
}

/**
 * Parse a model reference like "org/repo:q4" into { id, dtype }.
 * If no :suffix is present, falls back to the provided default dtype.
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

export function loadConfig(env: Record<string, string | undefined> = process.env): ServerConfig {
  const DEFAULT_DTYPE = "q4";

  const llmRaw = env.WANDLER_LLM ?? env.MODEL_ID ?? "onnx-community/gemma-4-E4B-it-ONNX:q4";
  const sttRaw = env.WANDLER_STT ?? env.STT_MODEL_ID ?? "onnx-community/whisper-tiny:q4";
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
    maxTokens: parseInt(env.WANDLER_MAX_TOKENS || "2048", 10),
    maxConcurrent: parseInt(env.WANDLER_MAX_CONCURRENT || "1", 10),
    timeout: parseInt(env.WANDLER_TIMEOUT || "120000", 10),
    logLevel: env.WANDLER_LOG_LEVEL || "info",
    hfToken: env.HF_TOKEN || env.WANDLER_HF_TOKEN || "",
    cacheDir: env.WANDLER_CACHE_DIR || "",
  };
}
