// ── Server configuration from environment variables ─────────────────────────

export interface ServerConfig {
  port: number;
  modelId: string;
  modelDtype: string;
  device: string;
  sttModelId: string;
  sttDtype: string;
  embeddingModelId: string;
  embeddingDtype: string;
  apiKey: string;
}

export function loadConfig(env: Record<string, string | undefined> = process.env): ServerConfig {
  return {
    port: parseInt(env.PORT || "8000", 10),
    modelId: env.MODEL_ID || "onnx-community/gemma-4-E4B-it-ONNX",
    modelDtype: env.DTYPE || "q4",
    device: env.DEVICE || "webgpu",
    sttModelId: env.STT_MODEL_ID || "onnx-community/whisper-tiny",
    sttDtype: env.STT_DTYPE || "q4",
    embeddingModelId: env.EMBEDDING_MODEL_ID || "",
    embeddingDtype: env.EMBEDDING_DTYPE || "q8",
    apiKey: env.WANDLER_API_KEY || "",
  };
}
