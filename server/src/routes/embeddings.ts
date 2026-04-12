import type { Context } from "hono";
import type { AppEnv } from "../server.js";
import type { EmbeddingRequest, EmbeddingResponse } from "../types/openai.js";

function float32ToBase64(arr: number[]): string {
  const float32 = new Float32Array(arr);
  const bytes = new Uint8Array(float32.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return btoa(binary);
}

export async function embeddings(c: Context<AppEnv>) {
  const config = c.get("config");
  const models = c.get("models");

  if (!models.embedder) {
    return c.json(
      { error: { message: "Embedding model not loaded. Set EMBEDDING_MODEL_ID to enable.", type: "server_error", param: null, code: null } },
      503,
    );
  }

  const params = await c.req.json<EmbeddingRequest>();

  if (!params.input) {
    return c.json(
      { error: { message: "input is required", type: "invalid_request_error", param: "input", code: null } },
      400,
    );
  }

  const inputs = Array.isArray(params.input) ? params.input : [params.input];
  const useBase64 = params.encoding_format === "base64";

  let totalTokens = 0;
  const data: EmbeddingResponse["data"] = [];

  for (let i = 0; i < inputs.length; i++) {
    const text = inputs[i]!;
    const tokens = models.tokenizer(text, { return_tensors: "pt" });
    totalTokens += tokens.input_ids.dims[1]!;

    const result = await models.embedder(text, { pooling: "mean", normalize: true });
    const embeddingArray = Array.from(result.data as Float32Array);

    data.push({
      object: "embedding",
      embedding: useBase64 ? float32ToBase64(embeddingArray) : embeddingArray,
      index: i,
    });
  }

  return c.json({
    object: "list",
    data,
    model: config.embeddingModelId || config.modelId,
    usage: { prompt_tokens: totalTokens, total_tokens: totalTokens },
  });
}
