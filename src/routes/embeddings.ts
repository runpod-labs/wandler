import type http from "node:http";
import type { LoadedModels } from "../models/manager.js";
import type { EmbeddingRequest, EmbeddingResponse } from "../types/openai.js";
import { errorJson, json, readBody } from "../utils/http.js";

function float32ToBase64(arr: number[]): string {
  const float32 = new Float32Array(arr);
  const bytes = new Uint8Array(float32.buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]!);
  }
  return btoa(binary);
}

export async function handleEmbeddings(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  models: LoadedModels,
  modelId: string,
): Promise<void> {
  if (!models.embedder) {
    errorJson(res, 503, "Embedding model not loaded. Set EMBEDDING_MODEL_ID to enable.", "server_error");
    return;
  }

  let params: EmbeddingRequest;
  try {
    params = JSON.parse((await readBody(req)).toString()) as EmbeddingRequest;
  } catch {
    errorJson(res, 400, "Invalid JSON");
    return;
  }

  if (!params.input) {
    errorJson(res, 400, "input is required", "invalid_request_error", "input");
    return;
  }

  const inputs = Array.isArray(params.input) ? params.input : [params.input];
  const useBase64 = params.encoding_format === "base64";

  try {
    let totalTokens = 0;
    const data: EmbeddingResponse["data"] = [];

    for (let i = 0; i < inputs.length; i++) {
      const text = inputs[i]!;
      // Approximate token count from tokenizer
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

    const response: EmbeddingResponse = {
      object: "list",
      data,
      model: modelId,
      usage: {
        prompt_tokens: totalTokens,
        total_tokens: totalTokens,
      },
    };
    json(res, 200, response);
  } catch (e) {
    console.error("[wandler] Embedding error:", e);
    errorJson(res, 500, (e as Error).message, "server_error");
  }
}
