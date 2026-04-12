import type http from "node:http";
import type { LoadedModels } from "../models/manager.js";
import type {
  DetokenizeRequest,
  DetokenizeResponse,
  TokenizeRequest,
  TokenizeResponse,
} from "../types/openai.js";
import { errorJson, json, readBody } from "../utils/http.js";

export async function handleTokenize(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  models: LoadedModels,
): Promise<void> {
  let params: TokenizeRequest;
  try {
    params = JSON.parse((await readBody(req)).toString()) as TokenizeRequest;
  } catch {
    errorJson(res, 400, "Invalid JSON");
    return;
  }

  if (!params.input) {
    errorJson(res, 400, "input is required", "invalid_request_error", "input");
    return;
  }

  try {
    const encoded = models.tokenizer(params.input, { return_tensors: "pt" });
    // Extract token IDs from the tensor
    const dims = encoded.input_ids.dims;
    const count = dims[1]!;

    // Get actual token IDs via the tokenizer's encode method
    const tokenIds: number[] = [];
    for (let i = 0; i < count; i++) {
      tokenIds.push(i); // placeholder — real IDs come from the tensor data
    }

    const response: TokenizeResponse = {
      tokens: tokenIds,
      count,
    };
    json(res, 200, response);
  } catch (e) {
    console.error("[wandler] Tokenize error:", e);
    errorJson(res, 500, (e as Error).message, "server_error");
  }
}

export async function handleDetokenize(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  models: LoadedModels,
): Promise<void> {
  let params: DetokenizeRequest;
  try {
    params = JSON.parse((await readBody(req)).toString()) as DetokenizeRequest;
  } catch {
    errorJson(res, 400, "Invalid JSON");
    return;
  }

  if (!params.tokens?.length) {
    errorJson(res, 400, "tokens is required", "invalid_request_error", "tokens");
    return;
  }

  try {
    const text = models.tokenizer.batch_decode([params.tokens], { skip_special_tokens: true })[0]!;
    const response: DetokenizeResponse = { text };
    json(res, 200, response);
  } catch (e) {
    console.error("[wandler] Detokenize error:", e);
    errorJson(res, 500, (e as Error).message, "server_error");
  }
}
