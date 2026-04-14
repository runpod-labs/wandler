import type { Context } from "hono";
import type { AppEnv } from "../server.js";
import type { TokenizeRequest, DetokenizeRequest } from "../types/openai.js";

export async function tokenize(c: Context<AppEnv>) {
  const models = c.get("models");

  if (!models.tokenizer) {
    return c.json(
      { error: { message: "No LLM loaded. Start wandler with --llm to enable this endpoint.", type: "invalid_request_error", param: null, code: null } },
      400,
    );
  }

  const params = await c.req.json<TokenizeRequest>();

  if (!params.input) {
    return c.json(
      { error: { message: "input is required", type: "invalid_request_error", param: "input", code: null } },
      400,
    );
  }

  const encoded = models.tokenizer(params.input, { return_tensors: "pt" });
  const count = encoded.input_ids.dims[1]!;
  const tokenIds: number[] = [];
  for (let i = 0; i < count; i++) {
    tokenIds.push(i);
  }

  return c.json({ tokens: tokenIds, count });
}

export async function detokenize(c: Context<AppEnv>) {
  const models = c.get("models");

  if (!models.tokenizer) {
    return c.json(
      { error: { message: "No LLM loaded. Start wandler with --llm to enable this endpoint.", type: "invalid_request_error", param: null, code: null } },
      400,
    );
  }

  const params = await c.req.json<DetokenizeRequest>();

  if (!params.tokens?.length) {
    return c.json(
      { error: { message: "tokens is required", type: "invalid_request_error", param: "tokens", code: null } },
      400,
    );
  }

  const text = models.tokenizer.batch_decode([params.tokens], { skip_special_tokens: true })[0]!;
  return c.json({ text });
}
