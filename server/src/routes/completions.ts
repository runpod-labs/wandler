import type { Context } from "hono";
import { streamSSE } from "hono/streaming";
import type { AppEnv } from "../server.js";
import type {
  CompletionChunk,
  CompletionRequest,
  CompletionResponse,
} from "../types/openai.js";
import { buildGenOpts, stripInternalGenOpts } from "../generation/options.js";
import { makeId } from "../utils/http.js";
import { requestStarted, trackFailedRequest, trackRequest } from "./admin.js";

export async function completions(c: Context<AppEnv>) {
  const route = "/v1/completions";
  const startedAt = Date.now();
  requestStarted();

  let tracked = false;
  const finalize = (overrides: {
    promptTokens?: number;
    completionTokens?: number;
    statusCode?: number;
    stream?: boolean;
  }) => {
    if (tracked) return;
    tracked = true;
    trackRequest({
      route,
      promptTokens: overrides.promptTokens,
      completionTokens: overrides.completionTokens,
      totalMs: Date.now() - startedAt,
      stream: overrides.stream,
      statusCode: overrides.statusCode ?? 200,
    });
  };

  const config = c.get("config");
  const models = c.get("models");
  const modelId = config.modelId;

  if (!models.model || !models.tokenizer) {
    finalize({ statusCode: 400 });
    return c.json(
      { error: { message: "No LLM loaded. Start wandler with --llm to enable this endpoint.", type: "invalid_request_error", param: null, code: null } },
      400,
    );
  }

  try {
    const params = await c.req.json<CompletionRequest>();

    if (!params.prompt) {
      finalize({ statusCode: 400 });
      return c.json(
        { error: { message: "prompt is required", type: "invalid_request_error", param: "prompt", code: null } },
        400,
      );
    }

    const prompts = Array.isArray(params.prompt) ? params.prompt : [params.prompt];
    const id = makeId("cmpl");
    const created = Math.floor(Date.now() / 1000);
    const genOpts = buildGenOpts(params, models.tokenizer!, config.maxTokens, models.maxContextLength, config.prefillChunkSize, models.device);
    const transformersGenOpts = stripInternalGenOpts(genOpts);

    // Streaming for single prompt
    if (params.stream && prompts.length === 1) {
      const includeUsage = params.stream_options?.include_usage ?? true;
      const prompt = prompts[0]!;
      const inputs = models.tokenizer!(prompt, { return_tensors: "pt" });
      const promptTokens = inputs.input_ids.dims[1]!;

      return streamSSE(c, async (stream) => {
        try {
          const { TextStreamer } = await import("@huggingface/transformers");
          let completionTokens = 0;

          const streamer = new TextStreamer(
            models.tokenizer! as unknown as ConstructorParameters<typeof TextStreamer>[0],
            {
              skip_prompt: true,
              callback_function: (token: string) => {
                completionTokens++;
                stream.writeSSE({ data: JSON.stringify({
                  id, object: "text_completion", created, model: modelId,
                  choices: [{ index: 0, text: token, finish_reason: null }],
                } satisfies CompletionChunk) });
              },
            },
          );

          await models.model!.generate({ ...inputs, ...transformersGenOpts, streamer });

          const finalChunk: CompletionChunk = {
            id, object: "text_completion", created, model: modelId,
            choices: [{ index: 0, text: "", finish_reason: "stop" }],
          };
          if (includeUsage) {
            finalChunk.usage = { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
          }
          await stream.writeSSE({ data: JSON.stringify(finalChunk) });
          await stream.writeSSE({ data: "[DONE]" });
          finalize({ promptTokens, completionTokens, stream: true });
        } catch (error) {
          finalize({ statusCode: 500, stream: true });
          throw error;
        }
      });
    }

    // Non-streaming
    const choices: CompletionResponse["choices"] = [];
    let totalPromptTokens = 0;
    let totalCompletionTokens = 0;

    for (let i = 0; i < prompts.length; i++) {
      const prompt = prompts[i]!;
      const inputs = models.tokenizer!(prompt, { return_tensors: "pt" });
      const outputIds = await models.model!.generate({ ...inputs, ...transformersGenOpts });

      const promptTokens = inputs.input_ids.dims[1]!;
      const completionTokens = outputIds.dims[1]! - promptTokens;
      const newIds = outputIds.slice(null, [promptTokens, null]);
      let text = models.tokenizer!.batch_decode(newIds, { skip_special_tokens: true })[0]!;

      if (params.echo) text = prompt + text;
      if (params.suffix) text = text + params.suffix;

      totalPromptTokens += promptTokens;
      totalCompletionTokens += completionTokens;
      choices.push({ index: i, text, finish_reason: "stop" });
    }

    finalize({
      promptTokens: totalPromptTokens,
      completionTokens: totalCompletionTokens,
      stream: false,
    });
    return c.json({
      id, object: "text_completion", created, model: modelId, choices,
      usage: { prompt_tokens: totalPromptTokens, completion_tokens: totalCompletionTokens, total_tokens: totalPromptTokens + totalCompletionTokens },
    } satisfies CompletionResponse);
  } catch (error) {
    if (!tracked) {
      trackFailedRequest(route);
      tracked = true;
    }
    throw error;
  }
}
