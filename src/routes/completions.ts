import type http from "node:http";
import type { LoadedModels } from "../models/manager.js";
import type {
  CompletionChunk,
  CompletionRequest,
  CompletionResponse,
} from "../types/openai.js";
import { buildGenOpts } from "../generation/options.js";
import { errorJson, json, makeId, readBody } from "../utils/http.js";

export async function handleCompletions(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  models: LoadedModels,
  modelId: string,
): Promise<void> {
  let params: CompletionRequest;
  try {
    params = JSON.parse((await readBody(req)).toString()) as CompletionRequest;
  } catch {
    errorJson(res, 400, "Invalid JSON");
    return;
  }

  if (!params.prompt) {
    errorJson(res, 400, "prompt is required", "invalid_request_error", "prompt");
    return;
  }

  // Normalize prompt to array for uniform handling
  const prompts = Array.isArray(params.prompt) ? params.prompt : [params.prompt];
  const id = makeId("cmpl");
  const created = Math.floor(Date.now() / 1000);
  const genOpts = buildGenOpts(params, models.tokenizer);

  try {
    // Handle streaming for single prompt
    if (params.stream && prompts.length === 1) {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "Access-Control-Allow-Origin": "*",
      });
      const includeUsage = params.stream_options?.include_usage ?? true;
      const sse = (data: CompletionChunk): void => {
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      };

      const prompt = prompts[0]!;
      const inputs = models.tokenizer(prompt, { return_tensors: "pt" });
      const promptTokens = inputs.input_ids.dims[1]!;

      const { TextStreamer } = await import("@huggingface/transformers");
      let completionTokens = 0;
      const streamer = new TextStreamer(
        models.tokenizer as unknown as ConstructorParameters<typeof TextStreamer>[0],
        {
          skip_prompt: true,
          callback_function: (token: string) => {
            completionTokens++;
            sse({
              id,
              object: "text_completion",
              created,
              model: modelId,
              choices: [{ index: 0, text: token, finish_reason: null }],
            });
          },
        },
      );

      await models.model.generate({ ...inputs, ...genOpts, streamer });

      const finalChunk: CompletionChunk = {
        id,
        object: "text_completion",
        created,
        model: modelId,
        choices: [{ index: 0, text: "", finish_reason: "stop" }],
      };
      if (includeUsage) {
        finalChunk.usage = {
          prompt_tokens: promptTokens,
          completion_tokens: completionTokens,
          total_tokens: promptTokens + completionTokens,
        };
      }
      sse(finalChunk);
      res.write("data: [DONE]\n\n");
      res.end();
      return;
    }

    // Non-streaming: generate for each prompt
    const choices: CompletionResponse["choices"] = [];
    let totalPromptTokens = 0;
    let totalCompletionTokens = 0;

    for (let i = 0; i < prompts.length; i++) {
      const prompt = prompts[i]!;
      const inputs = models.tokenizer(prompt, { return_tensors: "pt" });
      const outputIds = await models.model.generate({ ...inputs, ...genOpts });

      const promptTokens = inputs.input_ids.dims[1]!;
      const completionTokens = outputIds.dims[1]! - promptTokens;
      const newIds = outputIds.slice(null, [promptTokens, null]);
      let text = models.tokenizer.batch_decode(newIds, { skip_special_tokens: true })[0]!;

      // Echo: prepend the original prompt
      if (params.echo) {
        text = prompt + text;
      }

      // Suffix: append after the completion
      if (params.suffix) {
        text = text + params.suffix;
      }

      totalPromptTokens += promptTokens;
      totalCompletionTokens += completionTokens;

      choices.push({
        index: i,
        text,
        finish_reason: "stop",
      });
    }

    const response: CompletionResponse = {
      id,
      object: "text_completion",
      created,
      model: modelId,
      choices,
      usage: {
        prompt_tokens: totalPromptTokens,
        completion_tokens: totalCompletionTokens,
        total_tokens: totalPromptTokens + totalCompletionTokens,
      },
    };
    json(res, 200, response);
  } catch (e) {
    console.error("[wandler] Completion error:", e);
    errorJson(res, 500, (e as Error).message, "server_error");
  }
}
