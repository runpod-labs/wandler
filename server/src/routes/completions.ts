import type { Context } from "hono";
import { streamSSE } from "hono/streaming";
import type { AppEnv } from "../server.js";
import type {
  CompletionChunk,
  CompletionRequest,
  CompletionResponse,
  GenerationProfile,
} from "../types/openai.js";
import { buildGenOpts } from "../generation/options.js";
import { getGenerationProfile, getGenerationStatusCode } from "../generation/profile.js";
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
    generationProfile?: GenerationProfile;
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
      generationProfile: overrides.generationProfile,
    });
  };

  const config = c.get("config");
  const models = c.get("models");
  const backend = c.get("backend");
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
    const genOpts = buildGenOpts(params, models.tokenizer!, config.maxTokens, models.maxContextLength, config.prefillChunkSize);

    // Streaming for single prompt
    if (params.stream && prompts.length === 1) {
      const includeUsage = params.stream_options?.include_usage ?? true;
      const prompt = prompts[0]!;

      return streamSSE(c, async (stream) => {
        try {
          const result = await backend.streamCompletion(prompt, genOpts, async (token) => {
            await stream.writeSSE({ data: JSON.stringify({
              id, object: "text_completion", created, model: modelId,
              choices: [{ index: 0, text: token, finish_reason: null }],
            } satisfies CompletionChunk) });
          });

          const finalChunk: CompletionChunk = {
            id, object: "text_completion", created, model: modelId,
            choices: [{ index: 0, text: "", finish_reason: "stop" }],
          };
          if (includeUsage) {
            finalChunk.usage = {
              prompt_tokens: result.promptTokens,
              completion_tokens: result.completionTokens,
              total_tokens: result.promptTokens + result.completionTokens,
            };
          }
          await stream.writeSSE({ data: JSON.stringify(finalChunk) });
          await stream.writeSSE({ data: "[DONE]" });
          finalize({
            promptTokens: result.promptTokens,
            completionTokens: result.completionTokens,
            stream: true,
            generationProfile: result.profile,
          });
        } catch (error) {
          const profile = getGenerationProfile(error);
          finalize({
            statusCode: getGenerationStatusCode(error) ?? 500,
            stream: true,
            promptTokens: profile?.promptTokens,
            completionTokens: profile?.completionTokens,
            generationProfile: profile,
          });
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
      const result = await backend.generateCompletion(prompt, genOpts);
      let text = result.text;

      if (params.echo) text = prompt + text;
      if (params.suffix) text = text + params.suffix;

      totalPromptTokens += result.promptTokens;
      totalCompletionTokens += result.completionTokens;
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
