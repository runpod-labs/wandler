import type { Context } from "hono";
import { streamSSE } from "hono/streaming";
import type { AppEnv } from "../server.js";
import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
  GenerationProfile,
  ChatMessage,
} from "../types/openai.js";
import { generate } from "../generation/generate.js";
import { buildGenOpts } from "../generation/options.js";
import { generateStreamTokens } from "../generation/stream.js";
import { generateStreamWithTools } from "../generation/stream-tools.js";
import { getGenerationProfile, getGenerationStatusCode } from "../generation/profile.js";
import { parseToolCalls } from "../tools/parser.js";
import { makeId } from "../utils/http.js";
import { getTextContent } from "../utils/content.js";
import { requestStarted, trackRequest } from "./admin.js";

function applyResponseFormat(
  messages: ChatMessage[],
  responseFormat?: ChatCompletionRequest["response_format"],
): ChatMessage[] {
  if (!responseFormat || responseFormat.type === "text") return messages;

  let jsonInstruction: string;
  if (responseFormat.type === "json_schema" && responseFormat.json_schema) {
    const schemaStr = JSON.stringify(responseFormat.json_schema.schema);
    jsonInstruction = `Respond with valid JSON only that conforms to this JSON schema:\n${schemaStr}\nDo not include any text outside the JSON object.`;
  } else {
    jsonInstruction = "Respond with valid JSON only. Do not include any text outside the JSON object.";
  }

  const copy = messages.map((m) => ({ ...m }));
  const systemIdx = copy.findIndex((m) => m.role === "system");
  if (systemIdx >= 0) {
    const existing = getTextContent(copy[systemIdx]!.content);
    copy[systemIdx]!.content = `${existing}\n\n${jsonInstruction}`;
  } else {
    copy.unshift({ role: "system", content: jsonInstruction });
  }
  return copy;
}

export async function chatCompletions(c: Context<AppEnv>) {
  const route = "/v1/chat/completions";
  const startedAt = Date.now();
  requestStarted();

  let tracked = false;
  const finalize = (overrides: {
    promptTokens?: number;
    completionTokens?: number;
    statusCode?: number;
    stream?: boolean;
    toolsCount?: number;
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
      toolsCount: overrides.toolsCount,
      statusCode: overrides.statusCode ?? 200,
      generationProfile: overrides.generationProfile,
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
    const params = await c.req.json<ChatCompletionRequest>();

    if (!params.messages?.length) {
      finalize({ statusCode: 400 });
      return c.json(
        { error: { message: "messages is required", type: "invalid_request_error", param: "messages", code: null } },
        400,
      );
    }

    const id = makeId();
    const created = Math.floor(Date.now() / 1000);
    const genOpts = buildGenOpts(params, models.tokenizer, config.maxTokens, models.maxContextLength);
    const messages = applyResponseFormat(params.messages, params.response_format);
    const includeUsage = params.stream_options?.include_usage ?? true;
    const toolsCount = params.tools?.length ?? 0;

    // Streaming path: content tokens and tool calls both arrive incrementally.
    if (params.stream) {
      return streamSSE(c, async (stream) => {
        let promptTokens = 0;
        let completionTokens = 0;
        let profile: GenerationProfile | undefined;
        let finishReason: "stop" | "tool_calls" = "stop";

        try {
          await stream.writeSSE({ data: JSON.stringify({
            id, object: "chat.completion.chunk", created, model: modelId,
            choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
          } satisfies ChatCompletionChunk) });

          if (params.tools?.length) {
            const result = await generateStreamWithTools(
              models, modelId, messages, genOpts, params.tools,
              {
                onContent: async (delta) => {
                  if (!delta) return;
                  await stream.writeSSE({ data: JSON.stringify({
                    id, object: "chat.completion.chunk", created, model: modelId,
                    choices: [{ index: 0, delta: { content: delta }, finish_reason: null }],
                  } satisfies ChatCompletionChunk) });
                },
                onToolCalls: async (calls) => {
                  finishReason = "tool_calls";
                  await stream.writeSSE({ data: JSON.stringify({
                    id, object: "chat.completion.chunk", created, model: modelId,
                    choices: [{ index: 0, delta: { tool_calls: calls }, finish_reason: null }],
                  } satisfies ChatCompletionChunk) });
                },
              },
            );
            promptTokens = result.promptTokens;
            completionTokens = result.completionTokens;
          } else {
            const result = await generateStreamTokens(
              models, modelId, messages, genOpts,
              async (token) => {
                await stream.writeSSE({ data: JSON.stringify({
                  id, object: "chat.completion.chunk", created, model: modelId,
                  choices: [{ index: 0, delta: { content: token }, finish_reason: null }],
                } satisfies ChatCompletionChunk) });
              },
            );
            promptTokens = result.promptTokens;
            completionTokens = result.completionTokens;
            profile = result.profile;
          }

          const finalChunk: ChatCompletionChunk = {
            id, object: "chat.completion.chunk", created, model: modelId,
            choices: [{ index: 0, delta: {}, finish_reason: finishReason }],
          };
          if (includeUsage) {
            finalChunk.usage = { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
          }
          await stream.writeSSE({ data: JSON.stringify(finalChunk) });
          await stream.writeSSE({ data: "[DONE]" });
          finalize({ promptTokens, completionTokens, stream: true, toolsCount, generationProfile: profile });
        } catch (error) {
          const errorProfile = getGenerationProfile(error);
          const statusCode = getGenerationStatusCode(error) ?? 500;
          finalize({
            statusCode,
            stream: true,
            toolsCount,
            promptTokens: errorProfile?.promptTokens ?? promptTokens,
            completionTokens: errorProfile?.completionTokens ?? completionTokens,
            generationProfile: errorProfile ?? profile,
          });
          throw error;
        }
      });
    }

    // Non-streaming: generate the full response, then parse for tool calls.
    const result = await generate(models, modelId, messages, genOpts, params.tools);
    const toolCalls = params.tools?.length ? parseToolCalls(result.text) : null;

    const message = toolCalls
      ? { role: "assistant" as const, content: null, tool_calls: toolCalls }
      : { role: "assistant" as const, content: result.text };

    finalize({
      promptTokens: result.promptTokens,
      completionTokens: result.completionTokens,
      stream: false,
      toolsCount,
      generationProfile: result.profile,
    });
    return c.json({
      id, object: "chat.completion", created, model: modelId,
      choices: [{ index: 0, message, finish_reason: toolCalls ? "tool_calls" : "stop" }],
      usage: { prompt_tokens: result.promptTokens, completion_tokens: result.completionTokens, total_tokens: result.promptTokens + result.completionTokens },
    } satisfies ChatCompletionResponse);
  } catch (error) {
    const statusCode = getGenerationStatusCode(error) ?? 500;
    if (!tracked) {
      const profile = getGenerationProfile(error);
      finalize({
        statusCode,
        promptTokens: profile?.promptTokens,
        completionTokens: profile?.completionTokens,
        generationProfile: profile,
      });
    }
    if (statusCode < 500) {
      return c.json(
        { error: { message: error instanceof Error ? error.message : String(error), type: "invalid_request_error", param: null, code: "context_length_exceeded" } },
        statusCode as 400,
      );
    }
    throw error;
  }
}
