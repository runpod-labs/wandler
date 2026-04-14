import type { Context } from "hono";
import { streamSSE } from "hono/streaming";
import type { AppEnv } from "../server.js";
import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatMessage,
} from "../types/openai.js";
import { generate } from "../generation/generate.js";
import { buildGenOpts } from "../generation/options.js";
import { generateStreamTokens } from "../generation/stream.js";
import { parseToolCalls } from "../tools/parser.js";
import { makeId } from "../utils/http.js";
import { getTextContent } from "../utils/content.js";

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
  const config = c.get("config");
  const models = c.get("models");
  const modelId = config.modelId;

  if (!models.model || !models.tokenizer) {
    return c.json(
      { error: { message: "No LLM loaded. Start wandler with --llm to enable this endpoint.", type: "invalid_request_error", param: null, code: null } },
      400,
    );
  }

  const params = await c.req.json<ChatCompletionRequest>();

  if (!params.messages?.length) {
    return c.json(
      { error: { message: "messages is required", type: "invalid_request_error", param: "messages", code: null } },
      400,
    );
  }

  const id = makeId();
  const created = Math.floor(Date.now() / 1000);
  const genOpts = buildGenOpts(params, models.tokenizer, config.maxTokens);
  const messages = applyResponseFormat(params.messages, params.response_format);
  const includeUsage = params.stream_options?.include_usage ?? true;

  // Pure streaming (no tools)
  if (params.stream && !params.tools?.length) {
    return streamSSE(c, async (stream) => {
      await stream.writeSSE({ data: JSON.stringify({
        id, object: "chat.completion.chunk", created, model: modelId,
        choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
      } satisfies ChatCompletionChunk) });

      const { promptTokens, completionTokens } = await generateStreamTokens(
        models, modelId, messages, genOpts,
        async (token) => {
          await stream.writeSSE({ data: JSON.stringify({
            id, object: "chat.completion.chunk", created, model: modelId,
            choices: [{ index: 0, delta: { content: token }, finish_reason: null }],
          } satisfies ChatCompletionChunk) });
        },
        params.tools,
      );

      const finalChunk: ChatCompletionChunk = {
        id, object: "chat.completion.chunk", created, model: modelId,
        choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
      };
      if (includeUsage) {
        finalChunk.usage = { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
      }
      await stream.writeSSE({ data: JSON.stringify(finalChunk) });
      await stream.writeSSE({ data: "[DONE]" });
    });
  }

  // Generate full text (needed for tool call parsing, or non-streaming)
  const result = await generate(models, modelId, messages, genOpts, params.tools);
  const toolCalls = params.tools?.length ? parseToolCalls(result.text) : null;

  // Streaming response wrapping pre-generated text
  if (params.stream) {
    return streamSSE(c, async (stream) => {
      if (toolCalls) {
        await stream.writeSSE({ data: JSON.stringify({
          id, object: "chat.completion.chunk", created, model: modelId,
          choices: [{ index: 0, delta: { role: "assistant", tool_calls: toolCalls }, finish_reason: null }],
        } satisfies ChatCompletionChunk) });
        const finalChunk: ChatCompletionChunk = {
          id, object: "chat.completion.chunk", created, model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
        };
        if (includeUsage) {
          finalChunk.usage = { prompt_tokens: result.promptTokens, completion_tokens: result.completionTokens, total_tokens: result.promptTokens + result.completionTokens };
        }
        await stream.writeSSE({ data: JSON.stringify(finalChunk) });
      } else {
        await stream.writeSSE({ data: JSON.stringify({
          id, object: "chat.completion.chunk", created, model: modelId,
          choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
        } satisfies ChatCompletionChunk) });
        await stream.writeSSE({ data: JSON.stringify({
          id, object: "chat.completion.chunk", created, model: modelId,
          choices: [{ index: 0, delta: { content: result.text }, finish_reason: null }],
        } satisfies ChatCompletionChunk) });
        const finalChunk: ChatCompletionChunk = {
          id, object: "chat.completion.chunk", created, model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        };
        if (includeUsage) {
          finalChunk.usage = { prompt_tokens: result.promptTokens, completion_tokens: result.completionTokens, total_tokens: result.promptTokens + result.completionTokens };
        }
        await stream.writeSSE({ data: JSON.stringify(finalChunk) });
      }
      await stream.writeSSE({ data: "[DONE]" });
    });
  }

  // Non-streaming
  const message = toolCalls
    ? { role: "assistant" as const, content: null, tool_calls: toolCalls }
    : { role: "assistant" as const, content: result.text };

  return c.json({
    id, object: "chat.completion", created, model: modelId,
    choices: [{ index: 0, message, finish_reason: toolCalls ? "tool_calls" : "stop" }],
    usage: { prompt_tokens: result.promptTokens, completion_tokens: result.completionTokens, total_tokens: result.promptTokens + result.completionTokens },
  } satisfies ChatCompletionResponse);
}
