import type http from "node:http";
import type { LoadedModels } from "../models/manager.js";
import type {
  ChatCompletionChunk,
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatMessage,
  ToolCall,
} from "../types/openai.js";
import { generate } from "../generation/generate.js";
import { buildGenOpts } from "../generation/options.js";
import { generateStream } from "../generation/stream.js";
import { parseToolCalls } from "../tools/parser.js";
import { errorJson, json, makeId, readBody } from "../utils/http.js";

/** Inject a JSON instruction if response_format is json_object */
function applyResponseFormat(
  messages: ChatMessage[],
  responseFormat?: ChatCompletionRequest["response_format"],
): ChatMessage[] {
  if (responseFormat?.type !== "json_object") return messages;

  const jsonInstruction =
    "Respond with valid JSON only. Do not include any text outside the JSON object.";

  // If there's a system message, append to it; otherwise prepend one
  const copy = messages.map((m) => ({ ...m }));
  const systemIdx = copy.findIndex((m) => m.role === "system");
  if (systemIdx >= 0) {
    copy[systemIdx]!.content = `${copy[systemIdx]!.content}\n\n${jsonInstruction}`;
  } else {
    copy.unshift({ role: "system", content: jsonInstruction });
  }
  return copy;
}

function wrapToolCallsAsStreamChunks(
  toolCalls: ToolCall[],
  id: string,
  created: number,
  modelId: string,
  promptTokens: number,
  completionTokens: number,
  includeUsage: boolean,
): ChatCompletionChunk[] {
  const chunks: ChatCompletionChunk[] = [];

  chunks.push({
    id,
    object: "chat.completion.chunk",
    created,
    model: modelId,
    choices: [{ index: 0, delta: { role: "assistant", tool_calls: toolCalls }, finish_reason: null }],
  });

  const finalChunk: ChatCompletionChunk = {
    id,
    object: "chat.completion.chunk",
    created,
    model: modelId,
    choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
  };
  if (includeUsage) {
    finalChunk.usage = {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    };
  }
  chunks.push(finalChunk);

  return chunks;
}

export async function handleChatCompletions(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  models: LoadedModels,
  modelId: string,
): Promise<void> {
  let params: ChatCompletionRequest;
  try {
    params = JSON.parse((await readBody(req)).toString()) as ChatCompletionRequest;
  } catch {
    errorJson(res, 400, "Invalid JSON");
    return;
  }

  if (!params.messages?.length) {
    errorJson(res, 400, "messages is required", "invalid_request_error", "messages");
    return;
  }

  const id = makeId();
  const created = Math.floor(Date.now() / 1000);
  const genOpts = buildGenOpts(params as ChatCompletionRequest, models.tokenizer);
  const messages = applyResponseFormat(params.messages, params.response_format);
  const includeUsage = params.stream_options?.include_usage ?? true;

  try {
    // Pure streaming (no tools) — stream tokens as they generate
    if (params.stream && !params.tools?.length) {
      await generateStream(
        res,
        models,
        modelId,
        messages,
        genOpts,
        id,
        created,
        includeUsage,
        params.tools,
      );
      return;
    }

    // Generate full text (needed for tool call parsing, or non-streaming)
    const result = await generate(models, modelId, messages, genOpts, params.tools);
    const toolCalls = params.tools?.length ? parseToolCalls(result.text) : null;

    // Streaming response wrapping pre-generated text
    if (params.stream) {
      res.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
        "Access-Control-Allow-Origin": "*",
      });
      const sse = (data: ChatCompletionChunk): void => {
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      };

      if (toolCalls) {
        for (const chunk of wrapToolCallsAsStreamChunks(
          toolCalls, id, created, modelId,
          result.promptTokens, result.completionTokens, includeUsage,
        )) {
          sse(chunk);
        }
      } else {
        sse({
          id,
          object: "chat.completion.chunk",
          created,
          model: modelId,
          choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
        });
        sse({
          id,
          object: "chat.completion.chunk",
          created,
          model: modelId,
          choices: [{ index: 0, delta: { content: result.text }, finish_reason: null }],
        });
        const finalChunk: ChatCompletionChunk = {
          id,
          object: "chat.completion.chunk",
          created,
          model: modelId,
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
        };
        if (includeUsage) {
          finalChunk.usage = {
            prompt_tokens: result.promptTokens,
            completion_tokens: result.completionTokens,
            total_tokens: result.promptTokens + result.completionTokens,
          };
        }
        sse(finalChunk);
      }
      res.write("data: [DONE]\n\n");
      res.end();
      return;
    }

    // Non-streaming response
    const message = toolCalls
      ? { role: "assistant" as const, content: null, tool_calls: toolCalls }
      : { role: "assistant" as const, content: result.text };

    const response: ChatCompletionResponse = {
      id,
      object: "chat.completion",
      created,
      model: modelId,
      choices: [{
        index: 0,
        message,
        finish_reason: toolCalls ? "tool_calls" : "stop",
      }],
      usage: {
        prompt_tokens: result.promptTokens,
        completion_tokens: result.completionTokens,
        total_tokens: result.promptTokens + result.completionTokens,
      },
    };
    json(res, 200, response);
  } catch (e) {
    console.error("[wandler] LLM error:", e);
    errorJson(res, 500, (e as Error).message, "server_error");
  }
}
