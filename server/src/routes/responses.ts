import type { Context } from "hono";
import { streamSSE } from "hono/streaming";
import type { AppEnv } from "../server.js";
import type {
  ChatMessage,
  ResponsesRequest,
  ResponsesInputItem,
  ResponsesMessageItem,
  ResponsesFunctionCallItem,
  ResponsesFunctionCallOutputItem,
  ResponsesResponse,
  ResponsesOutputItem,
  ResponsesOutputMessage,
  ResponsesOutputFunctionCall,
  ResponsesTool,
  Tool,
  ToolCall,
} from "../types/openai.js";
import { generate } from "../generation/generate.js";
import { buildGenOpts } from "../generation/options.js";
import { generateStreamTokens } from "../generation/stream.js";
import { generateStreamWithTools } from "../generation/stream-tools.js";
import { parseToolCalls } from "../tools/parser.js";
import { makeId } from "../utils/http.js";
import { getTextContent } from "../utils/content.js";

// ── Format conversion helpers ─────────────────────────────────────────────

/**
 * Convert Responses API flat tool format to Chat Completions nested format.
 * Responses: { type: "function", name, description, parameters }
 * Chat:      { type: "function", function: { name, description, parameters } }
 */
function responsesToolToChat(tool: ResponsesTool): Tool {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    },
  };
}

/**
 * Convert Chat Completions ToolCall to Responses API function_call output item.
 */
function toolCallToFunctionCall(tc: ToolCall): ResponsesOutputFunctionCall {
  return {
    type: "function_call",
    id: makeId("fc"),
    call_id: tc.id,
    name: tc.function.name,
    arguments: tc.function.arguments,
    status: "completed",
  };
}

/**
 * Convert Responses API `input` + `instructions` into ChatMessage[].
 */
function inputToChatMessages(
  input: string | ResponsesInputItem[],
  instructions?: string | null,
): ChatMessage[] {
  const messages: ChatMessage[] = [];

  // Prepend instructions as system message
  if (instructions) {
    messages.push({ role: "system", content: instructions });
  }

  // String shorthand → single user message
  if (typeof input === "string") {
    messages.push({ role: "user", content: input });
    return messages;
  }

  for (const item of input) {
    // Function call output → tool message
    if ("type" in item && item.type === "function_call_output") {
      const fco = item as ResponsesFunctionCallOutputItem;
      messages.push({
        role: "tool",
        tool_call_id: fco.call_id,
        content: fco.output,
      });
      continue;
    }

    // Function call → assistant message with tool_calls
    if ("type" in item && item.type === "function_call") {
      const fc = item as ResponsesFunctionCallItem;
      messages.push({
        role: "assistant",
        content: null,
        tool_calls: [
          {
            id: fc.call_id,
            type: "function",
            function: { name: fc.name, arguments: fc.arguments },
          },
        ],
      });
      continue;
    }

    // Regular message item
    const msg = item as ResponsesMessageItem;
    const role = msg.role === "developer" ? "system" : msg.role;

    let content: string | null;
    if (typeof msg.content === "string") {
      content = msg.content;
    } else if (Array.isArray(msg.content)) {
      // Extract text from content parts
      content = msg.content
        .filter((p) => p.type === "input_text")
        .map((p) => (p as { type: "input_text"; text: string }).text)
        .join("");
    } else {
      content = null;
    }

    messages.push({ role: role as ChatMessage["role"], content });
  }

  return messages;
}

/**
 * Build an applyResponseFormat-compatible format object from the Responses
 * API `text` field.
 */
function textToResponseFormat(text?: ResponsesRequest["text"]) {
  if (!text?.format || text.format.type === "text") return undefined;
  return text.format;
}

/**
 * Inject JSON mode instruction into messages (same logic as chat.ts).
 */
function applyResponseFormat(
  messages: ChatMessage[],
  responseFormat?: { type: string; json_schema?: { name: string; strict?: boolean; schema: Record<string, unknown> } },
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

// ── Response builders ──────────────────────────────────────────────────────

function buildResponseSkeleton(
  id: string,
  model: string,
  status: ResponsesResponse["status"],
  output: ResponsesOutputItem[],
  usage: { input_tokens: number; output_tokens: number; total_tokens: number },
  params: ResponsesRequest,
): ResponsesResponse {
  return {
    id,
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
    model,
    status,
    output,
    usage,
    error: null,
    incomplete_details: null,
    instructions: params.instructions ?? null,
    metadata: {},
    temperature: params.temperature ?? null,
    top_p: params.top_p ?? null,
    max_output_tokens: params.max_output_tokens ?? null,
    text: { format: { type: params.text?.format?.type ?? "text" } },
  };
}

// ── Main handler ───────────────────────────────────────────────────────────

export async function responses(c: Context<AppEnv>) {
  const config = c.get("config");
  const models = c.get("models");
  const modelId = config.modelId;

  if (!models.model || !models.tokenizer) {
    return c.json(
      { error: { message: "No LLM loaded. Start wandler with --llm to enable this endpoint.", type: "invalid_request_error", param: null, code: null } },
      400,
    );
  }

  const params = await c.req.json<ResponsesRequest>();

  if (params.input === undefined || params.input === null) {
    return c.json(
      { error: { message: "input is required", type: "invalid_request_error", param: "input", code: null } },
      400,
    );
  }

  const respId = makeId("resp");
  const msgId = makeId("msg");

  // Convert Responses API formats to Chat Completions internals
  const messages = applyResponseFormat(
    inputToChatMessages(params.input, params.instructions),
    textToResponseFormat(params.text),
  );
  const chatTools = params.tools?.map(responsesToolToChat);

  // Map max_output_tokens → max_tokens for buildGenOpts
  const samplingParams = { ...params };
  if (params.max_output_tokens != null && params.max_tokens == null) {
    samplingParams.max_tokens = params.max_output_tokens;
  }
  const genOpts = buildGenOpts(samplingParams, models.tokenizer, config.maxTokens, models.maxContextLength, config.prefillChunkSize, models.device);

  // ── Streaming path ──────────────────────────────────────────────────────
  if (params.stream) {
    return streamSSE(c, async (stream) => {
      let fullText = "";
      let promptTokens = 0;
      let completionTokens = 0;
      let toolCallItems: ResponsesOutputFunctionCall[] = [];

      // 1. response.created
      const skeleton = buildResponseSkeleton(
        respId, modelId, "in_progress", [], { input_tokens: 0, output_tokens: 0, total_tokens: 0 }, params,
      );
      await stream.writeSSE({ event: "response.created", data: JSON.stringify({ type: "response.created", response: skeleton }) });

      if (chatTools?.length) {
        // Tool-aware streaming
        const result = await generateStreamWithTools(
          models, modelId, messages, genOpts, chatTools,
          {
            onContent: async (delta) => {
              if (!delta) return;
              // If this is the first content, emit output_item.added + content_part.added
              if (fullText === "") {
                const msgItem: ResponsesOutputMessage = {
                  type: "message", id: msgId, role: "assistant", status: "in_progress",
                  content: [{ type: "output_text", text: "", annotations: [] }],
                };
                await stream.writeSSE({ event: "response.output_item.added", data: JSON.stringify({ type: "response.output_item.added", output_index: 0, item: msgItem }) });
                await stream.writeSSE({ event: "response.content_part.added", data: JSON.stringify({ type: "response.content_part.added", item_id: msgId, output_index: 0, content_index: 0, part: { type: "output_text", text: "", annotations: [] } }) });
              }
              fullText += delta;
              await stream.writeSSE({ event: "response.output_text.delta", data: JSON.stringify({ type: "response.output_text.delta", item_id: msgId, output_index: 0, content_index: 0, delta }) });
            },
            onToolCalls: async (calls) => {
              // Close text content if any was emitted
              if (fullText !== "") {
                await stream.writeSSE({ event: "response.output_text.done", data: JSON.stringify({ type: "response.output_text.done", item_id: msgId, output_index: 0, content_index: 0, text: fullText }) });
                await stream.writeSSE({ event: "response.output_item.done", data: JSON.stringify({
                  type: "response.output_item.done", output_index: 0,
                  item: { type: "message", id: msgId, role: "assistant", status: "completed", content: [{ type: "output_text", text: fullText, annotations: [] }] },
                }) });
              }

              // Emit each tool call
              for (let i = 0; i < calls.length; i++) {
                const tc = calls[i]!;
                const fcItem = toolCallToFunctionCall(tc);
                toolCallItems.push(fcItem);
                const oi = fullText !== "" ? i + 1 : i;

                // output_item.added — skeleton with empty arguments
                await stream.writeSSE({ event: "response.output_item.added", data: JSON.stringify({
                  type: "response.output_item.added", output_index: oi,
                  item: { type: "function_call", id: fcItem.id, call_id: fcItem.call_id, name: fcItem.name, arguments: "", status: "in_progress" },
                }) });

                // function_call_arguments.delta — full arguments in one chunk
                await stream.writeSSE({ event: "response.function_call_arguments.delta", data: JSON.stringify({
                  type: "response.function_call_arguments.delta", item_id: fcItem.id, output_index: oi, delta: fcItem.arguments,
                }) });

                // function_call_arguments.done
                await stream.writeSSE({ event: "response.function_call_arguments.done", data: JSON.stringify({
                  type: "response.function_call_arguments.done", item_id: fcItem.id, output_index: oi, name: fcItem.name, arguments: fcItem.arguments,
                }) });

                // output_item.done — complete item
                await stream.writeSSE({ event: "response.output_item.done", data: JSON.stringify({
                  type: "response.output_item.done", output_index: oi,
                  item: { ...fcItem, status: "completed" },
                }) });
              }
            },
          },
        );
        promptTokens = result.promptTokens;
        completionTokens = result.completionTokens;
      } else {
        // Simple text streaming (no tools)

        // 2. output_item.added — message skeleton
        const msgItem: ResponsesOutputMessage = {
          type: "message", id: msgId, role: "assistant", status: "in_progress",
          content: [{ type: "output_text", text: "", annotations: [] }],
        };
        await stream.writeSSE({ event: "response.output_item.added", data: JSON.stringify({ type: "response.output_item.added", output_index: 0, item: msgItem }) });

        // 3. content_part.added
        await stream.writeSSE({ event: "response.content_part.added", data: JSON.stringify({
          type: "response.content_part.added", item_id: msgId, output_index: 0, content_index: 0,
          part: { type: "output_text", text: "", annotations: [] },
        }) });

        // 4. Stream text deltas
        const result = await generateStreamTokens(
          models, modelId, messages, genOpts,
          async (token) => {
            fullText += token;
            await stream.writeSSE({ event: "response.output_text.delta", data: JSON.stringify({
              type: "response.output_text.delta", item_id: msgId, output_index: 0, content_index: 0, delta: token,
            }) });
          },
        );
        promptTokens = result.promptTokens;
        completionTokens = result.completionTokens;
      }

      // 5. output_text.done (only if we emitted text and didn't already close it via tool path)
      if (fullText !== "" && toolCallItems.length === 0) {
        await stream.writeSSE({ event: "response.output_text.done", data: JSON.stringify({
          type: "response.output_text.done", item_id: msgId, output_index: 0, content_index: 0, text: fullText,
        }) });

        // 6. output_item.done
        await stream.writeSSE({ event: "response.output_item.done", data: JSON.stringify({
          type: "response.output_item.done", output_index: 0,
          item: { type: "message", id: msgId, role: "assistant", status: "completed", content: [{ type: "output_text", text: fullText, annotations: [] }] },
        }) });
      }

      // If no text was emitted and no tool calls either (empty response), still close the message
      if (fullText === "" && toolCallItems.length === 0) {
        await stream.writeSSE({ event: "response.output_item.done", data: JSON.stringify({
          type: "response.output_item.done", output_index: 0,
          item: { type: "message", id: msgId, role: "assistant", status: "completed", content: [{ type: "output_text", text: "", annotations: [] }] },
        }) });
      }

      // 7. response.completed
      const output: ResponsesOutputItem[] = [];
      if (fullText !== "" || toolCallItems.length === 0) {
        output.push({
          type: "message", id: msgId, role: "assistant", status: "completed",
          content: [{ type: "output_text", text: fullText, annotations: [] }],
        });
      }
      output.push(...toolCallItems);

      const usage = { input_tokens: promptTokens, output_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
      const finalResponse = buildResponseSkeleton(respId, modelId, "completed", output, usage, params);
      await stream.writeSSE({ event: "response.completed", data: JSON.stringify({ type: "response.completed", response: finalResponse }) });
    });
  }

  // ── Non-streaming path ────────────────────────────────────────────────────
  const result = await generate(models, modelId, messages, genOpts, chatTools);
  const toolCalls = chatTools?.length ? parseToolCalls(result.text) : null;

  const output: ResponsesOutputItem[] = [];

  if (toolCalls && toolCalls.length > 0) {
    // Tool calls — emit function_call output items
    for (const tc of toolCalls) {
      output.push(toolCallToFunctionCall(tc));
    }
  } else {
    // Text response
    output.push({
      type: "message",
      id: msgId,
      role: "assistant",
      status: "completed",
      content: [{ type: "output_text", text: result.text, annotations: [] }],
    });
  }

  const usage = { input_tokens: result.promptTokens, output_tokens: result.completionTokens, total_tokens: result.promptTokens + result.completionTokens };
  return c.json(buildResponseSkeleton(respId, modelId, "completed", output, usage, params));
}
