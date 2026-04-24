import { afterEach, describe, expect, it } from "vitest";
import type { TestServer } from "./helpers.js";
import { collectSSE, startTestServerWithChunks } from "./helpers.js";

// ── End-to-end tests for streaming tool calls ───────────────────────────────
//
// These tests drive the HTTP path with a scripted mock model that emits
// pre-decoded chunks via the streamer's `callback_function`. Each test spins
// up a fresh server with its own script so chunks are deterministic.

interface ChatChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    delta: {
      role?: string;
      content?: string;
      tool_calls?: Array<{
        id: string;
        type: string;
        function: { name: string; arguments: string };
      }>;
    };
    finish_reason: string | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

const WEATHER_TOOL = {
  type: "function" as const,
  function: {
    name: "get_weather",
    description: "Get the weather for a city",
    parameters: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
};

describe("POST /v1/chat/completions — streaming tool calls", () => {
  let ts: TestServer | null = null;

  afterEach(async () => {
    await ts?.close();
    ts = null;
  });

  const chatUrl = () => `${ts!.baseUrl}/v1/chat/completions`;

  it("emits a tool_calls delta when the model produces a Qwen-format call", async () => {
    // Model emits: "Checking…<tool_call>{"name":"get_weather","arguments":{"city":"LON"}}</tool_call>"
    // Split across chunks to exercise the safety-buffer / opener-lock logic.
    ts = await startTestServerWithChunks([
      "Checking", "…",
      "<tool_", "call>",
      '{"name":"get_weather",',
      '"arguments":{"city":"LON"}}',
      "</tool_", "call>",
    ]);

    const { status, events } = await collectSSE(chatUrl(), {
      messages: [{ role: "user", content: "What's the weather in London?" }],
      tools: [WEATHER_TOOL],
      stream: true,
    });

    expect(status).toBe(200);

    const chunks = events as ChatChunk[];
    // role chunk → some content chunk(s) for "Checking…" → tool_calls chunk → final
    expect(chunks.length).toBeGreaterThanOrEqual(3);

    // First chunk: assistant role
    expect(chunks[0]!.choices[0]!.delta.role).toBe("assistant");

    // Find the tool_calls delta
    const toolCallChunk = chunks.find((e) => e.choices[0]!.delta.tool_calls);
    expect(toolCallChunk).toBeDefined();
    const calls = toolCallChunk!.choices[0]!.delta.tool_calls!;
    expect(calls).toHaveLength(1);
    expect(calls[0]!.type).toBe("function");
    expect(calls[0]!.function.name).toBe("get_weather");
    expect(JSON.parse(calls[0]!.function.arguments)).toEqual({ city: "LON" });

    // Final chunk: finish_reason=tool_calls
    const last = chunks[chunks.length - 1]!;
    expect(last.choices[0]!.finish_reason).toBe("tool_calls");
    expect(last.usage).toBeDefined();
    expect(last.usage!.completion_tokens).toBeGreaterThan(0);

    // Content chunks that came before the tool call should contain "Checking"
    // but must never leak the `<tool_call>` marker.
    const contentChunks = chunks
      .map((e) => e.choices[0]!.delta.content ?? "")
      .filter(Boolean);
    const joined = contentChunks.join("");
    expect(joined).toContain("Checking");
    expect(joined).not.toContain("<tool_call");
    expect(joined).not.toContain("get_weather");
  });

  it("emits plain content when tools are present but the model doesn't call one", async () => {
    ts = await startTestServerWithChunks([
      "The ", "weather ", "is ", "sunny ", "today.",
    ]);

    const { status, events } = await collectSSE(chatUrl(), {
      messages: [{ role: "user", content: "Weather?" }],
      tools: [WEATHER_TOOL],
      stream: true,
    });

    expect(status).toBe(200);

    const chunks = events as ChatChunk[];
    const contentChunks = chunks
      .map((e) => e.choices[0]!.delta.content ?? "")
      .filter(Boolean);
    expect(contentChunks.join("")).toBe("The weather is sunny today.");

    // No tool_calls delta should appear.
    expect(
      chunks.some((e) => e.choices[0]!.delta.tool_calls),
    ).toBe(false);

    // Final chunk: finish_reason=stop
    const last = chunks[chunks.length - 1]!;
    expect(last.choices[0]!.finish_reason).toBe("stop");
  });

  it("never leaks a partial opener split at the chunk boundary", async () => {
    // Opener `<tool_call>` is split across two chunks — the first chunk alone
    // looks like content, so the safety buffer is what keeps it off the wire.
    ts = await startTestServerWithChunks([
      "ok ",
      "<tool_",           // partial — must NOT appear in the SSE output
      "call>",
      '{"name":"get_weather","arguments":{}}',
      "</tool_call>",
    ]);

    const { events } = await collectSSE(chatUrl(), {
      messages: [{ role: "user", content: "hi" }],
      tools: [WEATHER_TOOL],
      stream: true,
    });

    const chunks = events as ChatChunk[];
    const joinedContent = chunks
      .map((e) => e.choices[0]!.delta.content ?? "")
      .join("");
    expect(joinedContent).not.toMatch(/<tool/);

    // The tool call should still be reported.
    const toolCallChunk = chunks.find((e) => e.choices[0]!.delta.tool_calls);
    expect(toolCallChunk).toBeDefined();
    expect(toolCallChunk!.choices[0]!.delta.tool_calls![0]!.function.name).toBe(
      "get_weather",
    );
  });

  it("parses a Gemma call:name{…} tool call", async () => {
    ts = await startTestServerWithChunks([
      "Sure. ",
      "call:", "get_weather",
      "{city:NYC}",
    ]);

    const { events } = await collectSSE(chatUrl(), {
      messages: [{ role: "user", content: "weather?" }],
      tools: [WEATHER_TOOL],
      stream: true,
    });

    const chunks = events as ChatChunk[];
    const toolCallChunk = chunks.find((e) => e.choices[0]!.delta.tool_calls);
    expect(toolCallChunk).toBeDefined();
    expect(toolCallChunk!.choices[0]!.delta.tool_calls![0]!.function.name).toBe(
      "get_weather",
    );

    const content = chunks
      .map((e) => e.choices[0]!.delta.content ?? "")
      .join("");
    expect(content).toContain("Sure");
    expect(content).not.toContain("call:");

    const last = chunks[chunks.length - 1]!;
    expect(last.choices[0]!.finish_reason).toBe("tool_calls");
  });

  it("parses an OpenAI-JSON tool call and preserves the enclosing brace", async () => {
    // Regression guard for the opener-lock issue: `"tool_calls"` alone loses
    // the outer `{`, so `parseOpenAiJson`'s `/\{…\}/` would fail to match.
    // The opener now includes the leading brace.
    ts = await startTestServerWithChunks([
      '{"tool_calls":',
      '[{"function":{"name":"get_weather",',
      '"arguments":"{\\"city\\":\\"NYC\\"}"}}]}',
    ]);

    const { events } = await collectSSE(chatUrl(), {
      messages: [{ role: "user", content: "weather?" }],
      tools: [WEATHER_TOOL],
      stream: true,
    });

    const chunks = events as ChatChunk[];
    const toolCallChunk = chunks.find((e) => e.choices[0]!.delta.tool_calls);
    expect(toolCallChunk).toBeDefined();
    expect(toolCallChunk!.choices[0]!.delta.tool_calls![0]!.function.name).toBe(
      "get_weather",
    );
  });

  it("flushes a false-positive opener as content when no tool call materialises", async () => {
    // Model writes "I would call: someone later." — the `call:` substring
    // triggers an opener lock, but generation ends without a matching
    // `{…}` body. The locked tail must be flushed as plain content.
    ts = await startTestServerWithChunks([
      "I would ", "call:", " someone ", "later.",
    ]);

    const { events } = await collectSSE(chatUrl(), {
      messages: [{ role: "user", content: "what would you do?" }],
      tools: [WEATHER_TOOL],
      stream: true,
    });

    const chunks = events as ChatChunk[];
    const content = chunks
      .map((e) => e.choices[0]!.delta.content ?? "")
      .join("");
    expect(content).toBe("I would call: someone later.");

    expect(chunks.some((e) => e.choices[0]!.delta.tool_calls)).toBe(false);
    expect(chunks[chunks.length - 1]!.choices[0]!.finish_reason).toBe("stop");
  });
});
