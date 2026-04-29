import { describe, expect, it } from "vitest";
import {
  findOpenerIndex,
  generateStreamWithTools,
} from "../../../src/generation/stream-tools.js";
import type { LoadedModels } from "../../../src/models/manager.js";
import type { Tokenizer } from "../../../src/models/tokenizer.js";
import type { ChatMessage, GenerationOptions, Tool, ToolCall } from "../../../src/types/openai.js";

// ── findOpenerIndex ─────────────────────────────────────────────────────────

describe("findOpenerIndex", () => {
  it("returns -1 on plain text", () => {
    expect(findOpenerIndex("just some content")).toBe(-1);
  });

  it("detects Qwen <tool_call>", () => {
    expect(findOpenerIndex("sure thing <tool_call>{")).toBe(11);
  });

  it("detects Gemma <|tool_call>", () => {
    expect(findOpenerIndex("ok <|tool_call>call:foo")).toBe(3);
  });

  it("detects LFM JSON [tool_calls", () => {
    expect(findOpenerIndex("[tool_calls [{")).toBe(0);
  });

  it("detects OpenAI JSON {\"tool_calls\" (locks at the brace)", () => {
    // Lock at `{`, not at the `"tool_calls"` substring, so parseOpenAiJson's
    // `/\{...\}/` regex can still match the outer object.
    expect(findOpenerIndex('prefix {"tool_calls": [')).toBe(7);
  });

  it("detects Gemma call:", () => {
    expect(findOpenerIndex("ok call:get_weather{")).toBe(3);
  });

  it("detects LFM Pythonic [name(", () => {
    expect(findOpenerIndex('here it is [get_weather(city="NYC")]')).toBe(11);
  });

  it("picks the earliest opener when multiple appear", () => {
    expect(findOpenerIndex('call:foo {"tool_calls": [')).toBe(0);
  });

  it("does not treat bare brackets as LFM Pythonic", () => {
    // `[1, 2, 3]` is a list — no name, no `(`. Must not trigger.
    expect(findOpenerIndex("array: [1, 2, 3]")).toBe(-1);
  });
});

// ── mock helpers ────────────────────────────────────────────────────────────

/** Tokenizer mock compatible with transformers.js `TextStreamer`. */
function createMockTokenizer(): Tokenizer {
  const tokenizer = function (text: string, _opts?: Record<string, unknown>) {
    const tokens = text.split(/\s+/);
    return { input_ids: { dims: [1, tokens.length] } };
  };
  tokenizer.apply_chat_template = () => "<prompt>";
  tokenizer.batch_decode = () => [""];
  tokenizer.all_special_ids = [];
  tokenizer.decode = () => "";
  return tokenizer as unknown as Tokenizer;
}

/**
 * Mock model whose `.generate()` feeds a pre-scripted string to the streamer,
 * one character-chunk at a time, by calling `put()` with a fake token id per
 * chunk and having the mock tokenizer's `decode` return that chunk. We side-
 * step the transformers.js TextStreamer decoding path entirely by driving
 * `callback_function` directly via a simpler `.generate()` that mimics what
 * TextStreamer would do post-decode.
 */
function mockModelsWithScript(script: string[]): LoadedModels {
  const tokenizer = createMockTokenizer();
  const model = {
    async generate(opts: Record<string, unknown>) {
      const streamer = opts.streamer as { callback_function?: (s: string) => void } | undefined;
      // `TextStreamer` would normally decode bigint token ids into strings
      // and then invoke `callback_function`. For these tests we invoke the
      // callback directly with the scripted chunks — the downstream buffer
      // logic is what we're actually validating.
      const cb = streamer?.callback_function;
      if (cb) for (const chunk of script) cb(chunk);
      return {
        dims: [1, (opts.input_ids as { dims: number[] })?.dims?.[1] ?? 1 + script.length],
        slice: () => ({ data: [] }),
      };
    },
  };
  return {
    device: "cpu",
    tokenizer,
    chatTemplate: null,
    processor: null,
    isVision: false,
    model: model as LoadedModels["model"],
    transcriber: null,
    embedder: null,
    maxContextLength: null,
  };
}

async function runStream(script: string[], tools?: Tool[]): Promise<{
  content: string;
  toolCalls: ToolCall[] | null;
}> {
  const models = mockModelsWithScript(script);
  const messages: ChatMessage[] = [{ role: "user", content: "hi" }];
  const genOpts: GenerationOptions = { max_new_tokens: 16 };
  let content = "";
  let toolCalls: ToolCall[] | null = null;
  await generateStreamWithTools(models, "mock/model", messages, genOpts, tools, {
    onContent: (d) => { content += d; },
    onToolCalls: (c) => { toolCalls = c; },
  });
  return { content, toolCalls };
}

const TOOLS: Tool[] = [
  { type: "function", function: { name: "get_weather", parameters: { type: "object" } } },
];

// ── generateStreamWithTools ─────────────────────────────────────────────────

describe("generateStreamWithTools", () => {
  it("streams plain content when no tool call appears", async () => {
    const { content, toolCalls } = await runStream(
      ["Hello", ", ", "how ", "are ", "you?"],
      TOOLS,
    );
    expect(toolCalls).toBeNull();
    expect(content).toBe("Hello, how are you?");
  });

  it("detects Qwen <tool_call> split across many chunks", async () => {
    // "Checking…<tool_call>{"name":"get_weather","arguments":{}}</tool_call>"
    const { content, toolCalls } = await runStream(
      [
        "Checking", "…",
        "<tool_", "call>",
        '{"name":', '"get_weather",',
        '"arguments":{}}',
        "</tool_", "call>",
      ],
      TOOLS,
    );
    expect(toolCalls).not.toBeNull();
    expect(toolCalls!).toHaveLength(1);
    expect(toolCalls![0]!.function.name).toBe("get_weather");
    // Pre-tool prose should make it through as content.
    expect(content).toContain("Checking");
    // The tool-call markup must never leak into content.
    expect(content).not.toContain("<tool_call>");
    expect(content).not.toContain("name");
  });

  it("never leaks a partial opener as content", async () => {
    // Opener is split exactly at the boundary.
    const { content, toolCalls } = await runStream(
      [
        "ok ",
        "<tool_",        // partial — must be held back
        "call>",          // completes the opener — lock here
        '{"name":"x","arguments":{}}',
        "</tool_call>",
      ],
      TOOLS,
    );
    expect(toolCalls).not.toBeNull();
    expect(content).not.toMatch(/<tool/);
    expect(content.endsWith(" ")).toBe(true); // "ok " flushed
  });

  it("parses Gemma call:name{...} format", async () => {
    const { content, toolCalls } = await runStream(
      ["I'll check. ", "call:", "get_weather", "{city:NYC}"],
      TOOLS,
    );
    expect(toolCalls).not.toBeNull();
    expect(toolCalls![0]!.function.name).toBe("get_weather");
    expect(content).toContain("I'll check");
    expect(content).not.toContain("call:");
  });

  it("parses LFM JSON [tool_calls [...]] format", async () => {
    const { toolCalls } = await runStream(
      [
        "[tool_calls ",
        '[{"name":"get_weather",',
        '"arguments":{"city":"LON"}}',
        "]]",
      ],
      TOOLS,
    );
    expect(toolCalls).not.toBeNull();
    expect(toolCalls![0]!.function.name).toBe("get_weather");
  });

  it("parses LFM Pythonic [name(...)] format", async () => {
    const { content, toolCalls } = await runStream(
      ["Sure! ", "[get_weather(", 'city="LON")]'],
      TOOLS,
    );
    expect(toolCalls).not.toBeNull();
    expect(toolCalls![0]!.function.name).toBe("get_weather");
    expect(content).toContain("Sure");
    expect(content).not.toContain("[get_weather");
  });

  it("parses OpenAI JSON {\"tool_calls\":[...]}", async () => {
    const { toolCalls } = await runStream(
      [
        '{"tool_calls":',
        '[{"function":{"name":"get_weather",',
        '"arguments":"{\\"city\\":\\"NYC\\"}"}}]}',
      ],
      TOOLS,
    );
    expect(toolCalls).not.toBeNull();
    expect(toolCalls![0]!.function.name).toBe("get_weather");
  });

  it("flushes a false-positive lock as content if no tool call materialises", async () => {
    // Model uses "call:" mid-prose but never completes a tool call.
    const { content, toolCalls } = await runStream(
      ["I would call:", " someone ", "later."],
      TOOLS,
    );
    expect(toolCalls).toBeNull();
    expect(content).toBe("I would call: someone later.");
  });

  it("emits exactly one onToolCalls even if late tokens arrive", async () => {
    let calls = 0;
    const models = mockModelsWithScript([
      "<tool_call>",
      '{"name":"foo","arguments":{}}',
      "</tool_call>",
      " trailing garbage",
    ]);
    await generateStreamWithTools(models, "mock/model", [{ role: "user", content: "hi" }], { max_new_tokens: 8 }, TOOLS, {
      onContent: () => {},
      onToolCalls: () => { calls++; },
    });
    expect(calls).toBe(1);
  });
});
