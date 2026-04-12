import { describe, expect, it } from "vitest";
import { parseToolCalls } from "../../../src/tools/parser.js";

describe("parseToolCalls", () => {
  describe("LFM Pythonic format", () => {
    it("parses [func_name(arg=\"val\")] format", () => {
      const text = '[get_weather(city="San Francisco", unit="celsius")]';
      const result = parseToolCalls(text);
      expect(result).toHaveLength(1);
      expect(result![0]!.type).toBe("function");
      expect(result![0]!.function.name).toBe("get_weather");
      const args = JSON.parse(result![0]!.function.arguments);
      expect(args.city).toBe("San Francisco");
      expect(args.unit).toBe("celsius");
    });

    it("generates unique call IDs", () => {
      const text = '[get_weather(city="NYC")]';
      const r1 = parseToolCalls(text);
      const r2 = parseToolCalls(text);
      expect(r1![0]!.id).not.toBe(r2![0]!.id);
    });

    it("handles single argument", () => {
      const text = '[search(query="hello world")]';
      const result = parseToolCalls(text);
      expect(result).toHaveLength(1);
      expect(result![0]!.function.name).toBe("search");
      expect(JSON.parse(result![0]!.function.arguments)).toEqual({
        query: "hello world",
      });
    });
  });

  describe("LFM JSON format", () => {
    it("parses [tool_calls [{...}]] format", () => {
      const text =
        '[tool_calls [{"name": "get_weather", "arguments": {"city": "London"}}]]';
      const result = parseToolCalls(text);
      expect(result).toHaveLength(1);
      expect(result![0]!.function.name).toBe("get_weather");
      expect(JSON.parse(result![0]!.function.arguments)).toEqual({
        city: "London",
      });
    });

    it("parses multiple tool calls", () => {
      const text =
        '[tool_calls [{"name": "get_weather", "arguments": {"city": "A"}}, {"name": "get_time", "arguments": {"tz": "UTC"}}]]';
      const result = parseToolCalls(text);
      expect(result).toHaveLength(2);
      expect(result![0]!.function.name).toBe("get_weather");
      expect(result![1]!.function.name).toBe("get_time");
    });

    it("handles tool_call_end marker", () => {
      const text =
        '[tool_calls [{"name": "search", "arguments": {"q": "test"}}]<|tool_call_end|>';
      const result = parseToolCalls(text);
      expect(result).toHaveLength(1);
      expect(result![0]!.function.name).toBe("search");
    });
  });

  describe("Qwen format", () => {
    it("parses <tool_call>...</tool_call> format", () => {
      const text =
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>';
      const result = parseToolCalls(text);
      expect(result).toHaveLength(1);
      expect(result![0]!.function.name).toBe("get_weather");
      expect(JSON.parse(result![0]!.function.arguments)).toEqual({
        city: "Tokyo",
      });
    });

    it("handles string arguments", () => {
      const text =
        '<tool_call>{"name": "calc", "arguments": "{\\"x\\": 1}"}</tool_call>';
      const result = parseToolCalls(text);
      expect(result).toHaveLength(1);
      expect(result![0]!.function.name).toBe("calc");
      expect(result![0]!.function.arguments).toBe('{"x": 1}');
    });
  });

  describe("OpenAI JSON format", () => {
    it("parses {\"tool_calls\": [...]} format", () => {
      const text = JSON.stringify({
        tool_calls: [
          {
            function: {
              name: "get_weather",
              arguments: '{"city": "Paris"}',
            },
          },
        ],
      });
      const result = parseToolCalls(text);
      expect(result).toHaveLength(1);
      expect(result![0]!.function.name).toBe("get_weather");
      expect(result![0]!.function.arguments).toBe('{"city": "Paris"}');
    });

    it("handles object arguments (not stringified)", () => {
      const text = JSON.stringify({
        tool_calls: [
          {
            function: {
              name: "search",
              arguments: { query: "test" },
            },
          },
        ],
      });
      const result = parseToolCalls(text);
      expect(result).toHaveLength(1);
      expect(JSON.parse(result![0]!.function.arguments)).toEqual({
        query: "test",
      });
    });
  });

  describe("thinking block removal", () => {
    it("strips <think>...</think> blocks before parsing", () => {
      const text =
        "<think>Let me think about this...</think>\n" +
        '<tool_call>{"name": "search", "arguments": {"q": "test"}}</tool_call>';
      const result = parseToolCalls(text);
      expect(result).toHaveLength(1);
      expect(result![0]!.function.name).toBe("search");
    });

    it("strips multiple thinking blocks", () => {
      const text =
        "<think>First thought</think><think>Second thought</think>" +
        '[search(query="hello")]';
      const result = parseToolCalls(text);
      expect(result).toHaveLength(1);
      expect(result![0]!.function.name).toBe("search");
    });
  });

  describe("no tool calls", () => {
    it("returns null for plain text", () => {
      expect(parseToolCalls("Hello, how are you?")).toBeNull();
    });

    it("returns null for empty string", () => {
      expect(parseToolCalls("")).toBeNull();
    });

    it("returns null for invalid JSON in tool_calls", () => {
      expect(parseToolCalls('{"tool_calls": "not an array"}')).toBeNull();
    });
  });
});
