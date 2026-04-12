import { describe, expect, it } from "vitest";
import { formatChat, formatGemmaChat } from "../../../src/models/tokenizer.js";
import type { Tokenizer } from "../../../src/models/tokenizer.js";
import type { ChatMessage } from "../../../src/types/openai.js";

describe("formatGemmaChat", () => {
  it("formats a simple user message", () => {
    const messages: ChatMessage[] = [
      { role: "user", content: "Hello" },
    ];
    const result = formatGemmaChat(messages);
    expect(result).toBe(
      "<start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n",
    );
  });

  it("prepends system message to first user message", () => {
    const messages: ChatMessage[] = [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hi" },
    ];
    const result = formatGemmaChat(messages);
    expect(result).toContain("You are helpful.");
    expect(result).toContain("Hi");
    expect(result.endsWith("<start_of_turn>model\n")).toBe(true);
  });

  it("formats multi-turn conversation", () => {
    const messages: ChatMessage[] = [
      { role: "user", content: "What is 2+2?" },
      { role: "assistant", content: "4" },
      { role: "user", content: "And 3+3?" },
    ];
    const result = formatGemmaChat(messages);
    expect(result).toContain("<start_of_turn>user\nWhat is 2+2?<end_of_turn>");
    expect(result).toContain("<start_of_turn>model\n4<end_of_turn>");
    expect(result).toContain("<start_of_turn>user\nAnd 3+3?<end_of_turn>");
    expect(result.endsWith("<start_of_turn>model\n")).toBe(true);
  });
});

describe("formatChat", () => {
  it("uses tokenizer.apply_chat_template when available", () => {
    const mockTokenizer = {
      apply_chat_template: (_messages: ChatMessage[], _opts: Record<string, unknown>) =>
        "<formatted>",
    } as unknown as Tokenizer;

    const messages: ChatMessage[] = [{ role: "user", content: "Hello" }];
    const result = formatChat(mockTokenizer, messages, "test-model");
    expect(result).toBe("<formatted>");
  });

  it("passes tools to template when provided", () => {
    let capturedOpts: Record<string, unknown> | null = null;
    const mockTokenizer = {
      apply_chat_template: (_messages: ChatMessage[], opts: Record<string, unknown>) => {
        capturedOpts = opts;
        return "<formatted>";
      },
    } as unknown as Tokenizer;

    const tools = [
      { type: "function" as const, function: { name: "test", parameters: {} } },
    ];
    formatChat(mockTokenizer, [{ role: "user", content: "Hi" }], "test-model", tools);
    expect(capturedOpts).toHaveProperty("tools", tools);
  });

  it("falls back to Gemma format for gemma models", () => {
    const mockTokenizer = {
      apply_chat_template: () => {
        throw new Error("no template");
      },
    } as unknown as Tokenizer;

    const messages: ChatMessage[] = [{ role: "user", content: "Hello" }];
    const result = formatChat(mockTokenizer, messages, "onnx-community/gemma-4-ONNX");
    expect(result).toContain("<start_of_turn>user");
    expect(result).toContain("<start_of_turn>model");
  });

  it("falls back to generic format for unknown models", () => {
    const mockTokenizer = {
      apply_chat_template: () => {
        throw new Error("no template");
      },
    } as unknown as Tokenizer;

    const messages: ChatMessage[] = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi" },
    ];
    const result = formatChat(mockTokenizer, messages, "some-unknown-model");
    expect(result).toContain("User: Hello");
    expect(result).toContain("Assistant: Hi");
    expect(result.endsWith("\nAssistant: ")).toBe(true);
  });
});
