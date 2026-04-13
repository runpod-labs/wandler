import { describe, expect, it } from "vitest";
import { formatChat } from "../../../src/models/tokenizer.js";
import type { Tokenizer } from "../../../src/models/tokenizer.js";
import type { ChatMessage } from "../../../src/types/openai.js";

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

  it("passes external chat template when provided", () => {
    let capturedOpts: Record<string, unknown> | null = null;
    const mockTokenizer = {
      apply_chat_template: (_messages: ChatMessage[], opts: Record<string, unknown>) => {
        capturedOpts = opts;
        return "<formatted>";
      },
    } as unknown as Tokenizer;

    const template = "{{ messages | map('content') | join('\\n') }}";
    const messages: ChatMessage[] = [{ role: "user", content: "Hello" }];
    formatChat(mockTokenizer, messages, "test-model", undefined, template);
    expect(capturedOpts).toHaveProperty("chat_template", template);
  });

  it("does not set chat_template option when no external template", () => {
    let capturedOpts: Record<string, unknown> | null = null;
    const mockTokenizer = {
      apply_chat_template: (_messages: ChatMessage[], opts: Record<string, unknown>) => {
        capturedOpts = opts;
        return "<formatted>";
      },
    } as unknown as Tokenizer;

    formatChat(mockTokenizer, [{ role: "user", content: "Hi" }], "test-model");
    expect(capturedOpts).not.toHaveProperty("chat_template");
  });

  it("normalizes multimodal content to text strings", () => {
    let capturedMessages: ChatMessage[] | null = null;
    const mockTokenizer = {
      apply_chat_template: (messages: ChatMessage[], _opts: Record<string, unknown>) => {
        capturedMessages = messages;
        return "<formatted>";
      },
    } as unknown as Tokenizer;

    const messages: ChatMessage[] = [{
      role: "user",
      content: [
        { type: "text", text: "What is this?" },
        { type: "image_url", image_url: { url: "https://example.com/img.png" } },
      ],
    }];
    formatChat(mockTokenizer, messages, "test-model");
    // Content should be normalized to string (text parts only)
    expect(capturedMessages![0]!.content).toBe("What is this?");
  });
});
