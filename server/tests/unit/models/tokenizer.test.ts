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

  // ── Gemma tool-schema sanitization ─────────────────────────────────────
  // Gemma's chat_template.jinja calls `value['type'] | upper` on every tool
  // parameter property and crashes when a property has no explicit `type`
  // (which JSON Schema / the OpenAI API permit). `formatChat` defaults any
  // missing `type` to "string" before the template touches it.

  function captureTools(): {
    tokenizer: Tokenizer;
    captured: { tools?: unknown };
  } {
    const captured: { tools?: unknown } = {};
    const tokenizer = {
      apply_chat_template: (_m: ChatMessage[], opts: Record<string, unknown>) => {
        captured.tools = opts.tools;
        return "<formatted>";
      },
    } as unknown as Tokenizer;
    return { tokenizer, captured };
  }

  it("defaults tool property type to 'string' when missing", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "add",
          parameters: {
            type: "object",
            properties: {
              a: { description: "first number" }, // no type
              b: { description: "second number" }, // no type
            },
          },
        },
      },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    const sent = captured.tools as typeof tools;
    expect(sent[0]!.function.parameters).toEqual({
      type: "object",
      properties: {
        a: { description: "first number", type: "string" },
        b: { description: "second number", type: "string" },
      },
    });
  });

  it("leaves explicit property types untouched", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "book",
          parameters: {
            type: "object",
            properties: {
              city: { type: "string", description: "city name" },
              nights: { type: "integer" },
              refundable: { type: "boolean" },
            },
          },
        },
      },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    const sent = captured.tools as typeof tools;
    expect(sent[0]!.function.parameters).toEqual(tools[0]!.function.parameters);
  });

  it("mixes typed and untyped properties correctly", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "search",
          parameters: {
            type: "object",
            properties: {
              query: { type: "string", description: "search query" },
              limit: { description: "max results" }, // no type
            },
            required: ["query"],
          },
        },
      },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    const sent = captured.tools as typeof tools;
    const props = (sent[0]!.function.parameters as { properties: Record<string, Record<string, unknown>> }).properties;
    expect(props.query).toEqual({ type: "string", description: "search query" });
    expect(props.limit).toEqual({ description: "max results", type: "string" });
    // required array preserved
    expect((sent[0]!.function.parameters as { required?: string[] }).required).toEqual(["query"]);
  });

  it("passes through tools without parameters", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      { type: "function" as const, function: { name: "ping" } },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    expect((captured.tools as typeof tools)[0]).toEqual(tools[0]);
  });

  it("passes through tools with empty parameters object", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      { type: "function" as const, function: { name: "ping", parameters: {} } },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    expect((captured.tools as typeof tools)[0]!.function.parameters).toEqual({});
  });

  it("does not mutate the input tools array", () => {
    const { tokenizer } = captureTools();
    const original = {
      type: "function" as const,
      function: {
        name: "add",
        parameters: {
          type: "object",
          properties: { a: { description: "x" } },
        },
      },
    };
    const tools = [original];
    const snapshot = JSON.parse(JSON.stringify(tools));

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    // Caller's input stays intact
    expect(tools).toEqual(snapshot);
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
