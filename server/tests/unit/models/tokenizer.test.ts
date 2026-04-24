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

  it("recursively defaults nested object properties without type", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "read",
          description: "read a file",
          parameters: {
            type: "object",
            properties: {
              range: {
                type: "object",
                properties: {
                  start: { description: "start line" }, // no type
                  end: { description: "end line" }, // no type
                },
              },
            },
          },
        },
      },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    const sent = captured.tools as typeof tools;
    const rangeProps = (sent[0]!.function.parameters as any).properties.range.properties;
    expect(rangeProps.start).toEqual({ description: "start line", type: "string" });
    expect(rangeProps.end).toEqual({ description: "end line", type: "string" });
  });

  it("recursively defaults array item properties without type", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "batch_read",
          parameters: {
            type: "object",
            properties: {
              files: {
                type: "array",
                items: {
                  type: "object",
                  properties: {
                    path: { description: "file path" }, // no type
                  },
                },
              },
            },
          },
        },
      },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    const sent = captured.tools as typeof tools;
    const itemProps = (sent[0]!.function.parameters as any).properties.files.items.properties;
    expect(itemProps.path).toEqual({ description: "file path", type: "string" });
  });

  it("recursively defaults union schema properties without type", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "flexible",
          parameters: {
            type: "object",
            properties: {
              value: {
                anyOf: [
                  {
                    type: "object",
                    properties: {
                      id: { description: "numeric id" }, // no type
                    },
                  },
                  { type: "string" },
                ],
              },
            },
          },
        },
      },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    const sent = captured.tools as typeof tools;
    const anyOfSchemas = (sent[0]!.function.parameters as any).properties.value.anyOf;
    expect(anyOfSchemas[0]!.properties.id).toEqual({ description: "numeric id", type: "string" });
  });

  // ── Empty `properties` fallback (patternProperties bug) ─────────────────
  // Gemma's template has a fallback branch for type:"object" descriptors
  // without a `properties` key — it recursively iterates every OTHER key in
  // the descriptor as if it were a sub-property. That crashes on things
  // like `patternProperties` and `additionalProperties` because their
  // values aren't property descriptors. Real-world example: OpenClaw's
  // `exec` tool has `env: { type: "object", patternProperties: { "^.*$":
  // { type: "string" } } }` with no `properties` key — that hit the
  // fallback and crashed even after the nested-type fix was in place.
  // Sanitizer must inject an empty `properties: {}` so the primary branch
  // always handles it.
  it("injects empty properties:{} when type is object but properties is missing", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "exec",
          parameters: {
            type: "object",
            properties: {
              env: {
                type: "object",
                patternProperties: { "^.*$": { type: "string" } },
              },
            },
          },
        },
      },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    const sent = captured.tools as typeof tools;
    const env = (sent[0]!.function.parameters as {
      properties: { env: Record<string, unknown> };
    }).properties.env;
    // properties must exist (so the template's primary branch handles it)
    expect(env.properties).toEqual({});
    // patternProperties is preserved
    expect(env.patternProperties).toEqual({ "^.*$": { type: "string" } });
  });

  it("injects empty properties:{} on an object descriptor whose type was inferred", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "wrap",
          parameters: {
            type: "object",
            properties: {
              // No `type`, but has `additionalProperties` so type is
              // inferred as "object" — then we must also inject properties.
              cfg: {
                additionalProperties: true,
                description: "arbitrary config",
              },
            },
          },
        },
      },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    const sent = captured.tools as typeof tools;
    const cfg = (sent[0]!.function.parameters as {
      properties: { cfg: Record<string, unknown> };
    }).properties.cfg;
    // additionalProperties:true means we infer object even though neither
    // properties nor items exists — so after sanitization, properties
    // should exist so the Jinja fallback is avoided.
    // (inference for this case falls back to "string" because there's no
    // `.properties` or `.items` hint; that's fine — no crash either way
    // since strings don't hit the object fallback branch. Confirm we
    // don't corrupt the descriptor.)
    expect(cfg.description).toBe("arbitrary config");
    expect(cfg.additionalProperties).toBe(true);
    expect("type" in cfg).toBe(true);
  });

  it("recursively defaults additionalProperties when it's an object schema", () => {
    const { tokenizer, captured } = captureTools();
    const tools = [
      {
        type: "function" as const,
        function: {
          name: "flexible_map",
          parameters: {
            type: "object",
            properties: {
              metadata: {
                type: "object",
                additionalProperties: {
                  type: "object",
                  properties: {
                    value: { description: "some value" }, // no type
                  },
                },
              },
            },
          },
        },
      },
    ];

    formatChat(tokenizer, [{ role: "user", content: "hi" }], "gemma", tools);

    const sent = captured.tools as typeof tools;
    const additionalProps = (sent[0]!.function.parameters as any).properties.metadata
      .additionalProperties.properties;
    expect(additionalProps.value).toEqual({ description: "some value", type: "string" });
  });
});
