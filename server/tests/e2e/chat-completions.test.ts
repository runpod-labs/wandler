import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { TestServer } from "./helpers.js";
import {
  collectSSE,
  createMockModels,
  createTestConfig,
  postJson,
  startTestServer,
  startTestServerWithLoadedModels,
} from "./helpers.js";

describe("POST /v1/chat/completions", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  const chatUrl = () => `${ts.baseUrl}/v1/chat/completions`;

  describe("non-streaming", () => {
    it("returns a chat completion response", async () => {
      const { status, body } = await postJson(chatUrl(), {
        messages: [{ role: "user", content: "Hello" }],
      });

      expect(status).toBe(200);
      const response = body as {
        id: string;
        object: string;
        created: number;
        model: string;
        choices: Array<{
          index: number;
          message: { role: string; content: string };
          finish_reason: string;
        }>;
        usage: {
          prompt_tokens: number;
          completion_tokens: number;
          total_tokens: number;
        };
      };

      expect(response.id).toMatch(/^chatcmpl-/);
      expect(response.object).toBe("chat.completion");
      expect(response.model).toBe("mock-model/test");
      expect(response.choices).toHaveLength(1);
      expect(response.choices[0]!.index).toBe(0);
      expect(response.choices[0]!.message.role).toBe("assistant");
      expect(response.choices[0]!.message.content).toBeTruthy();
      expect(response.choices[0]!.finish_reason).toBe("stop");
      expect(response.usage.prompt_tokens).toBeGreaterThan(0);
      expect(response.usage.completion_tokens).toBeGreaterThan(0);
      expect(response.usage.total_tokens).toBe(
        response.usage.prompt_tokens + response.usage.completion_tokens,
      );
    });

    it("accepts temperature and top_p", async () => {
      const { status } = await postJson(chatUrl(), {
        messages: [{ role: "user", content: "Hi" }],
        temperature: 0.5,
        top_p: 0.9,
      });
      expect(status).toBe(200);
    });

    it("accepts max_tokens", async () => {
      const { status } = await postJson(chatUrl(), {
        messages: [{ role: "user", content: "Hi" }],
        max_tokens: 50,
      });
      expect(status).toBe(200);
    });

    it("rejects requests that exceed the model context before generation", async () => {
      const models = createMockModels();
      models.maxContextLength = 12;
      const limited = await startTestServerWithLoadedModels(models, createTestConfig({ maxTokens: null }));
      try {
        const { status, body } = await postJson(`${limited.baseUrl}/v1/chat/completions`, {
          messages: [{ role: "user", content: "one two three four five six seven eight nine ten eleven twelve" }],
          max_tokens: 4,
        });

        expect(status).toBe(400);
        expect((body as { error?: { code?: string } }).error?.code).toBe("context_length_exceeded");
      } finally {
        await limited.close();
      }
    });

    it("accepts presence_penalty and frequency_penalty", async () => {
      const { status } = await postJson(chatUrl(), {
        messages: [{ role: "user", content: "Hi" }],
        presence_penalty: 0.5,
        frequency_penalty: 0.3,
      });
      expect(status).toBe(200);
    });

    it("accepts response_format json_object", async () => {
      const { status } = await postJson(chatUrl(), {
        messages: [{ role: "user", content: "List items" }],
        response_format: { type: "json_object" },
      });
      expect(status).toBe(200);
    });
  });

  describe("streaming", () => {
    it("returns SSE stream with chunks", async () => {
      const { status, events } = await collectSSE(chatUrl(), {
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
      });

      expect(status).toBe(200);
      expect(events.length).toBeGreaterThan(2); // role + content chunks + final

      // First chunk should have role
      const first = events[0] as {
        choices: Array<{
          delta: { role?: string };
          finish_reason: string | null;
        }>;
      };
      expect(first.choices[0]!.delta.role).toBe("assistant");

      // Last chunk should have finish_reason
      const last = events[events.length - 1] as {
        choices: Array<{ finish_reason: string | null }>;
        usage?: { prompt_tokens: number };
      };
      expect(last.choices[0]!.finish_reason).toBe("stop");
    });

    it("includes usage in final chunk by default", async () => {
      const { events } = await collectSSE(chatUrl(), {
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
      });

      const last = events[events.length - 1] as {
        usage?: {
          prompt_tokens: number;
          completion_tokens: number;
          total_tokens: number;
        };
      };
      expect(last.usage).toBeDefined();
      expect(last.usage!.prompt_tokens).toBeGreaterThan(0);
    });

    it("respects stream_options.include_usage=false", async () => {
      const { events } = await collectSSE(chatUrl(), {
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
        stream_options: { include_usage: false },
      });

      const last = events[events.length - 1] as {
        usage?: unknown;
      };
      expect(last.usage).toBeUndefined();
    });

    it("all chunks have correct object type", async () => {
      const { events } = await collectSSE(chatUrl(), {
        messages: [{ role: "user", content: "Hello" }],
        stream: true,
      });

      for (const event of events) {
        expect((event as { object: string }).object).toBe(
          "chat.completion.chunk",
        );
      }
    });
  });

  describe("error handling", () => {
    it("returns 400 for invalid JSON", async () => {
      const res = await fetch(chatUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "not json",
      });
      expect(res.status).toBe(400);
      const body = (await res.json()) as {
        error: { message: string; type: string };
      };
      expect(body.error.message).toBe("Invalid JSON");
    });

    it("returns 400 for missing messages", async () => {
      const { status, body } = await postJson(chatUrl(), {});
      expect(status).toBe(400);
      expect(
        (body as { error: { message: string } }).error.message,
      ).toBe("messages is required");
    });

    it("returns 400 for empty messages array", async () => {
      const { status } = await postJson(chatUrl(), { messages: [] });
      expect(status).toBe(400);
    });
  });

  describe("tool calling", () => {
    it("accepts tools parameter without error", async () => {
      const { status } = await postJson(chatUrl(), {
        messages: [{ role: "user", content: "What's the weather?" }],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              description: "Get weather for a city",
              parameters: {
                type: "object",
                properties: {
                  city: { type: "string" },
                },
              },
            },
          },
        ],
      });
      expect(status).toBe(200);
    });
  });
});
