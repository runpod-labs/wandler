import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { TestServer } from "./helpers.js";
import { collectSSE, postJson, startTestServer } from "./helpers.js";

describe("POST /v1/completions", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  const url = () => `${ts.baseUrl}/v1/completions`;

  describe("non-streaming", () => {
    it("returns a text completion response", async () => {
      const { status, body } = await postJson(url(), {
        prompt: "Once upon a time",
      });

      expect(status).toBe(200);
      const response = body as {
        id: string;
        object: string;
        model: string;
        choices: Array<{
          index: number;
          text: string;
          finish_reason: string;
        }>;
        usage: {
          prompt_tokens: number;
          completion_tokens: number;
          total_tokens: number;
        };
      };

      expect(response.id).toMatch(/^cmpl-/);
      expect(response.object).toBe("text_completion");
      expect(response.choices).toHaveLength(1);
      expect(response.choices[0]!.index).toBe(0);
      expect(response.choices[0]!.text).toBeTruthy();
      expect(response.choices[0]!.finish_reason).toBe("stop");
      expect(response.usage.prompt_tokens).toBeGreaterThan(0);
    });

    it("accepts prompt as array", async () => {
      const { status, body } = await postJson(url(), {
        prompt: ["Hello", "World"],
      });

      expect(status).toBe(200);
      const response = body as {
        choices: Array<{ index: number; text: string }>;
      };
      expect(response.choices).toHaveLength(2);
      expect(response.choices[0]!.index).toBe(0);
      expect(response.choices[1]!.index).toBe(1);
    });

    it("supports echo parameter", async () => {
      const { status, body } = await postJson(url(), {
        prompt: "Test prompt",
        echo: true,
      });

      expect(status).toBe(200);
      const response = body as {
        choices: Array<{ text: string }>;
      };
      // Echo should include the original prompt in the response
      expect(response.choices[0]!.text).toContain("Test prompt");
    });

    it("supports suffix parameter", async () => {
      const { status, body } = await postJson(url(), {
        prompt: "Hello",
        suffix: " END",
      });

      expect(status).toBe(200);
      const response = body as {
        choices: Array<{ text: string }>;
      };
      expect(response.choices[0]!.text.endsWith(" END")).toBe(true);
    });
  });

  describe("streaming", () => {
    it("returns SSE stream", async () => {
      const { status, events } = await collectSSE(url(), {
        prompt: "Once upon a time",
        stream: true,
      });

      expect(status).toBe(200);
      expect(events.length).toBeGreaterThan(1);

      // All chunks should have text_completion object
      for (const event of events) {
        expect((event as { object: string }).object).toBe("text_completion");
      }

      // Last chunk should have finish_reason
      const last = events[events.length - 1] as {
        choices: Array<{ finish_reason: string | null }>;
      };
      expect(last.choices[0]!.finish_reason).toBe("stop");
    });
  });

  describe("error handling", () => {
    it("returns 400 for missing prompt", async () => {
      const { status, body } = await postJson(url(), {});
      expect(status).toBe(400);
      expect(
        (body as { error: { message: string } }).error.message,
      ).toBe("prompt is required");
    });

    it("returns 400 for invalid JSON", async () => {
      const res = await fetch(url(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "not json",
      });
      expect(res.status).toBe(400);
    });
  });

  describe("extended parameters", () => {
    it("accepts top_k parameter", async () => {
      const { status } = await postJson(url(), {
        prompt: "Hello",
        top_k: 50,
      });
      expect(status).toBe(200);
    });

    it("accepts repetition_penalty parameter", async () => {
      const { status } = await postJson(url(), {
        prompt: "Hello",
        repetition_penalty: 1.2,
      });
      expect(status).toBe(200);
    });
  });
});
