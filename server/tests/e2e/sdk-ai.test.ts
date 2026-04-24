import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { generateText, streamText, embed, embedMany } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import type { TestServer } from "./helpers.js";
import { startTestServer } from "./helpers.js";

describe("Vercel AI SDK compatibility", () => {
  let ts: TestServer;
  let provider: ReturnType<typeof createOpenAI>;

  beforeAll(async () => {
    ts = await startTestServer();
    provider = createOpenAI({
      baseURL: `${ts.baseUrl}/v1`,
      apiKey: "test-key",
    });
  });

  afterAll(async () => {
    await ts.close();
  });

  // ── Responses API path: provider("model") ─────────────────────────────

  describe("Responses API — provider('model')", () => {
    it("generateText() returns text via /v1/responses", async () => {
      const result = await generateText({
        model: provider("mock-model/test"),
        prompt: "Hello",
      });

      expect(result.text).toBeTruthy();
      expect(result.finishReason).toBe("stop");
      expect(result.usage).toBeDefined();
    });

    it("generateText() with system message via /v1/responses", async () => {
      const result = await generateText({
        model: provider("mock-model/test"),
        system: "You are helpful.",
        prompt: "Hi",
      });

      expect(result.text).toBeTruthy();
    });

    it("streamText() streams tokens via /v1/responses", async () => {
      const result = streamText({
        model: provider("mock-model/test"),
        prompt: "Hello",
      });

      const chunks: string[] = [];
      for await (const chunk of result.textStream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);
      const fullText = chunks.join("");
      expect(fullText).toBeTruthy();
    });
  });

  // ── Chat Completions path: provider.chat("model") ─────────────────────

  describe("Chat Completions — provider.chat('model')", () => {
    it("generateText() returns text", async () => {
      const result = await generateText({
        model: provider.chat("mock-model/test"),
        prompt: "Hello",
      });

      expect(result.text).toBeTruthy();
      expect(result.finishReason).toBe("stop");
      expect(result.usage).toBeDefined();
    });

    it("generateText() with system message", async () => {
      const result = await generateText({
        model: provider.chat("mock-model/test"),
        system: "You are helpful.",
        prompt: "Hi",
      });

      expect(result.text).toBeTruthy();
    });

    it("streamText() streams tokens", async () => {
      const result = streamText({
        model: provider.chat("mock-model/test"),
        prompt: "Hello",
      });

      const chunks: string[] = [];
      for await (const chunk of result.textStream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);
      const fullText = chunks.join("");
      expect(fullText).toBeTruthy();
    });
  });

  // ── Embeddings ─────────────────────────────────────────────────────────

  it("embed() returns embedding vector", async () => {
    const result = await embed({
      model: provider.embedding("mock-embedding/test"),
      value: "Hello world",
    });

    expect(result.embedding).toBeTruthy();
    expect(Array.isArray(result.embedding)).toBe(true);
    expect(result.embedding.length).toBeGreaterThan(0);
  });

  it("embedMany() returns multiple embeddings", async () => {
    const result = await embedMany({
      model: provider.embedding("mock-embedding/test"),
      values: ["Hello", "World"],
    });

    expect(result.embeddings).toHaveLength(2);
    expect(result.embeddings[0]!.length).toBeGreaterThan(0);
    expect(result.embeddings[1]!.length).toBeGreaterThan(0);
  });
});
