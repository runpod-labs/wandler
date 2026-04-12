import { afterAll, beforeAll, describe, expect, it } from "vitest";
import OpenAI from "openai";
import type { TestServer } from "./helpers.js";
import { startTestServer } from "./helpers.js";

describe("OpenAI SDK compatibility", () => {
  let ts: TestServer;
  let client: OpenAI;

  beforeAll(async () => {
    ts = await startTestServer();
    client = new OpenAI({
      baseURL: `${ts.baseUrl}/v1`,
      apiKey: "test-key", // not validated
    });
  });

  afterAll(async () => {
    await ts.close();
  });

  it("client.models.list() returns models", async () => {
    const models = await client.models.list();
    const list = [];
    for await (const model of models) {
      list.push(model);
    }
    expect(list.length).toBeGreaterThan(0);
    expect(list[0]!.object).toBe("model");
    expect(list[0]!.id).toBe("mock-model/test");
  });

  it("client.chat.completions.create() non-streaming", async () => {
    const completion = await client.chat.completions.create({
      model: "mock-model/test",
      messages: [{ role: "user", content: "Hello" }],
    });

    expect(completion.id).toMatch(/^chatcmpl-/);
    expect(completion.object).toBe("chat.completion");
    expect(completion.choices).toHaveLength(1);
    expect(completion.choices[0]!.message.role).toBe("assistant");
    expect(completion.choices[0]!.message.content).toBeTruthy();
    expect(completion.choices[0]!.finish_reason).toBe("stop");
    expect(completion.usage!.prompt_tokens).toBeGreaterThan(0);
    expect(completion.usage!.completion_tokens).toBeGreaterThan(0);
  });

  it("client.chat.completions.create() streaming", async () => {
    const stream = await client.chat.completions.create({
      model: "mock-model/test",
      messages: [{ role: "user", content: "Hello" }],
      stream: true,
    });

    const chunks: OpenAI.Chat.Completions.ChatCompletionChunk[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(1);
    // First chunk has role
    expect(chunks[0]!.choices[0]!.delta.role).toBe("assistant");
    // Last chunk has finish_reason
    const last = chunks[chunks.length - 1]!;
    expect(last.choices[0]!.finish_reason).toBe("stop");
  });

  it("client.chat.completions.create() with temperature and max_tokens", async () => {
    const completion = await client.chat.completions.create({
      model: "mock-model/test",
      messages: [{ role: "user", content: "Hi" }],
      temperature: 0.5,
      max_tokens: 100,
      top_p: 0.9,
    });

    expect(completion.choices[0]!.message.content).toBeTruthy();
  });

  it("client.completions.create() non-streaming", async () => {
    const completion = await client.completions.create({
      model: "mock-model/test",
      prompt: "Once upon a time",
    });

    expect(completion.id).toMatch(/^cmpl-/);
    expect(completion.object).toBe("text_completion");
    expect(completion.choices).toHaveLength(1);
    expect(completion.choices[0]!.text).toBeTruthy();
    expect(completion.choices[0]!.finish_reason).toBe("stop");
  });

  it("client.completions.create() streaming", async () => {
    const stream = await client.completions.create({
      model: "mock-model/test",
      prompt: "Once upon a time",
      stream: true,
    });

    const chunks: OpenAI.Completions.Completion[] = [];
    for await (const chunk of stream) {
      chunks.push(chunk);
    }

    expect(chunks.length).toBeGreaterThan(1);
    const last = chunks[chunks.length - 1]!;
    expect(last.choices[0]!.finish_reason).toBe("stop");
  });

  it("client.embeddings.create() returns embeddings", async () => {
    const result = await client.embeddings.create({
      model: "mock-embedding/test",
      input: "Hello world",
    });

    expect(result.object).toBe("list");
    expect(result.data).toHaveLength(1);
    expect(result.data[0]!.object).toBe("embedding");
    expect(Array.isArray(result.data[0]!.embedding)).toBe(true);
    expect(result.data[0]!.embedding.length).toBeGreaterThan(0);
    expect(result.usage.prompt_tokens).toBeGreaterThan(0);
  });

  it("client.embeddings.create() with array input", async () => {
    const result = await client.embeddings.create({
      model: "mock-embedding/test",
      input: ["Hello", "World"],
    });

    expect(result.data).toHaveLength(2);
    expect(result.data[0]!.index).toBe(0);
    expect(result.data[1]!.index).toBe(1);
  });
});
