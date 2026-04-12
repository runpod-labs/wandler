import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { TestServer } from "./helpers.js";
import { postJson, startTestServer } from "./helpers.js";

describe("POST /v1/embeddings", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  const url = () => `${ts.baseUrl}/v1/embeddings`;

  it("returns embedding for single input", async () => {
    const { status, body } = await postJson(url(), {
      input: "Hello world",
    });

    expect(status).toBe(200);
    const response = body as {
      object: string;
      data: Array<{
        object: string;
        embedding: number[];
        index: number;
      }>;
      usage: { prompt_tokens: number; total_tokens: number };
    };

    expect(response.object).toBe("list");
    expect(response.data).toHaveLength(1);
    expect(response.data[0]!.object).toBe("embedding");
    expect(response.data[0]!.index).toBe(0);
    expect(Array.isArray(response.data[0]!.embedding)).toBe(true);
    expect(response.data[0]!.embedding.length).toBeGreaterThan(0);
    expect(response.usage.prompt_tokens).toBeGreaterThan(0);
  });

  it("returns embeddings for array input", async () => {
    const { status, body } = await postJson(url(), {
      input: ["Hello", "World"],
    });

    expect(status).toBe(200);
    const response = body as {
      data: Array<{ index: number; embedding: number[] }>;
    };
    expect(response.data).toHaveLength(2);
    expect(response.data[0]!.index).toBe(0);
    expect(response.data[1]!.index).toBe(1);
  });

  it("supports base64 encoding format", async () => {
    const { status, body } = await postJson(url(), {
      input: "Hello",
      encoding_format: "base64",
    });

    expect(status).toBe(200);
    const response = body as {
      data: Array<{ embedding: string }>;
    };
    // Base64 should be a string
    expect(typeof response.data[0]!.embedding).toBe("string");
  });

  it("returns 400 for missing input", async () => {
    const { status, body } = await postJson(url(), {});
    expect(status).toBe(400);
    expect(
      (body as { error: { message: string } }).error.message,
    ).toBe("input is required");
  });
});
