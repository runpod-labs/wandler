import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { TestServer } from "./helpers.js";
import { postJson, startTestServer } from "./helpers.js";

describe("POST /tokenize", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  it("returns token count for input text", async () => {
    const { status, body } = await postJson(`${ts.baseUrl}/tokenize`, {
      input: "Hello world how are you",
    });

    expect(status).toBe(200);
    const response = body as { tokens: number[]; count: number };
    expect(response.count).toBeGreaterThan(0);
    expect(response.tokens).toHaveLength(response.count);
  });

  it("returns 400 for missing input", async () => {
    const { status, body } = await postJson(`${ts.baseUrl}/tokenize`, {});
    expect(status).toBe(400);
    expect(
      (body as { error: { message: string } }).error.message,
    ).toBe("input is required");
  });
});

describe("POST /detokenize", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  it("returns text for token IDs", async () => {
    const { status, body } = await postJson(`${ts.baseUrl}/detokenize`, {
      tokens: [1, 2, 3],
    });

    expect(status).toBe(200);
    const response = body as { text: string };
    expect(typeof response.text).toBe("string");
  });

  it("returns 400 for missing tokens", async () => {
    const { status, body } = await postJson(`${ts.baseUrl}/detokenize`, {});
    expect(status).toBe(400);
    expect(
      (body as { error: { message: string } }).error.message,
    ).toBe("tokens is required");
  });
});
