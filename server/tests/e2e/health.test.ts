import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { TestServer } from "./helpers.js";
import { startTestServer } from "./helpers.js";

describe("GET /health", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  it("returns ok status with engine info", async () => {
    const res = await fetch(`${ts.baseUrl}/health`);
    expect(res.status).toBe(200);

    const body = (await res.json()) as {
      status: string;
      engine: string;
      device: string;
      models: Record<string, string>;
    };
    expect(body.status).toBe("ok");
    expect(body.engine).toBe("transformers.js");
    expect(body.device).toBe("cpu");
    expect(body.models.llm).toBe("mock-model/test");
    expect(body.models.stt).toBe("mock-stt/test");
  });

  it("GET / also returns health info", async () => {
    const res = await fetch(ts.baseUrl);
    expect(res.status).toBe(200);
    const body = (await res.json()) as { status: string };
    expect(body.status).toBe("ok");
  });

  it("returns CORS headers", async () => {
    const res = await fetch(`${ts.baseUrl}/health`);
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
  });
});

describe("OPTIONS (CORS preflight)", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  it("returns 204 with CORS headers", async () => {
    const res = await fetch(`${ts.baseUrl}/v1/chat/completions`, {
      method: "OPTIONS",
    });
    expect(res.status).toBe(204);
    expect(res.headers.get("access-control-allow-origin")).toBe("*");
    expect(res.headers.get("access-control-allow-methods")).toContain("POST");
  });
});

describe("404 handling", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  it("returns 404 with error object for unknown paths", async () => {
    const res = await fetch(`${ts.baseUrl}/v1/unknown`);
    expect(res.status).toBe(404);
    const body = (await res.json()) as { error: { message: string } };
    expect(body.error.message).toBe("Not found");
  });
});
