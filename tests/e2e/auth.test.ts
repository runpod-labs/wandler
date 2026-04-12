import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { TestServer } from "./helpers.js";
import { startTestServer } from "./helpers.js";

describe("API key authentication", () => {
  let ts: TestServer;
  const API_KEY = "test-secret-key-123";

  beforeAll(async () => {
    ts = await startTestServer({ apiKey: API_KEY });
  });

  afterAll(async () => {
    await ts.close();
  });

  it("GET /health is always allowed without auth", async () => {
    const res = await fetch(`${ts.baseUrl}/health`);
    expect(res.status).toBe(200);
  });

  it("rejects requests without API key", async () => {
    const res = await fetch(`${ts.baseUrl}/v1/models`);
    expect(res.status).toBe(401);
    const body = (await res.json()) as { error: { code: string } };
    expect(body.error.code).toBe("invalid_api_key");
  });

  it("rejects requests with wrong API key", async () => {
    const res = await fetch(`${ts.baseUrl}/v1/models`, {
      headers: { Authorization: "Bearer wrong-key" },
    });
    expect(res.status).toBe(401);
  });

  it("accepts requests with correct Bearer token", async () => {
    const res = await fetch(`${ts.baseUrl}/v1/models`, {
      headers: { Authorization: `Bearer ${API_KEY}` },
    });
    expect(res.status).toBe(200);
  });

  it("accepts requests with raw API key (no Bearer prefix)", async () => {
    const res = await fetch(`${ts.baseUrl}/v1/models`, {
      headers: { Authorization: API_KEY },
    });
    expect(res.status).toBe(200);
  });

  it("protects /v1/chat/completions", async () => {
    const res = await fetch(`${ts.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "Hi" }] }),
    });
    expect(res.status).toBe(401);
  });

  it("allows /v1/chat/completions with valid key", async () => {
    const res = await fetch(`${ts.baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${API_KEY}`,
      },
      body: JSON.stringify({ messages: [{ role: "user", content: "Hi" }] }),
    });
    expect(res.status).toBe(200);
  });

  it("protects /admin/metrics", async () => {
    const res = await fetch(`${ts.baseUrl}/admin/metrics`);
    expect(res.status).toBe(401);
  });

  it("allows /admin/metrics with valid key", async () => {
    const res = await fetch(`${ts.baseUrl}/admin/metrics`, {
      headers: { Authorization: `Bearer ${API_KEY}` },
    });
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      uptime_seconds: number;
      total_requests: number;
      memory: { rss_mb: number };
    };
    expect(body.uptime_seconds).toBeGreaterThanOrEqual(0);
    expect(body.memory.rss_mb).toBeGreaterThan(0);
  });
});

describe("No API key configured (open access)", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer({ apiKey: "" });
  });

  afterAll(async () => {
    await ts.close();
  });

  it("allows all requests without auth", async () => {
    const res = await fetch(`${ts.baseUrl}/v1/models`);
    expect(res.status).toBe(200);
  });

  it("allows admin metrics without auth", async () => {
    const res = await fetch(`${ts.baseUrl}/admin/metrics`);
    expect(res.status).toBe(200);
  });
});
