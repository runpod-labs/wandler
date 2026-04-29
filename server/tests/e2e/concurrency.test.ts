import { serve } from "@hono/node-server";
import { afterAll, describe, expect, it } from "vitest";
import { createApp } from "../../src/server.js";
import { createMockModels, createTestConfig, postJson, type TestServer } from "./helpers.js";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function startLimitedServer(): Promise<{ ts: TestServer; peakGenerate: () => number }> {
  const config = createTestConfig({ maxConcurrent: 1, timeout: 2_000 });
  const models = createMockModels();
  const originalGenerate = models.model!.generate.bind(models.model);
  let activeGenerate = 0;
  let peakGenerate = 0;

  models.model = {
    ...models.model!,
    async generate(opts: Record<string, unknown>) {
      activeGenerate++;
      peakGenerate = Math.max(peakGenerate, activeGenerate);
      try {
        await sleep(100);
        return await originalGenerate(opts);
      } finally {
        activeGenerate--;
      }
    },
  };

  const app = createApp(config, models);
  return new Promise((resolve) => {
    const server = serve({ fetch: app.fetch, port: 0 }, (info) => {
      resolve({
        ts: {
          baseUrl: `http://localhost:${info.port}`,
          close: () => new Promise<void>((res) => server.close(() => res())),
        },
        peakGenerate: () => peakGenerate,
      });
    });
  });
}

describe("generation concurrency", () => {
  const servers: TestServer[] = [];

  afterAll(async () => {
    await Promise.all(servers.map((server) => server.close()));
  });

  it("serializes generation requests according to maxConcurrent", async () => {
    const { ts, peakGenerate } = await startLimitedServer();
    servers.push(ts);
    const url = `${ts.baseUrl}/v1/chat/completions`;
    const request = {
      messages: [{ role: "user", content: "Hello" }],
      max_tokens: 1,
    };

    const started = Date.now();
    const [first, second] = await Promise.all([
      postJson(url, request),
      postJson(url, request),
    ]);

    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    expect(peakGenerate()).toBe(1);
    expect(Date.now() - started).toBeGreaterThanOrEqual(190);
  });
});
