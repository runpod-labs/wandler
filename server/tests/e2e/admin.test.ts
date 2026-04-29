import { afterAll, beforeAll, describe, expect, it } from "vitest";
import type { TestServer } from "./helpers.js";
import { postJson, startTestServer } from "./helpers.js";

describe("GET /admin/metrics", () => {
  let ts: TestServer;

  beforeAll(async () => {
    ts = await startTestServer();
  });

  afterAll(async () => {
    await ts.close();
  });

  it("tracks request counts, tokens, and recent request details", async () => {
    const chatUrl = `${ts.baseUrl}/v1/chat/completions`;
    const metricsUrl = `${ts.baseUrl}/admin/metrics`;

    const completion = await postJson(chatUrl, {
      messages: [{ role: "user", content: "Hello" }],
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

    expect(completion.status).toBe(200);

    const metricsRes = await fetch(metricsUrl);
    expect(metricsRes.status).toBe(200);
    const metrics = await metricsRes.json() as {
      total_requests: number;
      active_requests: number;
      total_prompt_tokens: number;
      total_tokens_generated: number;
      tool_requests: number;
      average_latency_ms: number;
      recent_requests: Array<{
        route: string;
        promptTokens: number;
        completionTokens: number;
        toolsCount: number;
        statusCode: number;
      }>;
    };

    expect(metrics.total_requests).toBe(1);
    expect(metrics.active_requests).toBe(0);
    expect(metrics.total_prompt_tokens).toBeGreaterThan(0);
    expect(metrics.total_tokens_generated).toBeGreaterThan(0);
    expect(metrics.tool_requests).toBe(1);
    expect(metrics.average_latency_ms).toBeGreaterThanOrEqual(0);
    expect(metrics.recent_requests).toHaveLength(1);
    expect(metrics.recent_requests[0]!.route).toBe("/v1/chat/completions");
    expect(metrics.recent_requests[0]!.promptTokens).toBeGreaterThan(0);
    expect(metrics.recent_requests[0]!.completionTokens).toBeGreaterThan(0);
    expect(metrics.recent_requests[0]!.toolsCount).toBe(1);
    expect(metrics.recent_requests[0]!.statusCode).toBe(200);
  });
});
