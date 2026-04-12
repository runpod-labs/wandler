import type { Context } from "hono";
import type { AppEnv } from "../server.js";

const startedAt = Date.now();
let totalRequests = 0;
let totalTokensGenerated = 0;

export function trackRequest(completionTokens = 0): void {
  totalRequests++;
  totalTokensGenerated += completionTokens;
}

export function adminMetrics(c: Context<AppEnv>) {
  const config = c.get("config");
  const mem = process.memoryUsage();
  return c.json({
    uptime_seconds: Math.round((Date.now() - startedAt) / 1000),
    total_requests: totalRequests,
    total_tokens_generated: totalTokensGenerated,
    memory: {
      rss_mb: Math.round(mem.rss / 1024 / 1024),
      heap_used_mb: Math.round(mem.heapUsed / 1024 / 1024),
      heap_total_mb: Math.round(mem.heapTotal / 1024 / 1024),
      external_mb: Math.round(mem.external / 1024 / 1024),
    },
    models: {
      llm: config.modelId,
      dtype: config.modelDtype,
      device: config.device,
      stt: config.sttModelId || null,
      embedding: config.embeddingModelId || null,
    },
  });
}
