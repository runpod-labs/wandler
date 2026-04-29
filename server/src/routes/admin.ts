import type { Context } from "hono";
import type { AppEnv } from "../server.js";
import type { GenerationProfile } from "../types/openai.js";

const startedAt = Date.now();
let totalRequests = 0;
let activeRequests = 0;
let peakActiveRequests = 0;
let totalPromptTokens = 0;
let totalTokensGenerated = 0;
let totalLatencyMs = 0;
let streamedRequests = 0;
let toolRequests = 0;
let totalErrors = 0;

const RECENT_REQUEST_LIMIT = 25;

export interface RequestMetrics {
  route: string;
  promptTokens?: number;
  completionTokens?: number;
  totalMs?: number;
  stream?: boolean;
  toolsCount?: number;
  statusCode?: number;
  generationProfile?: GenerationProfile;
}

interface RecentRequest extends Required<Omit<RequestMetrics, "generationProfile">> {
  created_at: string;
  generationProfile?: GenerationProfile;
}

const recentRequests: RecentRequest[] = [];

export function resetMetrics(): void {
  totalRequests = 0;
  activeRequests = 0;
  peakActiveRequests = 0;
  totalPromptTokens = 0;
  totalTokensGenerated = 0;
  totalLatencyMs = 0;
  streamedRequests = 0;
  toolRequests = 0;
  totalErrors = 0;
  recentRequests.length = 0;
}

export function requestStarted(): void {
  activeRequests++;
  peakActiveRequests = Math.max(peakActiveRequests, activeRequests);
}

export function trackRequest(metrics: RequestMetrics): void {
  activeRequests = Math.max(0, activeRequests - 1);
  totalRequests++;
  totalPromptTokens += metrics.promptTokens ?? 0;
  totalTokensGenerated += metrics.completionTokens ?? 0;
  totalLatencyMs += metrics.totalMs ?? 0;
  if (metrics.stream) streamedRequests++;
  if ((metrics.toolsCount ?? 0) > 0) toolRequests++;
  if ((metrics.statusCode ?? 200) >= 400) totalErrors++;

  recentRequests.unshift({
    created_at: new Date().toISOString(),
    route: metrics.route,
    promptTokens: metrics.promptTokens ?? 0,
    completionTokens: metrics.completionTokens ?? 0,
    totalMs: metrics.totalMs ?? 0,
    stream: metrics.stream ?? false,
    toolsCount: metrics.toolsCount ?? 0,
    statusCode: metrics.statusCode ?? 200,
    generationProfile: metrics.generationProfile,
  });
  if (recentRequests.length > RECENT_REQUEST_LIMIT) {
    recentRequests.length = RECENT_REQUEST_LIMIT;
  }
}

export function trackFailedRequest(
  route: string,
  statusCode = 500,
  generationProfile?: GenerationProfile,
): void {
  trackRequest({
    route,
    statusCode,
    promptTokens: generationProfile?.promptTokens,
    completionTokens: generationProfile?.completionTokens,
    generationProfile,
  });
}

export function adminMetrics(c: Context<AppEnv>) {
  const config = c.get("config");
  const mem = process.memoryUsage();
  return c.json({
    uptime_seconds: Math.round((Date.now() - startedAt) / 1000),
    total_requests: totalRequests,
    active_requests: activeRequests,
    peak_active_requests: peakActiveRequests,
    total_prompt_tokens: totalPromptTokens,
    total_tokens_generated: totalTokensGenerated,
    average_latency_ms: totalRequests > 0 ? Math.round(totalLatencyMs / totalRequests) : 0,
    streamed_requests: streamedRequests,
    tool_requests: toolRequests,
    total_errors: totalErrors,
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
    recent_requests: recentRequests,
  });
}
