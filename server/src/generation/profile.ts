import type { LoadedModels } from "../models/manager.js";
import type { GenerationProfile, MemorySnapshot, Tool } from "../types/openai.js";
import { isQuiet, logInfo } from "../utils/logging.js";

export function nowMs(): number {
  return performance.now();
}

export function elapsedMs(start: number): number {
  return Math.round(performance.now() - start);
}

export function memorySnapshot(): MemorySnapshot {
  const mem = process.memoryUsage();
  return {
    rssMb: Math.round(mem.rss / 1024 / 1024),
    heapUsedMb: Math.round(mem.heapUsed / 1024 / 1024),
    heapTotalMb: Math.round(mem.heapTotal / 1024 / 1024),
    externalMb: Math.round(mem.external / 1024 / 1024),
  };
}

export function serializedToolsChars(tools?: Tool[]): number {
  if (!tools?.length) return 0;
  return JSON.stringify(tools).length;
}

export function estimateFullLogitsMb(models: LoadedModels, promptTokens: number): number | null {
  if (!models.vocabSize || promptTokens <= 0) return null;
  // ONNX logits are float32 in the Gemma exports we are diagnosing.
  return Math.round((promptTokens * models.vocabSize * 4) / 1024 / 1024);
}

export function estimateAttentionScoresMb(models: LoadedModels, promptTokens: number): number | null {
  if (!models.attentionHeads || promptTokens <= 0) return null;
  // Gemma ONNX CUDA materialized fp32 [heads, seq, seq] score tensors in tests.
  return Math.round((models.attentionHeads * promptTokens * promptTokens * 4) / 1024 / 1024);
}

function memoryDelta(before: MemorySnapshot, after: MemorySnapshot): string {
  const rss = after.rssMb - before.rssMb;
  const heap = after.heapUsedMb - before.heapUsedMb;
  const external = after.externalMb - before.externalMb;
  return `rss=${after.rssMb}MB(${rss >= 0 ? "+" : ""}${rss}) heap=${after.heapUsedMb}MB(${heap >= 0 ? "+" : ""}${heap}) external=${after.externalMb}MB(${external >= 0 ? "+" : ""}${external})`;
}

export function shouldLogGenerationProfile(profile: GenerationProfile): boolean {
  if (isQuiet()) return false;
  return (
    process.env.WANDLER_LOG_LEVEL === "debug" ||
    profile.promptTokens >= 1024 ||
    profile.toolsCount > 0 ||
    profile.totalMs >= 1000 ||
    profile.memoryAfterGenerate.rssMb - profile.memoryBefore.rssMb >= 256
  );
}

export function logGenerationProfile(profile: GenerationProfile): void {
  if (!shouldLogGenerationProfile(profile)) return;

  const fullLogits = profile.estimatedFullLogitsMb == null
    ? "unknown"
    : `${profile.estimatedFullLogitsMb}MB`;
  const attentionScores = profile.estimatedAttentionScoresMb == null
    ? "unknown"
    : `${profile.estimatedAttentionScoresMb}MB`;
  const patched = profile.numLogitsToKeepPatchedSessions.length
    ? profile.numLogitsToKeepPatchedSessions.join(",")
    : "none";

  logInfo(
    [
      "[wandler] generation profile",
      `path=${profile.path}`,
      `promptTokens=${profile.promptTokens}`,
      `completionTokens=${profile.completionTokens}`,
      `promptChars=${profile.promptChars}`,
      `tools=${profile.toolsCount}`,
      `toolsChars=${profile.toolsChars}`,
      `fullLogitsEstimate=${fullLogits}`,
      `attentionScoresEstimate=${attentionScores}`,
      `numLogitsInput=${profile.numLogitsToKeepInput}`,
      `patchedSessions=${patched}`,
      `formatMs=${profile.formatMs}`,
      `tokenizeMs=${profile.tokenizeMs}`,
      `generateMs=${profile.generateMs}`,
      `decodeMs=${profile.decodeMs}`,
      profile.prefillChunkSize ? `prefillChunkSize=${profile.prefillChunkSize}` : null,
      profile.prefillChunks ? `prefillChunks=${profile.prefillChunks}` : null,
      profile.prefillMs != null ? `prefillMs=${profile.prefillMs}` : null,
      profile.prefixCacheHit != null ? `prefixCache=${profile.prefixCacheHit ? "hit" : "miss"}` : null,
      profile.prefixCacheTokens != null ? `prefixCacheTokens=${profile.prefixCacheTokens}` : null,
      profile.decodeLoop != null ? `decodeLoop=${profile.decodeLoop ? "on" : "off"}` : null,
      `totalMs=${profile.totalMs}`,
      profile.failedStage ? `failedStage=${profile.failedStage}` : null,
      profile.errorMessage ? `error=${JSON.stringify(profile.errorMessage.slice(0, 240))}` : null,
      `memAfterGenerate=${memoryDelta(profile.memoryBefore, profile.memoryAfterGenerate)}`,
      `memAfterDecode=${memoryDelta(profile.memoryBefore, profile.memoryAfterDecode)}`,
    ].filter(Boolean).join(" "),
  );
}

export class GenerationExecutionError extends Error {
  profile: GenerationProfile;
  statusCode: number;

  constructor(error: unknown, profile: GenerationProfile, statusCode = 500) {
    super(error instanceof Error ? error.message : String(error));
    this.name = "GenerationExecutionError";
    this.cause = error;
    this.stack = error instanceof Error ? error.stack : this.stack;
    this.profile = profile;
    this.statusCode = statusCode;
  }
}

export function getGenerationProfile(error: unknown): GenerationProfile | undefined {
  if (error instanceof GenerationExecutionError) return error.profile;
  return undefined;
}

export function getGenerationStatusCode(error: unknown): number | undefined {
  if (error instanceof GenerationExecutionError) return error.statusCode;
  return undefined;
}
