import type { ServerConfig } from "../config.js";
import type { LoadedModels } from "../models/manager.js";
import { generate } from "./generate.js";

export interface WarmupResult {
  enabled: boolean;
  promptTokens?: number;
  completionTokens?: number;
  totalMs?: number;
  error?: string;
}

function warmupTokenBudget(config: ServerConfig, models: LoadedModels): number {
  if (!config.maxTokens && !models.maxContextLength && !config.warmupTokens) return config.warmupTokens;
  const contextLimit = config.maxTokens ?? models.maxContextLength ?? Number.POSITIVE_INFINITY;
  const reserved = config.warmupMaxNewTokens + 32;
  return Math.max(0, Math.min(config.warmupTokens, contextLimit - reserved));
}

export async function warmupLLM(config: ServerConfig, models: LoadedModels): Promise<WarmupResult> {
  if (!config.modelId || !models.model || !models.tokenizer || config.warmupTokens <= 0) {
    return { enabled: false };
  }

  const tokenBudget = warmupTokenBudget(config, models);
  if (tokenBudget <= 0) {
    return { enabled: false };
  }

  const started = Date.now();
  try {
    const result = await generate(
      models,
      config.modelId,
      [{ role: "user", content: Array(tokenBudget).fill("warmup").join(" ") }],
      {
        max_new_tokens: config.warmupMaxNewTokens,
        temperature: 0,
        top_p: 1,
        do_sample: false,
        prefill_chunk_size: config.prefillChunkSize,
      },
    );
    return {
      enabled: true,
      promptTokens: result.promptTokens,
      completionTokens: result.completionTokens,
      totalMs: Date.now() - started,
    };
  } catch (error) {
    return {
      enabled: true,
      totalMs: Date.now() - started,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
