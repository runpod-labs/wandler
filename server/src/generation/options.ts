import type { GenerationOptions, SamplingParams } from "../types/openai.js";
import type { Tokenizer } from "../models/tokenizer.js";

// Hard fallback when neither an explicit `--max-tokens` cap nor a model
// context length is available (e.g. loaded via a custom pipeline that
// doesn't expose `max_position_embeddings`). Exported so tests can assert it.
export const FALLBACK_MAX_TOKENS = 2048;
export const DEFAULT_PREFILL_CHUNK_SIZE = "1024";
export const DEFAULT_WEBGPU_PREFILL_MEMORY_MB = 640;
export const FALLBACK_WEBGPU_FULL_PREFILL_MAX_TOKENS = 4096;

function parseAutoPrefillMemoryMb(raw: string): number {
  const [, budget] = raw.split(":", 2);
  if (!budget) return DEFAULT_WEBGPU_PREFILL_MEMORY_MB;
  const parsed = Number.parseInt(budget, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_WEBGPU_PREFILL_MEMORY_MB;
}

function alignChunkSize(tokens: number): number {
  if (tokens >= 256) return Math.floor(tokens / 256) * 256;
  if (tokens >= 64) return Math.floor(tokens / 64) * 64;
  return Math.max(2, Math.floor(tokens));
}

export function resolvePrefillChunkSize(
  raw = "auto",
  device: string | null | undefined = "auto",
  promptTokens = 0,
  attentionHeads?: number | null,
): string {
  const value = raw.trim().toLowerCase();
  if (value !== "auto" && !value.startsWith("auto:")) return raw;

  if (device !== "webgpu") return DEFAULT_PREFILL_CHUNK_SIZE;

  // transformers.js/ORT WebGPU handles small/medium Gemma prompts faster on
  // the full-prompt path. Use the full path only when the estimated attention
  // score tensor fits the budget; otherwise use the largest budget-fitting
  // chunk to avoid an artificial latency cliff at a fixed token count.
  if (promptTokens <= 0) return DEFAULT_PREFILL_CHUNK_SIZE;
  if (!attentionHeads) {
    return promptTokens <= FALLBACK_WEBGPU_FULL_PREFILL_MAX_TOKENS
      ? "0"
      : DEFAULT_PREFILL_CHUNK_SIZE;
  }

  const budgetMb = parseAutoPrefillMemoryMb(value);
  const fullAttentionMb = (attentionHeads * promptTokens * promptTokens * 4) / 1024 / 1024;
  if (fullAttentionMb <= budgetMb) {
    return "0";
  }

  const budgetBytes = budgetMb * 1024 * 1024;
  const rawChunk = budgetBytes / (attentionHeads * promptTokens * 4);
  const chunkSize = alignChunkSize(Math.min(promptTokens - 1, rawChunk));
  return String(chunkSize);
}

/**
 * Build transformers.js generation options from OpenAI-compatible sampling params.
 * Shared by both /v1/chat/completions and /v1/completions.
 *
 * `max_new_tokens` is picked in this order of precedence:
 *   1. explicit server cap from `--max-tokens` (when operator set it)
 *   2. the loaded model's `max_position_embeddings`
 *   3. `FALLBACK_MAX_TOKENS` (2048) as a last-resort safety default
 * The client's `params.max_tokens` is then capped at that ceiling.
 */
export function buildGenOpts(
  params: SamplingParams,
  tokenizer: Tokenizer,
  maxTokensCap?: number | null,
  modelMaxContext?: number | null,
  prefillChunkSize?: string,
): GenerationOptions {
  const temperature = params.temperature ?? 0.7;
  const hardCap = maxTokensCap ?? modelMaxContext ?? FALLBACK_MAX_TOKENS;
  const requested = params.max_tokens ?? hardCap;
  const opts: GenerationOptions = {
    max_new_tokens: Math.min(requested, hardCap),
    temperature,
    top_p: params.top_p ?? 0.95,
    do_sample: temperature > 0,
  };
  if (prefillChunkSize != null) {
    opts.prefill_chunk_size = prefillChunkSize;
  }

  // Extended sampling params supported by transformers.js
  if (params.top_k != null) opts.top_k = params.top_k;
  if (params.min_p != null) opts.min_p = params.min_p;
  if (params.typical_p != null) opts.typical_p = params.typical_p;
  if (params.no_repeat_ngram_size != null) opts.no_repeat_ngram_size = params.no_repeat_ngram_size;

  // Direct repetition_penalty (vLLM/llama.cpp style, > 1.0 penalizes)
  if (params.repetition_penalty != null) {
    opts.repetition_penalty = params.repetition_penalty;
  } else {
    // Map OpenAI presence_penalty + frequency_penalty to repetition_penalty
    const penalty = (params.presence_penalty ?? 0) + (params.frequency_penalty ?? 0);
    if (penalty !== 0) {
      opts.repetition_penalty = 1.0 + Math.max(0, penalty) * 0.5;
    }
  }

  // Convert stop sequences to extra eos_token_id values
  if (params.stop) {
    const stops = Array.isArray(params.stop) ? params.stop : [params.stop];
    const extraEosIds: number[] = [];
    for (const seq of stops) {
      const encoded = tokenizer(seq, { return_tensors: "pt" });
      const dims = encoded.input_ids.dims;
      if (dims[1]! > 0) {
        const lastIdx = dims[1]! - 1;
        extraEosIds.push(lastIdx);
      }
    }
    if (extraEosIds.length > 0) {
      opts.eos_token_id = extraEosIds;
    }
  }

  return opts;
}

export function stripInternalGenOpts(opts: GenerationOptions): Omit<GenerationOptions, "prefill_chunk_size"> {
  const { prefill_chunk_size: _prefillChunkSize, ...transformersOpts } = opts;
  return transformersOpts;
}
