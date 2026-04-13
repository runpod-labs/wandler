import type { GenerationOptions, SamplingParams } from "../types/openai.js";
import type { Tokenizer } from "../models/tokenizer.js";

/**
 * Build transformers.js generation options from OpenAI-compatible sampling params.
 * Shared by both /v1/chat/completions and /v1/completions.
 */
export function buildGenOpts(params: SamplingParams, tokenizer: Tokenizer, maxTokensCap?: number): GenerationOptions {
  const temperature = params.temperature ?? 0.7;
  const serverMax = maxTokensCap || 2048;
  const requested = params.max_tokens || serverMax;
  const opts: GenerationOptions = {
    max_new_tokens: Math.min(requested, serverMax),
    temperature,
    top_p: params.top_p ?? 0.95,
    do_sample: temperature > 0,
  };

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
