import { describe, expect, it } from "vitest";
import {
  buildGenOpts,
  FALLBACK_MAX_TOKENS,
  resolvePrefillChunkSize,
  WEBGPU_FULL_PREFILL_MAX_TOKENS,
} from "../../../src/generation/options.js";
import type { Tokenizer } from "../../../src/models/tokenizer.js";
import type { SamplingParams } from "../../../src/types/openai.js";

const mockTokenizer = Object.assign(
  (text: string, _opts: Record<string, unknown>) => ({
    input_ids: { dims: [1, text.split(/\s+/).length] },
  }),
  {
    apply_chat_template: () => "",
    batch_decode: () => [""],
  },
) as unknown as Tokenizer;

describe("buildGenOpts", () => {
  it("falls back to FALLBACK_MAX_TOKENS when no server cap and no model context", () => {
    const opts = buildGenOpts({}, mockTokenizer);
    expect(opts.max_new_tokens).toBe(FALLBACK_MAX_TOKENS);
    expect(opts.temperature).toBe(0.7);
    expect(opts.top_p).toBe(0.95);
    expect(opts.do_sample).toBe(true);
  });

  it("sets do_sample to false when temperature is 0", () => {
    const opts = buildGenOpts({ temperature: 0 }, mockTokenizer);
    expect(opts.do_sample).toBe(false);
    expect(opts.temperature).toBe(0);
  });

  it("passes top_k when provided", () => {
    const opts = buildGenOpts({ top_k: 50 }, mockTokenizer);
    expect(opts.top_k).toBe(50);
  });

  it("passes min_p when provided", () => {
    const opts = buildGenOpts({ min_p: 0.05 }, mockTokenizer);
    expect(opts.min_p).toBe(0.05);
  });

  it("passes typical_p when provided", () => {
    const opts = buildGenOpts({ typical_p: 0.9 }, mockTokenizer);
    expect(opts.typical_p).toBe(0.9);
  });

  it("passes no_repeat_ngram_size when provided", () => {
    const opts = buildGenOpts({ no_repeat_ngram_size: 3 }, mockTokenizer);
    expect(opts.no_repeat_ngram_size).toBe(3);
  });

  it("uses repetition_penalty directly when provided", () => {
    const opts = buildGenOpts({ repetition_penalty: 1.5 }, mockTokenizer);
    expect(opts.repetition_penalty).toBe(1.5);
  });

  it("maps presence_penalty to repetition_penalty", () => {
    const opts = buildGenOpts({ presence_penalty: 1.0 }, mockTokenizer);
    expect(opts.repetition_penalty).toBe(1.5);
  });

  it("maps combined penalties to repetition_penalty", () => {
    const opts = buildGenOpts(
      { presence_penalty: 0.5, frequency_penalty: 0.5 } as SamplingParams,
      mockTokenizer,
    );
    expect(opts.repetition_penalty).toBe(1.5);
  });

  it("prefers explicit repetition_penalty over mapped penalties", () => {
    const opts = buildGenOpts(
      { repetition_penalty: 1.2, presence_penalty: 1.0 },
      mockTokenizer,
    );
    expect(opts.repetition_penalty).toBe(1.2);
  });

  it("does not set repetition_penalty when penalties are zero", () => {
    const opts = buildGenOpts(
      { presence_penalty: 0, frequency_penalty: 0 },
      mockTokenizer,
    );
    expect(opts.repetition_penalty).toBeUndefined();
  });

  // ── max_new_tokens resolution ──────────────────────────────────────────

  it("uses model max context when server cap is unset", () => {
    const opts = buildGenOpts({}, mockTokenizer, null, 32768);
    expect(opts.max_new_tokens).toBe(32768);
  });

  it("uses explicit server cap when set (takes precedence over model context)", () => {
    const opts = buildGenOpts({}, mockTokenizer, 4096, 131072);
    expect(opts.max_new_tokens).toBe(4096);
  });

  it("honors client max_tokens below the effective cap", () => {
    const opts = buildGenOpts({ max_tokens: 100 }, mockTokenizer, null, 131072);
    expect(opts.max_new_tokens).toBe(100);
  });

  it("caps client max_tokens that exceeds the server cap", () => {
    const opts = buildGenOpts({ max_tokens: 999999 }, mockTokenizer, 4096, 131072);
    expect(opts.max_new_tokens).toBe(4096);
  });

  it("caps client max_tokens that exceeds the model context", () => {
    const opts = buildGenOpts({ max_tokens: 999999 }, mockTokenizer, null, 131072);
    expect(opts.max_new_tokens).toBe(131072);
  });

  it("treats undefined server cap and model context as fallback", () => {
    const opts = buildGenOpts({ max_tokens: 500 }, mockTokenizer, undefined, undefined);
    expect(opts.max_new_tokens).toBe(500);
  });

  it("max_tokens=0 from the client is treated as 0, not 'unset'", () => {
    // Using `??` instead of `||` so explicit 0 is respected rather than
    // silently replaced with the effective cap.
    const opts = buildGenOpts({ max_tokens: 0 }, mockTokenizer, null, 131072);
    expect(opts.max_new_tokens).toBe(0);
  });

  it("passes server prefill chunk size through to generation", () => {
    const opts = buildGenOpts({}, mockTokenizer, null, 131072, "1024");
    expect(opts.prefill_chunk_size).toBe("1024");
  });

  it("passes auto prefill through until prompt length is known", () => {
    const opts = buildGenOpts({}, mockTokenizer, null, 131072, "auto");
    expect(opts.prefill_chunk_size).toBe("auto");
  });

  it("disables auto prefill chunking for small WebGPU prompts only", () => {
    expect(resolvePrefillChunkSize("auto", "webgpu", WEBGPU_FULL_PREFILL_MAX_TOKENS)).toBe("0");
    expect(resolvePrefillChunkSize("auto", "webgpu", WEBGPU_FULL_PREFILL_MAX_TOKENS + 1)).toBe("1024");
  });

  it("keeps auto prefill chunking on non-WebGPU backends and unknown prompt lengths", () => {
    expect(resolvePrefillChunkSize("auto", "webgpu")).toBe("1024");
    expect(resolvePrefillChunkSize("auto", "cuda", 2048)).toBe("1024");
    expect(resolvePrefillChunkSize("auto", "cpu", 2048)).toBe("1024");
  });
});
