import { describe, expect, it } from "vitest";
import { buildGenOpts } from "../../../src/generation/options.js";
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
  it("returns defaults for empty params", () => {
    const opts = buildGenOpts({}, mockTokenizer);
    expect(opts.max_new_tokens).toBe(2048);
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
});
