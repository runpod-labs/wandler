import { afterEach, describe, expect, it, vi } from "vitest";
import { TransformersJsBackend } from "../../../src/backends/transformersjs.js";
import { WandlerBackend } from "../../../src/backends/wandler.js";
import type { LoadedModels } from "../../../src/models/manager.js";
import type { Tokenizer } from "../../../src/models/tokenizer.js";

type TensorLike = {
  dims: number[];
  data?: ArrayLike<number | bigint>;
  tolist?: () => bigint[][];
  slice(dim: unknown, range: unknown): TensorLike;
  to?: (dtype: string) => TensorLike;
};

type ModelStats = {
  forwardInputTokens: number[];
  generateInputTokens: number[];
  generateUsedPast: boolean[];
  forwardUsedPast: boolean[];
};

function tensor(tokens: number): TensorLike {
  const data = Array.from({ length: tokens }, (_, index) => BigInt(index + 1));
  return {
    dims: [1, tokens],
    data,
    tolist: () => [data],
    slice(_dim: unknown, range: unknown) {
      const [start, end] = range as [number, number];
      return tensor(Math.max(0, end - start));
    },
    to: () => tensor(tokens),
  };
}

function logits(nextToken: number): TensorLike {
  const data = new Float32Array(32);
  data[nextToken] = 100;
  return {
    dims: [1, 1, data.length],
    data,
    tolist: () => [Array.from(data, BigInt)],
    slice: () => logits(nextToken),
    to: () => logits(nextToken),
  };
}

function tokenizer(): Tokenizer {
  const tok = Object.assign(
    (text: string) => ({
      input_ids: tensor(text.trim() ? text.trim().split(/\s+/).length : 0),
    }),
    {
      apply_chat_template: () => "",
      batch_decode: () => ["decoded"],
      all_special_ids: [],
      decode: () => "token ",
    },
  );
  return tok as unknown as Tokenizer;
}

function createModels(): { models: LoadedModels; stats: ModelStats } {
  const stats: ModelStats = {
    forwardInputTokens: [],
    generateInputTokens: [],
    generateUsedPast: [],
    forwardUsedPast: [],
  };
  let decodeToken = 10;
  const model = {
    prepare_inputs_for_generation(
      _inputIds: bigint[][],
      modelInputs: Record<string, unknown>,
      _generationConfig: Record<string, unknown>,
    ) {
      return modelInputs;
    },
    async forward(modelInputs: Record<string, unknown>) {
      const inputIds = modelInputs.input_ids as TensorLike;
      stats.forwardInputTokens.push(inputIds.dims[1] ?? 0);
      stats.forwardUsedPast.push(Boolean(modelInputs.past_key_values));
      return { present: tensor(1), logits: logits(decodeToken++) };
    },
    async generate(opts: Record<string, unknown>) {
      const inputIds = opts.input_ids as TensorLike;
      const promptTokens = inputIds.dims[1] ?? 0;
      stats.generateInputTokens.push(promptTokens);
      stats.generateUsedPast.push(Boolean(opts.past_key_values));
      return {
        dims: [1, promptTokens + 4],
        slice(_dim: unknown, _range: unknown) {
          return tensor(4);
        },
      };
    },
  };

  return {
    stats,
    models: {
      device: "cpu",
      tokenizer: tokenizer(),
      chatTemplate: null,
      processor: null,
      isVision: false,
      model: model as unknown as LoadedModels["model"],
      transcriber: null,
      embedder: null,
      maxContextLength: 2048,
      vocabSize: 32000,
      generationDiagnostics: {
        numLogitsToKeepInput: false,
        numLogitsToKeepPatchedSessions: [],
      },
      attentionHeads: 8,
    },
  };
}

describe("WandlerBackend text engine", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("chunks long prompts and falls back to generate in auto decode mode", async () => {
    const prompt = "one two three four five six seven eight";
    const genOpts = {
      max_new_tokens: 4,
      temperature: 0,
      top_p: 1,
      do_sample: false,
      prefill_chunk_size: "3",
    };

    const wandler = createModels();
    const result = await new WandlerBackend(wandler.models).generateCompletion(prompt, genOpts);

    expect(wandler.stats.forwardInputTokens).toEqual([3, 3, 1]);
    expect(wandler.stats.generateInputTokens).toEqual([1]);
    expect(result.profile?.decodeLoop).toBe(false);
  });

  it("chunks long prompts and owns decode when the custom decode loop is enabled", async () => {
    const prompt = "one two three four five six seven eight";
    const genOpts = {
      max_new_tokens: 4,
      temperature: 0,
      top_p: 1,
      do_sample: false,
      prefill_chunk_size: "3",
    };

    const wandler = createModels();
    const result = await new WandlerBackend(wandler.models, { decodeLoop: "on" }).generateCompletion(prompt, genOpts);

    expect(wandler.stats.forwardInputTokens).toEqual([3, 3, 1, 1, 1, 1, 1]);
    expect(wandler.stats.forwardUsedPast).toEqual([false, true, true, true, true, true, true]);
    expect(wandler.stats.generateInputTokens).toEqual([]);
    expect(wandler.stats.generateUsedPast).toEqual([]);
    expect(result.profile?.decodeLoop).toBe(true);

    const transformers = createModels();
    await new TransformersJsBackend(transformers.models).generateCompletion(prompt, genOpts);

    expect(transformers.stats.forwardInputTokens).toEqual([]);
    expect(transformers.stats.generateInputTokens).toEqual([8]);
    expect(transformers.stats.generateUsedPast).toEqual([false]);
  });

  it("falls back to generate when the custom decode loop is disabled", async () => {
    vi.stubEnv("WANDLER_DECODE_LOOP", "0");
    const wandler = createModels();

    const result = await new WandlerBackend(wandler.models).generateCompletion(
      "one two three four five six seven eight",
      {
        max_new_tokens: 4,
        temperature: 0,
        top_p: 1,
        do_sample: false,
        prefill_chunk_size: "3",
      },
    );

    expect(wandler.stats.forwardInputTokens).toEqual([3, 3, 1]);
    expect(wandler.stats.generateInputTokens).toEqual([1]);
    expect(wandler.stats.generateUsedPast).toEqual([true]);
    expect(result.profile?.decodeLoop).toBe(false);
  });

  it("streams through the owned decode loop without using generate", async () => {
    const wandler = createModels();
    const tokens: string[] = [];

    const result = await new WandlerBackend(wandler.models, { decodeLoop: "on" }).streamCompletion(
      "one two three four five six seven eight",
      {
        max_new_tokens: 4,
        temperature: 0,
        top_p: 1,
        do_sample: false,
        prefill_chunk_size: "3",
      },
      (token) => {
        tokens.push(token);
      },
    );

    expect(result.profile?.decodeLoop).toBe(true);
    expect(result.completionTokens).toBe(4);
    expect(tokens.join("")).toContain("token");
    expect(wandler.stats.generateInputTokens).toEqual([]);
  });
});
