import { describe, expect, it } from "vitest";
import { TransformersJsBackend } from "../../../src/backends/transformersjs.js";
import { WandlerBackend } from "../../../src/backends/wandler.js";
import type { LoadedModels } from "../../../src/models/manager.js";
import type { Tokenizer } from "../../../src/models/tokenizer.js";

type TensorLike = {
  dims: number[];
  slice(dim: unknown, range: unknown): TensorLike;
};

type ModelStats = {
  forwardInputTokens: number[];
  generateInputTokens: number[];
  generateUsedPast: boolean[];
};

function tensor(tokens: number): TensorLike {
  return {
    dims: [1, tokens],
    slice(_dim: unknown, range: unknown) {
      const [start, end] = range as [number, number];
      return tensor(Math.max(0, end - start));
    },
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
    },
  );
  return tok as unknown as Tokenizer;
}

function createModels(): { models: LoadedModels; stats: ModelStats } {
  const stats: ModelStats = {
    forwardInputTokens: [],
    generateInputTokens: [],
    generateUsedPast: [],
  };
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
      return { present: tensor(1) };
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
  it("chunks long prompts before decode while transformers.js sends the full prompt to generate", async () => {
    const prompt = "one two three four five six seven eight";
    const genOpts = {
      max_new_tokens: 4,
      temperature: 0,
      top_p: 1,
      do_sample: false,
      prefill_chunk_size: "3",
    };

    const wandler = createModels();
    await new WandlerBackend(wandler.models).generateCompletion(prompt, genOpts);

    expect(wandler.stats.forwardInputTokens).toEqual([3, 3, 1]);
    expect(wandler.stats.generateInputTokens).toEqual([1]);
    expect(wandler.stats.generateUsedPast).toEqual([true]);

    const transformers = createModels();
    await new TransformersJsBackend(transformers.models).generateCompletion(prompt, genOpts);

    expect(transformers.stats.forwardInputTokens).toEqual([]);
    expect(transformers.stats.generateInputTokens).toEqual([8]);
    expect(transformers.stats.generateUsedPast).toEqual([false]);
  });
});
