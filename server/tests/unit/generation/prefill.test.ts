import { afterEach, describe, expect, it, vi } from "vitest";
import { buildPrefixCandidate, preparePrefill } from "../../../src/generation/prefill.js";
import type { Tokenizer } from "../../../src/models/tokenizer.js";
import type { LoadedModels } from "../../../src/models/manager.js";
import type { ChatMessage, Tool } from "../../../src/types/openai.js";

function tokenizerFor(fullPrompt: string): Tokenizer {
  const tokenizer = Object.assign(
    (text: string) => ({
      input_ids: { dims: [1, text.trim() ? text.trim().split(/\s+/).length : 0] },
    }),
    {
      apply_chat_template: (messages: ChatMessage[], opts: Record<string, unknown>) => {
        const toolText = opts.tools ? "[tools] stable schema [/tools]\n" : "";
        const body = messages.map((m) => `<${m.role}> ${m.content ?? ""}`).join("\n");
        const suffix = opts.add_generation_prompt === false ? "\n" : "\n<assistant>";
        return `${toolText}${body}${suffix}`;
      },
      batch_decode: () => [""],
    },
  );
  tokenizer.apply_chat_template = () => fullPrompt;
  return tokenizer as unknown as Tokenizer;
}

describe("buildPrefixCandidate", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("builds a prefix before the last message for multi-turn prompts", () => {
    vi.stubEnv("WANDLER_PREFIX_CACHE_MIN_TOKENS", "1");
    const messages: ChatMessage[] = [
      { role: "system", content: "stable system prompt" },
      { role: "user", content: "new request" },
    ];
    const prefix = "[tools] stable schema [/tools]\n<system> stable system prompt\n";
    const fullPrompt = `${prefix}<user> new request\n<assistant>`;
    const tokenizer = tokenizerFor(prefix);

    const candidate = buildPrefixCandidate(tokenizer, messages, "model", [tool()], null, fullPrompt);

    expect(candidate?.text).toBe(prefix);
    expect(candidate?.tokens).toBeGreaterThan(0);
  });

  it("falls back to the text before the single user message for tool-heavy prompts", () => {
    vi.stubEnv("WANDLER_PREFIX_CACHE_MIN_TOKENS", "1");
    const messages: ChatMessage[] = [{ role: "user", content: "new request" }];
    const prefix = "[tools] stable schema [/tools]\n<user> ";
    const fullPrompt = `${prefix}new request\n<assistant>`;
    const tokenizer = tokenizerFor("");

    const candidate = buildPrefixCandidate(tokenizer, messages, "model", [tool()], null, fullPrompt);

    expect(candidate?.text).toBe(prefix);
    expect(candidate?.tokens).toBeGreaterThan(0);
  });

  it("caches prefix token counts for repeated tool/system prefixes", () => {
    vi.stubEnv("WANDLER_PREFIX_CACHE_MIN_TOKENS", "1");
    const messages: ChatMessage[] = [{ role: "user", content: "new request" }];
    const prefix = "[tools] stable schema [/tools]\n<user> ";
    const fullPrompt = `${prefix}new request\n<assistant>`;
    let tokenizations = 0;
    const tokenizer = Object.assign(
      (text: string) => {
        tokenizations++;
        return { input_ids: { dims: [1, text.trim() ? text.trim().split(/\s+/).length : 0] } };
      },
      {
        apply_chat_template: () => "",
        batch_decode: () => [""],
      },
    ) as unknown as Tokenizer;

    const first = buildPrefixCandidate(tokenizer, messages, "model-cache-test", [tool()], null, fullPrompt);
    const second = buildPrefixCandidate(tokenizer, messages, "model-cache-test", [tool()], null, fullPrompt);

    expect(first?.tokens).toBe(second?.tokens);
    expect(tokenizations).toBe(1);
  });
});

describe("preparePrefill", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("passes a short suffix directly to generate after caching a repeated prefix", async () => {
    vi.stubEnv("WANDLER_PREFIX_CACHE", "true");
    vi.stubEnv("WANDLER_PREFIX_CACHE_MIN_TOKENS", "1");
    const models = mockModels();
    const inputIds = tensor(12);

    const first = await preparePrefill(
      models,
      inputIds,
      12,
      { max_new_tokens: 1, temperature: 0, top_p: 1, do_sample: false, prefill_chunk_size: "4" },
      "stable prefix user one",
      { text: "stable prefix", tokens: 8 },
    );
    await first.cleanup();

    const second = await preparePrefill(
      models,
      inputIds,
      12,
      { max_new_tokens: 1, temperature: 0, top_p: 1, do_sample: false, prefill_chunk_size: "4" },
      "stable prefix user two",
      { text: "stable prefix", tokens: 8 },
    );

    expect(second.prefixCacheHit).toBe(true);
    expect(second.inputIds.dims[1]).toBe(4);
    expect(models.stats.forwardInputTokens).toEqual([4, 4]);
    await second.cleanup();
  });
});

function tool(): Tool {
  return {
    type: "function",
    function: {
      name: "search",
      description: "Search",
      parameters: { type: "object", properties: { q: { type: "string" } } },
    },
  };
}

type MockTensor = {
  dims: number[];
  slice(dim: unknown, range: unknown): MockTensor;
  dispose(): Promise<void>;
};

function tensor(tokens: number): MockTensor {
  return {
    dims: [1, tokens],
    slice(_dim: unknown, range: unknown) {
      const [start, end] = range as [number, number];
      return tensor(Math.max(0, end - start));
    },
    async dispose() {},
  };
}

function mockModels(): LoadedModels & { stats: { forwardInputTokens: number[] } } {
  const stats = { forwardInputTokens: [] as number[] };
  return {
    stats,
    device: "cpu",
    tokenizer: tokenizerFor(""),
    chatTemplate: null,
    processor: null,
    isVision: false,
    model: {
      prepare_inputs_for_generation: (_inputIds: bigint[][], modelInputs: Record<string, unknown>) => modelInputs,
      async forward(modelInputs: Record<string, unknown>) {
        const inputIds = modelInputs.input_ids as MockTensor;
        stats.forwardInputTokens.push(inputIds.dims[1] ?? 0);
        return { present: tensor(1) };
      },
    } as unknown as LoadedModels["model"],
    transcriber: null,
    embedder: null,
    maxContextLength: 2048,
    vocabSize: 32000,
    generationDiagnostics: {
      numLogitsToKeepInput: false,
      numLogitsToKeepPatchedSessions: [],
    },
    attentionHeads: 8,
  };
}
