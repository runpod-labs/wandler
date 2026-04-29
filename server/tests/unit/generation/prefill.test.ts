import { afterEach, describe, expect, it, vi } from "vitest";
import { buildPrefixCandidate } from "../../../src/generation/prefill.js";
import type { Tokenizer } from "../../../src/models/tokenizer.js";
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
