import type { LoadedModels } from "../models/manager.js";
import type { Tokenizer } from "../models/tokenizer.js";
import { formatChat } from "../models/tokenizer.js";
import type { ChatMessage, GenerationOptions, Tool } from "../types/openai.js";
import { getTextContent } from "../utils/content.js";
import { elapsedMs, nowMs } from "./profile.js";
import { resolvePrefillChunkSize } from "./options.js";

export type TensorLike = {
  dims: number[];
  location?: string;
  slice(...args: unknown[]): TensorLike;
  dispose?(): Promise<void>;
};

type CacheTensor = TensorLike;

export interface PrefillResult {
  inputIds: TensorLike;
  pastKeyValues: WandlerDynamicCache | null;
  prefillChunkSize?: number;
  prefillChunks?: number;
  prefillMs?: number;
  prefixCacheHit?: boolean;
  prefixCacheTokens?: number;
  cleanup(): Promise<void>;
}

export interface PrefixCandidate {
  text: string;
  tokens: number;
}

export class WandlerDynamicCache {
  [key: string]: CacheTensor | unknown;

  private shared = new Set<CacheTensor>();

  constructor(entries?: Record<string, CacheTensor>, shared?: Iterable<CacheTensor>) {
    if (shared) this.shared = new Set(shared);
    if (!entries) return;
    this.update(entries);
  }

  get_seq_length(): number {
    for (const [name, tensor] of Object.entries(this)) {
      if (name.startsWith("past_key_values.") && isTensorLike(tensor)) {
        return tensor.dims.at(-2) ?? 0;
      }
    }
    return 0;
  }

  update(entries: Record<string, CacheTensor>): void {
    for (const [name, tensor] of Object.entries(entries)) {
      const old = this[name];
      if (
        isTensorLike(old) &&
        old !== tensor &&
        old.location === "gpu-buffer" &&
        !this.shared.has(old)
      ) {
        void old.dispose?.();
      }
      this[name] = tensor;
      this.shared.delete(tensor);
    }
  }

  cloneSharingValues(): WandlerDynamicCache {
    const entries: Record<string, CacheTensor> = Object.create(null);
    for (const [name, tensor] of Object.entries(this)) {
      if (isTensorLike(tensor)) entries[name] = tensor;
    }
    return new WandlerDynamicCache(entries, Object.values(entries));
  }

  async dispose(): Promise<void> {
    await Promise.all(
      Object.values(this)
        .filter(isTensorLike)
        .filter((tensor) => tensor.location === "gpu-buffer")
        .filter((tensor) => !this.shared.has(tensor))
        .map((tensor) => tensor.dispose?.() ?? Promise.resolve()),
    );
  }
}

function isTensorLike(value: unknown): value is TensorLike {
  return Boolean(
    value &&
    typeof value === "object" &&
    "dims" in value &&
    Array.isArray((value as { dims?: unknown }).dims),
  );
}

export function updatePastKeyValuesFromOutputs(
  outputs: Record<string, unknown>,
  cache: WandlerDynamicCache | null,
): WandlerDynamicCache {
  const entries: Record<string, CacheTensor> = Object.create(null);
  for (const [name, value] of Object.entries(outputs)) {
    if (!name.startsWith("present") || !isTensorLike(value)) continue;
    const newName = name
      .replace("present_ssm", "past_ssm")
      .replace("present_conv", "past_conv")
      .replace("present_recurrent", "past_recurrent")
      .replace("present", "past_key_values");
    entries[newName] = value;
  }
  if (cache) {
    cache.update(entries);
    return cache;
  }
  return new WandlerDynamicCache(entries);
}

export async function disposeUnusedOutputs(
  outputs: Record<string, unknown>,
  cache: WandlerDynamicCache,
  keep: Iterable<unknown> = [],
): Promise<void> {
  const cached = new Set(Object.values(cache));
  for (const value of keep) cached.add(value);
  await Promise.all(
    Object.values(outputs)
      .filter(isTensorLike)
      .filter((tensor) => tensor.location === "gpu-buffer" && !cached.has(tensor))
      .map((tensor) => tensor.dispose?.() ?? Promise.resolve()),
  );
}

function readPrefillChunkSize(
  promptTokens: number,
  raw = process.env.WANDLER_PREFILL_CHUNK_SIZE ?? "auto",
  device?: string | null,
  attentionHeads?: number | null,
): number | null {
  const resolved = resolvePrefillChunkSize(raw, device, promptTokens, attentionHeads);
  if (["0", "false", "off", "no"].includes(resolved.toLowerCase())) return null;
  const chunkSize = Number.parseInt(resolved, 10);
  if (!Number.isFinite(chunkSize) || chunkSize < 2 || chunkSize >= promptTokens) return null;
  return chunkSize;
}

async function prefillPromptCache(
  model: LoadedModels["model"],
  inputIds: TensorLike,
  endToken: number,
  chunkSize: number,
  cache: WandlerDynamicCache | null = null,
  startToken = 0,
): Promise<{ cache: WandlerDynamicCache | null; chunks: number; prefillMs: number }> {
  const m = model as unknown as {
    prepare_inputs_for_generation(inputIds: bigint[][], modelInputs: Record<string, unknown>, generationConfig: Record<string, unknown>): Record<string, unknown>;
    forward(modelInputs: Record<string, unknown>): Promise<Record<string, unknown>>;
  };

  const started = nowMs();
  let chunks = 0;

  for (let start = startToken; start < endToken; start += chunkSize) {
    const end = Math.min(start + chunkSize, endToken);
    const chunkInputIds = inputIds.slice(null, [start, end]);
    let modelInputs: Record<string, unknown> = {
      input_ids: chunkInputIds,
      past_key_values: cache,
    };
    modelInputs = m.prepare_inputs_for_generation([], modelInputs, {});
    const outputs = await m.forward(modelInputs);
    cache = updatePastKeyValuesFromOutputs(outputs, cache);
    await disposeUnusedOutputs(outputs, cache);
    chunks++;
  }

  return { cache, chunks, prefillMs: elapsedMs(started) };
}

interface PrefixCacheEntry {
  key: string;
  tokens: number;
  cache: WandlerDynamicCache;
  refs: number;
  evicted: boolean;
  lastUsed: number;
}

class PrefixCacheStore {
  private entries = new Map<string, PrefixCacheEntry>();

  acquire(prompt: string): { entry: PrefixCacheEntry; cache: WandlerDynamicCache } | null {
    const now = Date.now();
    let best: PrefixCacheEntry | null = null;
    for (const entry of this.entries.values()) {
      if (!prompt.startsWith(entry.key)) continue;
      if (!best || entry.tokens > best.tokens) best = entry;
    }
    if (!best) return null;
    best.refs++;
    best.lastUsed = now;
    return { entry: best, cache: best.cache.cloneSharingValues() };
  }

  release(entry: PrefixCacheEntry): void {
    entry.refs = Math.max(0, entry.refs - 1);
    if (entry.refs === 0 && entry.evicted) {
      void entry.cache.dispose();
    }
  }

  async put(key: string, tokens: number, cache: WandlerDynamicCache): Promise<void> {
    const existing = this.entries.get(key);
    if (existing) {
      existing.lastUsed = Date.now();
      await cache.dispose();
      return;
    }

    this.entries.set(key, {
      key,
      tokens,
      cache,
      refs: 0,
      evicted: false,
      lastUsed: Date.now(),
    });
    await this.evictIfNeeded();
  }

  private async evictIfNeeded(): Promise<void> {
    const maxEntries = readPrefixCacheEntries();
    while (this.entries.size > maxEntries) {
      const evict = [...this.entries.values()]
        .sort((a, b) => a.lastUsed - b.lastUsed)[0];
      if (!evict) return;
      this.entries.delete(evict.key);
      evict.evicted = true;
      if (evict.refs === 0) await evict.cache.dispose();
    }
  }
}

const prefixCache = new PrefixCacheStore();

function readPrefixCacheEntries(): number {
  const raw = process.env.WANDLER_PREFIX_CACHE_ENTRIES;
  if (raw == null || raw === "") return 2;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 2;
}

function readPrefixCacheMinTokens(): number {
  const raw = process.env.WANDLER_PREFIX_CACHE_MIN_TOKENS;
  if (raw == null || raw === "") return 512;
  const parsed = Number.parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 512;
}

function prefixCacheEnabled(): boolean {
  const raw = process.env.WANDLER_PREFIX_CACHE;
  if (raw == null || raw === "") return true;
  return !["0", "false", "off", "no"].includes(raw.toLowerCase());
}

export function buildPrefixCandidate(
  tokenizer: Tokenizer,
  messages: ChatMessage[],
  modelId: string,
  tools: Tool[] | undefined,
  chatTemplate: string | null,
  fullPrompt: string,
): PrefixCandidate | null {
  if (!prefixCacheEnabled() || messages.length === 0) return null;

  let text = "";
  const prefixMessages = messages.slice(0, -1);
  if (prefixMessages.length > 0) {
    text = formatChat(
      tokenizer,
      prefixMessages,
      modelId,
      tools,
      chatTemplate,
      { addGenerationPrompt: false },
    );
  }

  if (!text || !fullPrompt.startsWith(text)) {
    const lastContent = getTextContent(messages[messages.length - 1]!.content);
    const idx = lastContent ? fullPrompt.lastIndexOf(lastContent) : -1;
    text = idx > 0 ? fullPrompt.slice(0, idx) : "";
  }
  if (!text || !fullPrompt.startsWith(text)) return null;

  const tokens = tokenizer(text, { return_tensors: "pt" }).input_ids.dims[1]!;
  if (tokens < readPrefixCacheMinTokens()) return null;
  return { text, tokens };
}

export async function preparePrefill(
  models: LoadedModels,
  inputIds: TensorLike,
  promptTokens: number,
  genOpts: GenerationOptions,
  fullPrompt?: string,
  prefixCandidate?: PrefixCandidate | null,
): Promise<PrefillResult> {
  const chunkSize = readPrefillChunkSize(
    promptTokens,
    genOpts.prefill_chunk_size,
    models.device,
    models.attentionHeads,
  );
  if (!chunkSize) {
    return {
      inputIds,
      pastKeyValues: null,
      cleanup: async () => {},
    };
  }

  let cache: WandlerDynamicCache | null = null;
  let prefillChunks = 0;
  let prefillMs = 0;
  let startToken = 0;
  let prefixCacheHit = false;
  let prefixCacheTokens: number | undefined;
  let acquired: { entry: PrefixCacheEntry; cache: WandlerDynamicCache } | null = null;

  if (fullPrompt && prefixCandidate) {
    acquired = prefixCache.acquire(fullPrompt);
    if (acquired) {
      cache = acquired.cache;
      startToken = acquired.entry.tokens;
      prefixCacheHit = true;
      prefixCacheTokens = acquired.entry.tokens;
    } else if (prefixCandidate.tokens < promptTokens - 1) {
      const prefixPrefill = await prefillPromptCache(
        models.model,
        inputIds,
        prefixCandidate.tokens,
        chunkSize,
      );
      if (prefixPrefill.cache) {
        prefillChunks += prefixPrefill.chunks;
        prefillMs += prefixPrefill.prefillMs;
        await prefixCache.put(prefixCandidate.text, prefixCandidate.tokens, prefixPrefill.cache);
        const cached = prefixCache.acquire(fullPrompt);
        if (cached) {
          acquired = cached;
          cache = cached.cache;
          startToken = prefixCandidate.tokens;
          prefixCacheTokens = prefixCandidate.tokens;
        }
      }
    }
  }

  const prefillEnd = promptTokens - 1;
  if (startToken < prefillEnd) {
    const suffixPrefill = await prefillPromptCache(
      models.model,
      inputIds,
      prefillEnd,
      chunkSize,
      cache,
      startToken,
    );
    cache = suffixPrefill.cache;
    prefillChunks += suffixPrefill.chunks;
    prefillMs += suffixPrefill.prefillMs;
  }

  return {
    inputIds: inputIds.slice(null, [promptTokens - 1, promptTokens]),
    pastKeyValues: cache,
    prefillChunkSize: chunkSize,
    prefillChunks,
    prefillMs,
    prefixCacheHit,
    prefixCacheTokens,
    cleanup: async () => {
      await cache?.dispose();
      if (acquired) prefixCache.release(acquired.entry);
    },
  };
}
