import { TextStreamer } from "@huggingface/transformers";
import type { LoadedModels } from "../models/manager.js";
import type { ChatMessage, GenerationOptions, Tool, ToolCall } from "../types/openai.js";
import { formatChat } from "../models/tokenizer.js";
import { GenerationExecutionError, memorySnapshot, nowMs, elapsedMs, estimateFullLogitsMb, estimateAttentionScoresMb, serializedToolsChars, logGenerationProfile } from "./profile.js";
import { parseToolCalls } from "../tools/parser.js";

// ── Tool-aware token streamer ───────────────────────────────────────────────
//
// Plain streaming emits every token straight to the client. That doesn't
// work when `tools` are present: a model might output
//
//     "Let me check the weather.<tool_call>{\"name\":...}</tool_call>"
//
// and the client expects the `<tool_call>{"name":...}` portion to arrive as
// a structured `delta.tool_calls[]` chunk, not as content.
//
// Strategy:
//   1. Accumulate every decoded token in `buffer`.
//   2. Until an opener is spotted, emit tokens as content deltas but hold
//      back the trailing `SAFETY_BUFFER` characters so a partial opener
//      (e.g. "<tool_" before "call>" arrives) never leaks.
//   3. Once an opener is seen, lock at that position, stop emitting content,
//      and run `parseToolCalls()` on each new token until it returns a full
//      match. That's when the structured tool_calls delta fires.
//   4. If generation ends without a parse hit, flush whatever was locked as
//      plain content — the opener was a false positive.
//
// Supported openers cover the five formats `parseToolCalls()` understands:
//   - Qwen:          `<tool_call>`
//   - Gemma raw:     `<|tool_call>`
//   - Gemma key-val: `call:`
//   - LFM JSON:      `[tool_calls`
//   - OpenAI JSON:   `"tool_calls"`
//   - LFM Pythonic:  `[word(` (regex — no unique keyword)

/** Longest fixed opener is `<|tool_call>` (12 chars); allow some slack. */
const SAFETY_BUFFER = 16;

/**
 * Each opener is the start marker for one tool-call format. Regexes are
 * used when the marker must include surrounding syntax (`{` before
 * `"tool_calls"` for OpenAI JSON, `[name(` for LFM Pythonic).
 */
const OPENERS: ReadonlyArray<string | RegExp> = [
  "<tool_call>",            // Qwen
  "<|tool_call>",           // Gemma raw
  "[tool_calls",            // LFM JSON
  "call:",                  // Gemma key-val
  /\{\s*"tool_calls"/,      // OpenAI JSON — include the enclosing `{`
  /\[[A-Za-z_]\w*\(/,       // LFM Pythonic — `[name(`
] as const;

/**
 * First index in `buf` where any supported tool-call opener appears.
 * Returns -1 when no opener is present.
 */
export function findOpenerIndex(buf: string): number {
  let best = -1;
  for (const op of OPENERS) {
    let idx = -1;
    if (typeof op === "string") {
      idx = buf.indexOf(op);
    } else {
      const m = op.exec(buf);
      if (m && m.index !== undefined) idx = m.index;
    }
    if (idx >= 0 && (best < 0 || idx < best)) best = idx;
  }
  return best;
}

export interface StreamToolHandlers {
  onContent: (delta: string) => void | Promise<void>;
  onToolCalls: (calls: ToolCall[]) => void | Promise<void>;
}

/**
 * Generate with tool-aware token buffering. Content tokens flow into
 * `onContent`; a complete tool-call parse fires `onToolCalls` exactly once.
 * Handlers are invoked fire-and-forget from the synchronous TextStreamer
 * callback — the caller's transport (e.g. Hono `streamSSE`) is expected
 * to preserve write ordering internally.
 */
export async function generateStreamWithTools(
  models: LoadedModels,
  modelId: string,
  messages: ChatMessage[],
  genOpts: GenerationOptions,
  tools: Tool[] | undefined,
  handlers: StreamToolHandlers,
): Promise<{ promptTokens: number; completionTokens: number }> {
  const started = nowMs();
  const memoryBefore = memorySnapshot();
  const prompt = formatChat(models.tokenizer!, messages, modelId, tools, models.chatTemplate);
  const inputs = models.tokenizer!(prompt, { return_tensors: "pt" });
  const memoryAfterTokenize = memorySnapshot();
  const promptTokens = inputs.input_ids.dims[1]!;
  let completionTokens = 0;
  if (models.maxContextLength && promptTokens >= models.maxContextLength) {
    const message = (
      `Prompt has ${promptTokens} tokens, but the model context is ${models.maxContextLength} tokens. ` +
      "Reduce the prompt/tools."
    );
    const profile = {
      path: "stream" as const,
      promptChars: prompt.length,
      toolsCount: tools?.length ?? 0,
      toolsChars: serializedToolsChars(tools),
      promptTokens,
      completionTokens: 0,
      formatMs: 0,
      tokenizeMs: 0,
      generateMs: 0,
      decodeMs: 0,
      totalMs: elapsedMs(started),
      memoryBefore,
      memoryAfterTokenize,
      memoryAfterGenerate: memoryAfterTokenize,
      memoryAfterDecode: memoryAfterTokenize,
      estimatedFullLogitsMb: estimateFullLogitsMb(models, promptTokens),
      estimatedAttentionScoresMb: estimateAttentionScoresMb(models, promptTokens),
      numLogitsToKeepInput: models.generationDiagnostics.numLogitsToKeepInput,
      numLogitsToKeepPatchedSessions: models.generationDiagnostics.numLogitsToKeepPatchedSessions,
      failedStage: "tokenize" as const,
      errorMessage: message,
    };
    logGenerationProfile(profile);
    throw new GenerationExecutionError(new Error(message), profile, 400);
  }
  const effectiveGenOpts = models.maxContextLength
    ? {
        ...genOpts,
        max_new_tokens: Math.min(
          genOpts.max_new_tokens,
          Math.max(1, models.maxContextLength - promptTokens),
        ),
      }
    : genOpts;

  let buffer = "";
  let emittedLen = 0;
  let lockPosition: number | null = null;
  let toolCalls: ToolCall[] | null = null;

  const streamer = new TextStreamer(
    models.tokenizer as unknown as ConstructorParameters<typeof TextStreamer>[0],
    {
      skip_prompt: true,
      callback_function: (token: string) => {
        if (toolCalls) return; // Already resolved — ignore late tokens.
        completionTokens++;
        buffer += token;

        // Stage 1 — not yet locked: scan for an opener.
        if (lockPosition === null) {
          const idx = findOpenerIndex(buffer);
          if (idx >= 0) {
            lockPosition = idx;
            // Flush any content that appeared before the opener.
            if (idx > emittedLen) {
              const pre = buffer.slice(emittedLen, idx);
              void handlers.onContent(pre);
              emittedLen = idx;
            }
          } else {
            // No opener yet — emit tokens up to `len - SAFETY_BUFFER`.
            const safeEnd = buffer.length - SAFETY_BUFFER;
            if (safeEnd > emittedLen) {
              const delta = buffer.slice(emittedLen, safeEnd);
              void handlers.onContent(delta);
              emittedLen = safeEnd;
            }
            return;
          }
        }

        // Stage 2 — locked: try to parse a complete tool call.
        const suspect = buffer.slice(lockPosition);
        const calls = parseToolCalls(suspect);
        if (calls && calls.length > 0) {
          toolCalls = calls;
        }
      },
    },
  );

  await models.model!.generate({ ...inputs, ...effectiveGenOpts, streamer });

  if (toolCalls) {
    await handlers.onToolCalls(toolCalls);
  } else {
    // Either no opener ever appeared (flush SAFETY_BUFFER tail), or we locked
    // on a false positive (e.g. the word "call:" inside prose) and generation
    // ended without a real tool call. The unemitted tail is just content.
    const tail = buffer.slice(emittedLen);
    if (tail.length > 0) await handlers.onContent(tail);
  }

  return { promptTokens, completionTokens };
}
