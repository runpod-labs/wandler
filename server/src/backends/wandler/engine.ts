import { TextStreamer } from "@huggingface/transformers";
import type { LoadedModels } from "../../models/manager.js";
import { formatChat } from "../../models/tokenizer.js";
import { parseToolCalls } from "../../tools/parser.js";
import type {
  ChatMessage,
  GenerationOptions,
  GenerationProfile,
  GenerationResult,
  Tool,
  ToolCall,
} from "../../types/openai.js";
import { stripInternalGenOpts } from "../../generation/options.js";
import {
  buildPrefixCandidate,
  preparePrefill,
  type PrefillResult,
  type TensorLike,
} from "../../generation/prefill.js";
import {
  elapsedMs,
  estimateAttentionScoresMb,
  estimateFullLogitsMb,
  GenerationExecutionError,
  logGenerationProfile,
  memorySnapshot,
  nowMs,
  serializedToolsChars,
} from "../../generation/profile.js";

type GenerationOutput = {
  dims: number[];
  slice(...args: unknown[]): unknown;
};

type TokenizedInputs = Record<string, unknown> & {
  input_ids: TensorLike;
};

type StreamToolHandlers = {
  onContent: (delta: string) => void | Promise<void>;
  onToolCalls: (calls: ToolCall[]) => void | Promise<void>;
};

type TextPromptRequest = {
  prompt: string;
  genOpts: GenerationOptions;
  tools?: Tool[];
  modelId?: string;
  messages?: ChatMessage[];
};

type PromptTiming = {
  started: number;
  memoryBefore: ReturnType<typeof memorySnapshot>;
  formatMs: number;
};

type PreparedTextPrompt = {
  started: number;
  memoryBefore: ReturnType<typeof memorySnapshot>;
  prompt: string;
  inputs: TokenizedInputs;
  promptTokens: number;
  formatMs: number;
  tokenizeMs: number;
  memoryAfterTokenize: ReturnType<typeof memorySnapshot>;
  toolsCount: number;
  toolsChars: number;
  effectiveGenOpts: GenerationOptions;
  prefixCandidate: ReturnType<typeof buildPrefixCandidate>;
};

// Keep enough tail content buffered so partial tool-call openers never leak.
const TOOL_SAFETY_BUFFER = 16;
const TOOL_OPENERS: ReadonlyArray<string | RegExp> = [
  "<tool_call>",
  "<|tool_call>",
  "[tool_calls",
  "call:",
  /\{\s*"tool_calls"/,
  /\[[A-Za-z_]\w*\(/,
] as const;

function findToolOpenerIndex(buffer: string): number {
  let best = -1;
  for (const opener of TOOL_OPENERS) {
    const idx = typeof opener === "string"
      ? buffer.indexOf(opener)
      : opener.exec(buffer)?.index ?? -1;
    if (idx >= 0 && (best < 0 || idx < best)) best = idx;
  }
  return best;
}

function promptTooLongErrorMessage(promptTokens: number, maxContextLength: number, suffix: string): string | null {
  if (promptTokens < maxContextLength) return null;
  return (
    `Prompt has ${promptTokens} tokens, but the model context is ${maxContextLength} tokens. ` +
    suffix
  );
}

function capGenOpts(models: LoadedModels, promptTokens: number, genOpts: GenerationOptions): GenerationOptions {
  return models.maxContextLength
    ? {
        ...genOpts,
        max_new_tokens: Math.min(
          genOpts.max_new_tokens,
          Math.max(1, models.maxContextLength - promptTokens),
        ),
      }
    : genOpts;
}

function completionPromptOffset(prefill: PrefillResult, promptTokens: number): number {
  return prefill.prefillChunkSize ? 1 : promptTokens;
}

export class WandlerTextEngine {
  constructor(private readonly models: LoadedModels) {}

  async generateChat(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    tools?: Tool[],
  ): Promise<GenerationResult> {
    const started = nowMs();
    const memoryBefore = memorySnapshot();
    const formatStart = nowMs();
    const prompt = formatChat(this.requireTokenizer(), messages, modelId, tools, this.models.chatTemplate);
    const formatMs = elapsedMs(formatStart);
    return this.generateText({
      prompt,
      genOpts,
      tools,
      modelId,
      messages,
    }, { started, memoryBefore, formatMs }, "Reduce the prompt/tools.");
  }

  async streamChat(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    onToken: (token: string) => void | Promise<void>,
    tools?: Tool[],
  ): Promise<{ promptTokens: number; completionTokens: number; profile: GenerationProfile }> {
    const started = nowMs();
    const memoryBefore = memorySnapshot();
    const formatStart = nowMs();
    const prompt = formatChat(this.requireTokenizer(), messages, modelId, tools, this.models.chatTemplate);
    const formatMs = elapsedMs(formatStart);
    return this.streamText({
      prompt,
      genOpts,
      tools,
      modelId,
      messages,
    }, { started, memoryBefore, formatMs }, onToken, "Reduce the prompt/tools.");
  }

  async streamChatWithTools(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    tools: Tool[] | undefined,
    handlers: StreamToolHandlers,
  ): Promise<{ promptTokens: number; completionTokens: number; profile: GenerationProfile }> {
    const started = nowMs();
    const memoryBefore = memorySnapshot();
    const formatStart = nowMs();
    const prompt = formatChat(this.requireTokenizer(), messages, modelId, tools, this.models.chatTemplate);
    const formatMs = elapsedMs(formatStart);
    const prepared = this.prepareTextPrompt({
      prompt,
      genOpts,
      tools,
      modelId,
      messages,
    }, { started, memoryBefore, formatMs }, "stream", "Reduce the prompt/tools.");

    let buffer = "";
    let emittedLen = 0;
    let lockPosition: number | null = null;
    let toolCalls: ToolCall[] | null = null;
    let completionTokens = 0;

    const streamer = new TextStreamer(
      this.requireTokenizer() as unknown as ConstructorParameters<typeof TextStreamer>[0],
      {
        skip_prompt: true,
        callback_function: (token: string) => {
          if (toolCalls) return;
          completionTokens++;
          buffer += token;

          if (lockPosition === null) {
            const idx = findToolOpenerIndex(buffer);
            if (idx >= 0) {
              lockPosition = idx;
              if (idx > emittedLen) {
                const pre = buffer.slice(emittedLen, idx);
                void handlers.onContent(pre);
                emittedLen = idx;
              }
            } else {
              const safeEnd = buffer.length - TOOL_SAFETY_BUFFER;
              if (safeEnd > emittedLen) {
                const delta = buffer.slice(emittedLen, safeEnd);
                void handlers.onContent(delta);
                emittedLen = safeEnd;
              }
              return;
            }
          }

          const suspect = buffer.slice(lockPosition);
          const calls = parseToolCalls(suspect);
          if (calls && calls.length > 0) {
            toolCalls = calls;
          }
        },
      },
    );

    const { profile } = await this.runGenerate(prepared, "stream", streamer, () => completionTokens);
    if (toolCalls) {
      await handlers.onToolCalls(toolCalls);
    } else {
      const tail = buffer.slice(emittedLen);
      if (tail.length > 0) await handlers.onContent(tail);
    }

    return { promptTokens: prepared.promptTokens, completionTokens, profile };
  }

  async generateCompletion(prompt: string, genOpts: GenerationOptions): Promise<GenerationResult> {
    return this.generateText(
      { prompt, genOpts },
      { started: nowMs(), memoryBefore: memorySnapshot(), formatMs: 0 },
      "Reduce the prompt.",
    );
  }

  async streamCompletion(
    prompt: string,
    genOpts: GenerationOptions,
    onToken: (token: string) => void | Promise<void>,
  ): Promise<{ promptTokens: number; completionTokens: number; profile: GenerationProfile }> {
    return this.streamText(
      { prompt, genOpts },
      { started: nowMs(), memoryBefore: memorySnapshot(), formatMs: 0 },
      onToken,
      "Reduce the prompt.",
    );
  }

  private async generateText(
    request: TextPromptRequest,
    timing: PromptTiming,
    contextErrorSuffix: string,
  ): Promise<GenerationResult> {
    const prepared = this.prepareTextPrompt(request, timing, "text", contextErrorSuffix);
    const { outputIds, profile } = await this.runGenerate(prepared, "text");
    const decodeStart = nowMs();
    const promptOffset = profile.prefillChunkSize ? 1 : prepared.promptTokens;
    const newIds = outputIds!.slice(null, [promptOffset, null]);
    const text = this.requireTokenizer().batch_decode(newIds, { skip_special_tokens: true })[0]!;
    profile.decodeMs = elapsedMs(decodeStart);
    profile.memoryAfterDecode = memorySnapshot();
    profile.totalMs = elapsedMs(prepared.started);
    logGenerationProfile(profile);
    return {
      text,
      promptTokens: prepared.promptTokens,
      completionTokens: profile.completionTokens,
      profile,
    };
  }

  private async streamText(
    request: TextPromptRequest,
    timing: PromptTiming,
    onToken: (token: string) => void | Promise<void>,
    contextErrorSuffix: string,
  ): Promise<{ promptTokens: number; completionTokens: number; profile: GenerationProfile }> {
    const prepared = this.prepareTextPrompt(request, timing, "stream", contextErrorSuffix);
    let completionTokens = 0;
    const streamer = new TextStreamer(
      this.requireTokenizer() as unknown as ConstructorParameters<typeof TextStreamer>[0],
      {
        skip_prompt: true,
        callback_function: (token: string) => {
          completionTokens++;
          onToken(token);
        },
      },
    );
    const { profile } = await this.runGenerate(prepared, "stream", streamer, () => completionTokens);
    return { promptTokens: prepared.promptTokens, completionTokens, profile };
  }

  private prepareTextPrompt(
    request: TextPromptRequest,
    timing: PromptTiming,
    path: "text" | "stream",
    contextErrorSuffix: string,
  ): PreparedTextPrompt {
    const tokenizeStart = nowMs();
    const inputs = this.requireTokenizer()(request.prompt, { return_tensors: "pt" }) as TokenizedInputs;
    const tokenizeMs = elapsedMs(tokenizeStart);
    const memoryAfterTokenize = memorySnapshot();
    const promptTokens = inputs.input_ids.dims[1] ?? 0;
    const toolsCount = request.tools?.length ?? 0;
    const toolsChars = serializedToolsChars(request.tools);

    const contextError = this.models.maxContextLength
      ? promptTooLongErrorMessage(promptTokens, this.models.maxContextLength, contextErrorSuffix)
      : null;
    if (contextError) {
      const profile = this.profile({
        path,
        started: timing.started,
        prompt: request.prompt,
        toolsCount,
        toolsChars,
        promptTokens,
        completionTokens: 0,
        formatMs: timing.formatMs,
        tokenizeMs,
        generateMs: 0,
        decodeMs: 0,
        memoryBefore: timing.memoryBefore,
        memoryAfterTokenize,
        memoryAfterGenerate: memoryAfterTokenize,
        memoryAfterDecode: memoryAfterTokenize,
        failedStage: "tokenize",
        errorMessage: contextError,
      });
      logGenerationProfile(profile);
      throw new GenerationExecutionError(new Error(contextError), profile, 400);
    }

    return {
      started: timing.started,
      memoryBefore: timing.memoryBefore,
      prompt: request.prompt,
      inputs,
      promptTokens,
      formatMs: timing.formatMs,
      tokenizeMs,
      memoryAfterTokenize,
      toolsCount,
      toolsChars,
      effectiveGenOpts: capGenOpts(this.models, promptTokens, request.genOpts),
      prefixCandidate: request.messages && request.modelId
        ? buildPrefixCandidate(
            this.requireTokenizer(),
            request.messages,
            request.modelId,
            request.tools,
            this.models.chatTemplate,
            request.prompt,
          )
        : null,
    };
  }

  private async runGenerate(
    prepared: PreparedTextPrompt,
    path: "text" | "stream",
    streamer?: TextStreamer,
    streamedCompletionTokens?: () => number,
  ): Promise<{ outputIds?: GenerationOutput; profile: GenerationProfile }> {
    const transformersGenOpts = stripInternalGenOpts(prepared.effectiveGenOpts);
    const generateStart = nowMs();
    let prefill: PrefillResult | null = null;
    try {
      prefill = await preparePrefill(
        this.models,
        prepared.inputs.input_ids,
        prepared.promptTokens,
        prepared.effectiveGenOpts,
        prepared.prompt,
        prepared.prefixCandidate,
      );

      const generateArgs = prefill.pastKeyValues
        ? {
            input_ids: prefill.inputIds,
            past_key_values: prefill.pastKeyValues,
            ...transformersGenOpts,
            ...(streamer ? { streamer } : {}),
          }
        : {
            ...prepared.inputs,
            ...transformersGenOpts,
            ...(streamer ? { streamer } : {}),
          };

      const outputIds = await this.requireModel().generate(generateArgs);
      const generateMs = elapsedMs(generateStart);
      const memoryAfterGenerate = memorySnapshot();
      const promptOffset = completionPromptOffset(prefill, prepared.promptTokens);
      const completionTokens = streamedCompletionTokens
        ? streamedCompletionTokens()
        : Math.max(0, (outputIds.dims[1] ?? promptOffset) - promptOffset);
      await prefill.cleanup();

      const profile = this.profile({
        path,
        started: prepared.started,
        prompt: prepared.prompt,
        toolsCount: prepared.toolsCount,
        toolsChars: prepared.toolsChars,
        promptTokens: prepared.promptTokens,
        completionTokens,
        formatMs: prepared.formatMs,
        tokenizeMs: prepared.tokenizeMs,
        generateMs,
        decodeMs: 0,
        memoryBefore: prepared.memoryBefore,
        memoryAfterTokenize: prepared.memoryAfterTokenize,
        memoryAfterGenerate,
        memoryAfterDecode: memoryAfterGenerate,
        prefill,
      });
      if (path === "stream") logGenerationProfile(profile);
      return { outputIds, profile };
    } catch (error) {
      await prefill?.cleanup();
      if (error instanceof GenerationExecutionError) throw error;
      const memoryAfterGenerate = memorySnapshot();
      const profile = this.profile({
        path,
        started: prepared.started,
        prompt: prepared.prompt,
        toolsCount: prepared.toolsCount,
        toolsChars: prepared.toolsChars,
        promptTokens: prepared.promptTokens,
        completionTokens: streamedCompletionTokens?.() ?? 0,
        formatMs: prepared.formatMs,
        tokenizeMs: prepared.tokenizeMs,
        generateMs: elapsedMs(generateStart),
        decodeMs: 0,
        memoryBefore: prepared.memoryBefore,
        memoryAfterTokenize: prepared.memoryAfterTokenize,
        memoryAfterGenerate,
        memoryAfterDecode: memoryAfterGenerate,
        prefill,
        failedStage: "generate",
        errorMessage: error instanceof Error ? error.message : String(error),
      });
      logGenerationProfile(profile);
      throw new GenerationExecutionError(error, profile);
    }
  }

  private profile(args: {
    path: "text" | "stream";
    started: number;
    prompt: string;
    toolsCount: number;
    toolsChars: number;
    promptTokens: number;
    completionTokens: number;
    formatMs: number;
    tokenizeMs: number;
    generateMs: number;
    decodeMs: number;
    memoryBefore: ReturnType<typeof memorySnapshot>;
    memoryAfterTokenize: ReturnType<typeof memorySnapshot>;
    memoryAfterGenerate: ReturnType<typeof memorySnapshot>;
    memoryAfterDecode: ReturnType<typeof memorySnapshot>;
    prefill?: PrefillResult | null;
    failedStage?: GenerationProfile["failedStage"];
    errorMessage?: string;
  }): GenerationProfile {
    return {
      path: args.path,
      promptChars: args.prompt.length,
      toolsCount: args.toolsCount,
      toolsChars: args.toolsChars,
      promptTokens: args.promptTokens,
      completionTokens: args.completionTokens,
      formatMs: args.formatMs,
      tokenizeMs: args.tokenizeMs,
      generateMs: args.generateMs,
      decodeMs: args.decodeMs,
      totalMs: elapsedMs(args.started),
      prefillChunkSize: args.prefill?.prefillChunkSize,
      prefillChunks: args.prefill?.prefillChunks,
      prefillMs: args.prefill?.prefillMs,
      prefixCacheHit: args.prefill?.prefixCacheHit,
      prefixCacheTokens: args.prefill?.prefixCacheTokens,
      memoryBefore: args.memoryBefore,
      memoryAfterTokenize: args.memoryAfterTokenize,
      memoryAfterGenerate: args.memoryAfterGenerate,
      memoryAfterDecode: args.memoryAfterDecode,
      estimatedFullLogitsMb: estimateFullLogitsMb(this.models, args.promptTokens),
      estimatedAttentionScoresMb: estimateAttentionScoresMb(this.models, args.promptTokens),
      numLogitsToKeepInput: this.models.generationDiagnostics?.numLogitsToKeepInput ?? false,
      numLogitsToKeepPatchedSessions: this.models.generationDiagnostics?.numLogitsToKeepPatchedSessions ?? [],
      failedStage: args.failedStage,
      errorMessage: args.errorMessage,
    };
  }

  private requireTokenizer(): NonNullable<LoadedModels["tokenizer"]> {
    if (!this.models.tokenizer) {
      throw new Error("LLM tokenizer is not loaded");
    }
    return this.models.tokenizer;
  }

  private requireModel(): NonNullable<LoadedModels["model"]> {
    if (!this.models.model) {
      throw new Error("LLM model is not loaded");
    }
    return this.models.model;
  }
}
