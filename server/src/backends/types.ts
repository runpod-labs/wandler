import type { LoadedModels } from "../models/manager.js";
import type { ChatMessage, GenerationOptions, GenerationProfile, GenerationResult, Tool, ToolCall } from "../types/openai.js";

export interface StreamToolHandlers {
  onContent: (delta: string) => void | Promise<void>;
  onToolCalls: (calls: ToolCall[]) => void | Promise<void>;
}

export interface LLMBackend {
  readonly name: "wandler" | "transformersjs";
  readonly models: LoadedModels;
  generateChat(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    tools?: Tool[],
  ): Promise<GenerationResult>;
  streamChat(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    onToken: (token: string) => void | Promise<void>,
    tools?: Tool[],
  ): Promise<{ promptTokens: number; completionTokens: number; profile?: GenerationProfile }>;
  streamChatWithTools(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    tools: Tool[] | undefined,
    handlers: StreamToolHandlers,
  ): Promise<{ promptTokens: number; completionTokens: number; profile?: GenerationProfile }>;
  generateCompletion(
    prompt: string,
    genOpts: GenerationOptions,
  ): Promise<GenerationResult>;
  streamCompletion(
    prompt: string,
    genOpts: GenerationOptions,
    onToken: (token: string) => void | Promise<void>,
  ): Promise<{ promptTokens: number; completionTokens: number; profile?: GenerationProfile }>;
}
