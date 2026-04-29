import type { LoadedModels } from "../models/manager.js";
import type { ChatMessage, GenerationOptions, Tool } from "../types/openai.js";
import { generateCompletion, streamCompletionTokens } from "../generation/completion.js";
import { generate } from "../generation/generate.js";
import { generateStreamTokens } from "../generation/stream.js";
import { generateStreamWithTools } from "../generation/stream-tools.js";
import type { LLMBackend, StreamToolHandlers } from "./types.js";

export class WandlerBackend implements LLMBackend {
  readonly name = "wandler" as const;

  constructor(readonly models: LoadedModels) {}

  generateChat(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    tools?: Tool[],
  ) {
    return generate(this.models, modelId, messages, genOpts, tools);
  }

  streamChat(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    onToken: (token: string) => void | Promise<void>,
    tools?: Tool[],
  ) {
    return generateStreamTokens(this.models, modelId, messages, genOpts, onToken, tools);
  }

  streamChatWithTools(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    tools: Tool[] | undefined,
    handlers: StreamToolHandlers,
  ) {
    return generateStreamWithTools(this.models, modelId, messages, genOpts, tools, handlers);
  }

  generateCompletion(prompt: string, genOpts: GenerationOptions) {
    return generateCompletion(this.models, prompt, genOpts);
  }

  streamCompletion(
    prompt: string,
    genOpts: GenerationOptions,
    onToken: (token: string) => void | Promise<void>,
  ) {
    return streamCompletionTokens(this.models, prompt, genOpts, onToken);
  }
}
