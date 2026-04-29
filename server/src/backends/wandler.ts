import type { LoadedModels } from "../models/manager.js";
import type { ChatMessage, GenerationOptions, Tool } from "../types/openai.js";
import { generate } from "../generation/generate.js";
import { getImageUrls } from "../utils/content.js";
import type { LLMBackend, StreamToolHandlers } from "./types.js";
import { WandlerTextEngine } from "./wandler/engine.js";

export class WandlerBackend implements LLMBackend {
  readonly name = "wandler" as const;
  private readonly textEngine: WandlerTextEngine;

  constructor(readonly models: LoadedModels) {
    this.textEngine = new WandlerTextEngine(models);
  }

  generateChat(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    tools?: Tool[],
  ) {
    if (this.hasVisionImages(messages)) {
      return generate(this.models, modelId, messages, genOpts, tools);
    }
    return this.textEngine.generateChat(modelId, messages, genOpts, tools);
  }

  streamChat(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    onToken: (token: string) => void | Promise<void>,
    tools?: Tool[],
  ) {
    return this.textEngine.streamChat(modelId, messages, genOpts, onToken, tools);
  }

  streamChatWithTools(
    modelId: string,
    messages: ChatMessage[],
    genOpts: GenerationOptions,
    tools: Tool[] | undefined,
    handlers: StreamToolHandlers,
  ) {
    return this.textEngine.streamChatWithTools(modelId, messages, genOpts, tools, handlers);
  }

  generateCompletion(prompt: string, genOpts: GenerationOptions) {
    return this.textEngine.generateCompletion(prompt, genOpts);
  }

  streamCompletion(
    prompt: string,
    genOpts: GenerationOptions,
    onToken: (token: string) => void | Promise<void>,
  ) {
    return this.textEngine.streamCompletion(prompt, genOpts, onToken);
  }

  private hasVisionImages(messages: ChatMessage[]): boolean {
    return Boolean(
      this.models.isVision &&
      this.models.processor &&
      messages.some((message) => getImageUrls(message.content).length > 0),
    );
  }
}
