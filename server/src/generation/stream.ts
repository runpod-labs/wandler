import { TextStreamer } from "@huggingface/transformers";
import type { LoadedModels } from "../models/manager.js";
import type { ChatMessage, GenerationOptions, Tool } from "../types/openai.js";
import { formatChat } from "../models/tokenizer.js";

/**
 * Generate tokens with a callback for each token.
 * Framework-agnostic — works with Hono streamSSE, raw http, or anything else.
 */
export async function generateStreamTokens(
  models: LoadedModels,
  modelId: string,
  messages: ChatMessage[],
  genOpts: GenerationOptions,
  onToken: (token: string) => void | Promise<void>,
  tools?: Tool[],
): Promise<{ promptTokens: number; completionTokens: number }> {
  const prompt = formatChat(models.tokenizer, messages, modelId, tools, models.chatTemplate);
  const inputs = models.tokenizer(prompt, { return_tensors: "pt" });
  const promptTokens = inputs.input_ids.dims[1]!;
  let completionTokens = 0;

  const streamer = new TextStreamer(
    models.tokenizer as unknown as ConstructorParameters<typeof TextStreamer>[0],
    {
      skip_prompt: true,
      callback_function: (token: string) => {
        completionTokens++;
        onToken(token);
      },
    },
  );

  await models.model.generate({ ...inputs, ...genOpts, streamer });

  return { promptTokens, completionTokens };
}
