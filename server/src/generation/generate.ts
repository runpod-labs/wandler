import type { LoadedModels } from "../models/manager.js";
import type {
  ChatMessage,
  GenerationOptions,
  GenerationResult,
  Tool,
} from "../types/openai.js";
import { formatChat } from "../models/tokenizer.js";

export async function generate(
  models: LoadedModels,
  modelId: string,
  messages: ChatMessage[],
  genOpts: GenerationOptions,
  tools?: Tool[],
): Promise<GenerationResult> {
  const prompt = formatChat(models.tokenizer, messages, modelId, tools, models.chatTemplate);
  const inputs = models.tokenizer(prompt, { return_tensors: "pt" });
  const outputIds = await models.model.generate({ ...inputs, ...genOpts });

  const promptTokens = inputs.input_ids.dims[1]!;
  const completionTokens = outputIds.dims[1]! - promptTokens;
  const newIds = outputIds.slice(null, [promptTokens, null]);
  const text = models.tokenizer.batch_decode(newIds, { skip_special_tokens: true })[0]!;
  return { text, promptTokens, completionTokens };
}
