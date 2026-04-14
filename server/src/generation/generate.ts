import type { LoadedModels } from "../models/manager.js";
import { loadImage } from "../models/manager.js";
import type {
  ChatMessage,
  GenerationOptions,
  GenerationResult,
  Tool,
} from "../types/openai.js";
import { formatChat } from "../models/tokenizer.js";
import { getImageUrls } from "../utils/content.js";

/**
 * Prepare inputs for vision models: process text + images through the processor.
 */
async function prepareVisionInputs(
  models: LoadedModels,
  messages: ChatMessage[],
  _modelId: string,
  tools?: Tool[],
) {
  const proc = models.processor as {
    apply_chat_template: (msgs: unknown[], opts: Record<string, unknown>) => string;
    (textOrImages: unknown, imagesOrOpts?: unknown): Promise<Record<string, unknown>>;
    batch_decode: (ids: unknown, opts: Record<string, unknown>) => string[];
  };

  // Convert OpenAI message format to transformers.js format
  // { type: "image_url", image_url: { url } } -> { type: "image" }
  const templateMessages = messages.map((m) => {
    if (typeof m.content === "string" || m.content == null) return m;
    return {
      ...m,
      content: m.content.map((part) =>
        part.type === "image_url" ? { type: "image" as const } : part,
      ),
    };
  });

  const opts: Record<string, unknown> = {
    tokenize: false,
    add_generation_prompt: true,
  };
  if (tools?.length) opts.tools = tools;
  if (models.chatTemplate) opts.chat_template = models.chatTemplate;

  const text = proc.apply_chat_template(templateMessages, opts);

  // Collect all images from messages
  const imageUrls: string[] = [];
  for (const m of messages) {
    imageUrls.push(...getImageUrls(m.content));
  }

  const images = await Promise.all(imageUrls.map(loadImage));

  // Process text + images through the processor
  // Most processors: processor(text, images), LLaVA: processor(images, text)
  const inputs = images.length > 0
    ? await proc(text, images.length === 1 ? images[0] : images)
    : await proc(text);

  return { inputs, batchDecode: proc.batch_decode.bind(proc) };
}

export async function generate(
  models: LoadedModels,
  modelId: string,
  messages: ChatMessage[],
  genOpts: GenerationOptions,
  tools?: Tool[],
): Promise<GenerationResult> {
  // Vision model path: use processor for text+image inputs
  if (models.isVision && models.processor) {
    const imageUrls = messages.flatMap((m) => getImageUrls(m.content));
    if (imageUrls.length > 0) {
      const { inputs, batchDecode } = await prepareVisionInputs(models, messages, modelId, tools);
      const inputIds = inputs.input_ids as { dims: number[] };
      const outputIds = await models.model!.generate({ ...inputs, ...genOpts });

      const promptTokens = inputIds.dims[1]!;
      const completionTokens = outputIds.dims[1]! - promptTokens;
      const newIds = outputIds.slice(null, [promptTokens, null]);
      const text = batchDecode(newIds, { skip_special_tokens: true })[0]!;
      return { text, promptTokens, completionTokens };
    }
  }

  // Text-only path (default)
  const prompt = formatChat(models.tokenizer!, messages, modelId, tools, models.chatTemplate);
  const inputs = models.tokenizer!(prompt, { return_tensors: "pt" });
  const outputIds = await models.model!.generate({ ...inputs, ...genOpts });

  const promptTokens = inputs.input_ids.dims[1]!;
  const completionTokens = outputIds.dims[1]! - promptTokens;
  const newIds = outputIds.slice(null, [promptTokens, null]);
  const text = models.tokenizer!.batch_decode(newIds, { skip_special_tokens: true })[0]!;
  return { text, promptTokens, completionTokens };
}
