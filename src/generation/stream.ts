import http from "node:http";
import { TextStreamer } from "@huggingface/transformers";
import type { LoadedModels } from "../models/manager.js";
import type {
  ChatCompletionChunk,
  ChatMessage,
  GenerationOptions,
  Tool,
} from "../types/openai.js";
import { formatChat } from "../models/tokenizer.js";

export async function generateStream(
  res: http.ServerResponse,
  models: LoadedModels,
  modelId: string,
  messages: ChatMessage[],
  genOpts: GenerationOptions,
  id: string,
  created: number,
  includeUsage: boolean,
  tools?: Tool[],
): Promise<void> {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "Access-Control-Allow-Origin": "*",
  });

  const sse = (data: ChatCompletionChunk): void => {
    res.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  // Initial role chunk
  sse({
    id,
    object: "chat.completion.chunk",
    created,
    model: modelId,
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  });

  const prompt = formatChat(models.tokenizer, messages, modelId, tools);
  const inputs = models.tokenizer(prompt, { return_tensors: "pt" });
  const promptTokens = inputs.input_ids.dims[1]!;
  let completionTokens = 0;

  const streamer = new TextStreamer(
    models.tokenizer as unknown as ConstructorParameters<typeof TextStreamer>[0],
    {
      skip_prompt: true,
      callback_function: (token: string) => {
        completionTokens++;
        sse({
          id,
          object: "chat.completion.chunk",
          created,
          model: modelId,
          choices: [{ index: 0, delta: { content: token }, finish_reason: null }],
        });
      },
    },
  );

  await models.model.generate({ ...inputs, ...genOpts, streamer });

  // Final chunk
  const finalChunk: ChatCompletionChunk = {
    id,
    object: "chat.completion.chunk",
    created,
    model: modelId,
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
  };
  if (includeUsage) {
    finalChunk.usage = {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    };
  }
  sse(finalChunk);

  res.write("data: [DONE]\n\n");
  res.end();
}
