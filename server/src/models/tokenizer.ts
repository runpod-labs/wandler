import type { ChatMessage, Tool } from "../types/openai.js";
import { getTextContent } from "../utils/content.js";

// ── Tokenizer type (subset of transformers.js PreTrainedTokenizer) ──────────
// We define a minimal interface so this module can be unit-tested without
// loading a real model.

export interface Tokenizer {
  apply_chat_template(
    messages: ChatMessage[],
    options: Record<string, unknown>,
  ): string;
  (text: string, options: Record<string, unknown>): { input_ids: { dims: number[] } };
  batch_decode(ids: unknown, options: Record<string, unknown>): string[];
}

/**
 * Normalize messages so content is always a string (for tokenizers that
 * don't understand multimodal content arrays).
 */
function normalizeMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((m) => ({
    ...m,
    content: getTextContent(m.content),
  }));
}

// ── Generic chat template with external template support ────────────────────

export function formatChat(
  tokenizer: Tokenizer,
  messages: ChatMessage[],
  _modelId: string,
  tools?: Tool[],
  chatTemplate?: string | null,
): string {
  // Normalize multimodal content to strings for tokenization
  const normalized = normalizeMessages(messages);

  const opts: Record<string, unknown> = {
    tokenize: false,
    add_generation_prompt: true,
  };
  if (tools?.length) {
    opts.tools = tools;
  }
  // Pass external chat template (e.g. from chat_template.jinja) if the
  // tokenizer doesn't have one built-in. This handles Gemma 4 and any
  // other model that ships the template as a separate file.
  if (chatTemplate) {
    opts.chat_template = chatTemplate;
  }

  return tokenizer.apply_chat_template(normalized, opts);
}
