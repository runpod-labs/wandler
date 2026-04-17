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

/**
 * Gemma's chat_template.jinja does `value['type'] | upper` on every property
 * in a tool's parameter schema. JSON Schema allows a property without an
 * explicit `type` (a description alone is valid), and OpenAI-style clients
 * sometimes emit such schemas — which crashes the template with
 * `Cannot apply filter "upper" to type: UndefinedValue`.
 *
 * Defensively default every property to `type: "string"` when missing.
 */
function sanitizeTools(tools: Tool[]): Tool[] {
  return tools.map((tool) => {
    const params = tool.function.parameters as
      | { properties?: Record<string, Record<string, unknown>> }
      | undefined;
    if (!params?.properties) return tool;

    const sanitizedProps: Record<string, Record<string, unknown>> = {};
    for (const [key, prop] of Object.entries(params.properties)) {
      sanitizedProps[key] = prop && typeof prop === "object" && "type" in prop
        ? prop
        : { ...prop, type: "string" };
    }

    return {
      ...tool,
      function: {
        ...tool.function,
        parameters: { ...params, properties: sanitizedProps },
      },
    };
  });
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
    opts.tools = sanitizeTools(tools);
  }
  // Pass external chat template (e.g. from chat_template.jinja) if the
  // tokenizer doesn't have one built-in. This handles Gemma 4 and any
  // other model that ships the template as a separate file.
  if (chatTemplate) {
    opts.chat_template = chatTemplate;
  }

  return tokenizer.apply_chat_template(normalized, opts);
}
