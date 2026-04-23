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
 * Recursively sanitize a schema property: ensure every property descriptor
 * (at any depth) has a `type` field (defaulting to "string" if missing).
 * Handles:
 * - Nested object properties
 * - Array items
 * - anyOf / oneOf / allOf unions
 * - additionalProperties when it's an object
 *
 * Gemma's chat_template.jinja does `value['type'] | upper` on every property
 * in a tool's parameter schema. JSON Schema allows a property without an
 * explicit `type` (a description alone is valid), and OpenAI-style clients
 * sometimes emit such schemas — which crashes the template with
 * `Cannot apply filter "upper" to type: UndefinedValue`.
 */
function sanitizeSchema(
  schema: Record<string, unknown>,
  isPropertyDescriptor = false,
): Record<string, unknown> {
  if (!schema || typeof schema !== "object") return schema;

  const sanitized = { ...schema };

  // Only add type to property descriptors that lack schema-defining keywords
  // (like type, properties, items, anyOf, etc). Empty objects stay empty.
  const hasSchemaKeyword = [
    "type",
    "properties",
    "items",
    "anyOf",
    "oneOf",
    "allOf",
    "const",
    "enum",
    "$ref",
    "additionalProperties",
  ].some((key) => key in sanitized);

  if (isPropertyDescriptor && !hasSchemaKeyword) {
    sanitized.type = "string";
  }

  // Recursively sanitize nested object properties
  if (sanitized.properties && typeof sanitized.properties === "object") {
    const props = sanitized.properties as Record<string, unknown>;
    sanitized.properties = Object.fromEntries(
      Object.entries(props).map(([key, prop]) => [
        key,
        typeof prop === "object" && prop !== null
          ? sanitizeSchema(prop as Record<string, unknown>, true)
          : prop,
      ]),
    );
  }

  // Recursively sanitize array items
  if (sanitized.items && typeof sanitized.items === "object") {
    sanitized.items = sanitizeSchema(sanitized.items as Record<string, unknown>, false);
  }

  // Recursively sanitize union types
  for (const unionKey of ["anyOf", "oneOf", "allOf"]) {
    if (sanitized[unionKey] && Array.isArray(sanitized[unionKey])) {
      sanitized[unionKey] = (sanitized[unionKey] as unknown[]).map((schema) =>
        typeof schema === "object" && schema !== null
          ? sanitizeSchema(schema as Record<string, unknown>, false)
          : schema,
      );
    }
  }

  // Recursively sanitize additionalProperties when it's a schema
  if (sanitized.additionalProperties && typeof sanitized.additionalProperties === "object") {
    sanitized.additionalProperties = sanitizeSchema(
      sanitized.additionalProperties as Record<string, unknown>,
      false,
    );
  }

  return sanitized;
}

/**
 * Gemma's chat_template.jinja does `value['type'] | upper` on every property
 * in a tool's parameter schema. JSON Schema allows a property without an
 * explicit `type` (a description alone is valid), and OpenAI-style clients
 * sometimes emit such schemas — which crashes the template with
 * `Cannot apply filter "upper" to type: UndefinedValue`.
 *
 * Recursively default every property to `type: "string"` when missing.
 */
function sanitizeTools(tools: Tool[]): Tool[] {
  return tools.map((tool) => {
    const params = tool.function.parameters as
      | Record<string, unknown>
      | undefined;
    if (!params) return tool;

    const sanitizedParams = sanitizeSchema(params);

    return {
      ...tool,
      function: {
        ...tool.function,
        parameters: sanitizedParams,
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
