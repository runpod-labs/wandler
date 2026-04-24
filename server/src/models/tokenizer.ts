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
 * Recursively sanitize a JSON-Schema property descriptor: ensure it has a
 * `type` field (inferred from structure — "object" when it has `properties`,
 * "array" when it has `items`, otherwise "string") and recurse into every
 * nested schema location the Gemma template might walk:
 *   - `properties.<k>` (object schemas)
 *   - `items` (array item schemas, single or tuple form)
 *   - `anyOf` / `oneOf` / `allOf` (schema unions)
 *   - `additionalProperties` (when it's a schema, not a boolean)
 *
 * Gemma's chat_template.jinja does `value['type'] | upper` on every property
 * descriptor. JSON Schema allows a property without an explicit `type` (a
 * description alone is valid), and real-world tool schemas nest deeply with
 * descriptors that omit `type` — which crashes the template with
 * `Cannot apply filter "upper" to type: UndefinedValue`.
 */
function sanitizeProperty(prop: unknown): unknown {
  if (!prop || typeof prop !== "object" || Array.isArray(prop)) return prop;
  const p = prop as Record<string, unknown>;
  const out: Record<string, unknown> = { ...p };

  if (!("type" in out)) {
    if (out.properties) out.type = "object";
    else if (out.items) out.type = "array";
    else out.type = "string";
  }

  // Gemma's template has a fallback branch for type=="object" descriptors
  // that lack a `properties` key — it recursively treats every other key
  // in the descriptor (e.g. `patternProperties`, `additionalProperties`)
  // as a sub-property, which crashes `upper` on non-descriptor values.
  // Always supply `properties` so the primary branch handles it.
  if (out.type === "object" && !("properties" in out)) {
    out.properties = {};
  }

  // Recurse into nested object properties
  if (
    out.properties &&
    typeof out.properties === "object" &&
    !Array.isArray(out.properties)
  ) {
    const sanitized: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(out.properties as Record<string, unknown>)) {
      sanitized[k] = sanitizeProperty(v);
    }
    out.properties = sanitized;
  }

  // Recurse into array item schema(s) — may be a single schema or a tuple
  if (out.items) {
    if (Array.isArray(out.items)) {
      out.items = out.items.map(sanitizeProperty);
    } else if (typeof out.items === "object") {
      out.items = sanitizeProperty(out.items);
    }
  }

  // Recurse into schema unions
  for (const key of ["anyOf", "oneOf", "allOf"] as const) {
    if (Array.isArray(out[key])) {
      out[key] = (out[key] as unknown[]).map(sanitizeProperty);
    }
  }

  // Recurse into additionalProperties when it's a schema (object, not bool)
  if (
    out.additionalProperties &&
    typeof out.additionalProperties === "object" &&
    !Array.isArray(out.additionalProperties)
  ) {
    out.additionalProperties = sanitizeProperty(out.additionalProperties);
  }

  return out;
}

/**
 * Sanitize every tool's parameter schema so the Gemma chat template won't
 * crash on missing `type` fields. The top-level `parameters` object itself
 * is left untouched if it has no `properties` — empty / absent parameters
 * are valid for parameterless tools.
 */
function sanitizeTools(tools: Tool[]): Tool[] {
  return tools.map((tool) => {
    const params = tool.function.parameters as Record<string, unknown> | undefined;
    if (!params || typeof params !== "object") return tool;
    if (
      !params.properties ||
      typeof params.properties !== "object" ||
      Array.isArray(params.properties)
    ) {
      return tool;
    }

    const sanitizedProps: Record<string, unknown> = {};
    for (const [key, prop] of Object.entries(
      params.properties as Record<string, unknown>,
    )) {
      sanitizedProps[key] = sanitizeProperty(prop);
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
