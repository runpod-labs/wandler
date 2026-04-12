import type { ChatMessage, Tool } from "../types/openai.js";

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

// ── Gemma chat template (not included in ONNX tokenizer config) ─────────────

export function formatGemmaChat(messages: ChatMessage[]): string {
  let prompt = "";
  let inUserTurn = false; // track if system message started a user turn
  for (const msg of messages) {
    if (msg.role === "system") {
      // Gemma has no system role — start a user turn and prepend system content
      prompt += `<start_of_turn>user\n${msg.content}\n`;
      inUserTurn = true;
    } else if (msg.role === "user") {
      if (inUserTurn) {
        // Continue the user turn started by a system message
        prompt += msg.content + "<end_of_turn>\n";
        inUserTurn = false;
      } else {
        prompt += `<start_of_turn>user\n${msg.content}<end_of_turn>\n`;
      }
    } else if (msg.role === "assistant") {
      if (inUserTurn) {
        // Close unclosed user turn before assistant
        prompt += "<end_of_turn>\n";
        inUserTurn = false;
      }
      prompt += `<start_of_turn>model\n${msg.content}<end_of_turn>\n`;
    }
  }
  prompt += "<start_of_turn>model\n";
  return prompt;
}

// ── Generic chat template with fallback ─────────────────────────────────────

export function formatChat(
  tokenizer: Tokenizer,
  messages: ChatMessage[],
  modelId: string,
  tools?: Tool[],
): string {
  try {
    const opts: Record<string, unknown> = {
      tokenize: false,
      add_generation_prompt: true,
    };
    if (tools?.length) {
      opts.tools = tools;
    }
    return tokenizer.apply_chat_template(messages, opts);
  } catch {
    // Fallback for models without chat template (e.g. Gemma ONNX exports)
    if (modelId.toLowerCase().includes("gemma")) {
      return formatGemmaChat(messages);
    }
    // Generic fallback
    return (
      messages
        .map((m) =>
          m.role === "assistant"
            ? `Assistant: ${m.content}`
            : `User: ${m.content}`,
        )
        .join("\n") + "\nAssistant: "
    );
  }
}
