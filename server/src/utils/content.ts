import type { ChatMessage, ContentPart } from "../types/openai.js";

/**
 * Extract the text content from a message, whether it's a plain string
 * or an array of content parts.
 */
export function getTextContent(content: ChatMessage["content"]): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  return content
    .filter((p): p is Extract<ContentPart, { type: "text" }> => p.type === "text")
    .map((p) => p.text)
    .join("\n");
}

/**
 * Extract image URLs from a multimodal content array.
 * Returns empty array for plain string content.
 */
export function getImageUrls(content: ChatMessage["content"]): string[] {
  if (content == null || typeof content === "string") return [];
  return content
    .filter((p): p is Extract<ContentPart, { type: "image_url" }> => p.type === "image_url")
    .map((p) => p.image_url.url);
}

/**
 * Check if any message in the conversation contains images.
 */
export function hasImages(messages: ChatMessage[]): boolean {
  return messages.some((m) => getImageUrls(m.content).length > 0);
}

/**
 * Convert messages to text-only format for non-vision models.
 * Strips image parts, keeps only text.
 */
export function toTextMessages(messages: ChatMessage[]): ChatMessage[] {
  return messages.map((m) => ({
    ...m,
    content: getTextContent(m.content),
  }));
}
