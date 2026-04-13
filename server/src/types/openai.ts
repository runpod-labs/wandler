// ── OpenAI-compatible type definitions ──────────────────────────────────────

export interface ContentPartText {
  type: "text";
  text: string;
}

export interface ContentPartImageUrl {
  type: "image_url";
  image_url: { url: string; detail?: "auto" | "low" | "high" };
}

export type ContentPart = ContentPartText | ContentPartImageUrl;

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | ContentPart[] | null;
  name?: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface ToolFunction {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
}

export interface Tool {
  type: "function";
  function: ToolFunction;
}

export interface ToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

// ── Shared sampling parameters (used by both chat and completions) ──────────

export interface SamplingParams {
  temperature?: number;
  top_p?: number;
  top_k?: number;
  min_p?: number;
  typical_p?: number;
  max_tokens?: number;
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  repetition_penalty?: number;
  logit_bias?: Record<string, number>;
  seed?: number;
  n?: number;
  no_repeat_ngram_size?: number;
  response_format?: {
    type: "text" | "json_object" | "json_schema";
    json_schema?: { name: string; strict?: boolean; schema: Record<string, unknown> };
  };
  user?: string;
}

// ── Chat Completions ────────────────────────────────────────────────────────

export interface ChatCompletionRequest extends SamplingParams {
  model?: string;
  messages: ChatMessage[];
  tools?: Tool[];
  tool_choice?: "none" | "auto" | "required" | { type: "function"; function: { name: string } };
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionChoice {
  index: number;
  message: ChatMessage;
  finish_reason: "stop" | "length" | "tool_calls";
}

export interface ChatCompletionResponse {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: TokenUsage;
}

export interface ChatCompletionChunkDelta {
  role?: "assistant";
  content?: string;
  tool_calls?: ToolCall[];
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: ChatCompletionChunkDelta;
  finish_reason: "stop" | "length" | "tool_calls" | null;
}

export interface ChatCompletionChunk {
  id: string;
  object: "chat.completion.chunk";
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
  usage?: TokenUsage;
}

// ── Text Completions ────────────────────────────────────────────────────────

export interface CompletionRequest extends SamplingParams {
  model?: string;
  prompt: string | string[];
  echo?: boolean;
  suffix?: string;
}

export interface CompletionChoice {
  index: number;
  text: string;
  finish_reason: "stop" | "length";
}

export interface CompletionResponse {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  choices: CompletionChoice[];
  usage: TokenUsage;
}

export interface CompletionChunkChoice {
  index: number;
  text: string;
  finish_reason: "stop" | "length" | null;
}

export interface CompletionChunk {
  id: string;
  object: "text_completion";
  created: number;
  model: string;
  choices: CompletionChunkChoice[];
  usage?: TokenUsage;
}

// ── Embeddings ──────────────────────────────────────────────────────────────

export interface EmbeddingRequest {
  model?: string;
  input: string | string[];
  encoding_format?: "float" | "base64";
  user?: string;
}

export interface EmbeddingObject {
  object: "embedding";
  embedding: number[] | string; // number[] for float, string for base64
  index: number;
}

export interface EmbeddingResponse {
  object: "list";
  data: EmbeddingObject[];
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

// ── Tokenization ────────────────────────────────────────────────────────────

export interface TokenizeRequest {
  model?: string;
  input: string;
  add_special_tokens?: boolean;
}

export interface TokenizeResponse {
  tokens: number[];
  count: number;
}

export interface DetokenizeRequest {
  model?: string;
  tokens: number[];
}

export interface DetokenizeResponse {
  text: string;
}

// ── Models ──────────────────────────────────────────────────────────────────

export interface ModelObject {
  id: string;
  object: "model";
  created: number;
  owned_by: string;
}

export interface ModelListResponse {
  object: "list";
  data: ModelObject[];
}

// ── Health ───────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: "ok";
  engine: string;
  device: string;
  models: Record<string, string>;
}

// ── Errors ──────────────────────────────────────────────────────────────────

export interface ErrorResponse {
  error: {
    message: string;
    type: string;
    param: string | null;
    code: string | null;
  };
}

// ── Internal ────────────────────────────────────────────────────────────────

export interface GenerationResult {
  text: string;
  promptTokens: number;
  completionTokens: number;
}

export interface GenerationOptions {
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  top_k?: number;
  min_p?: number;
  typical_p?: number;
  do_sample: boolean;
  repetition_penalty?: number;
  no_repeat_ngram_size?: number;
  eos_token_id?: number[];
}
