// ── OpenAI-compatible type definitions ──────────────────────────────────────

export interface ChatMessage {
  role: "system" | "user" | "assistant" | "tool";
  content: string | null;
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

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
  stream_options?: { include_usage?: boolean };
  stop?: string | string[];
  presence_penalty?: number;
  frequency_penalty?: number;
  logit_bias?: Record<string, number>;
  seed?: number;
  n?: number;
  response_format?: { type: "text" | "json_object" };
  tools?: Tool[];
  tool_choice?: "none" | "auto" | "required" | { type: "function"; function: { name: string } };
  user?: string;
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

export interface HealthResponse {
  status: "ok";
  engine: string;
  device: string;
  models: Record<string, string>;
}

export interface ErrorResponse {
  error: {
    message: string;
    type: string;
    param: string | null;
    code: string | null;
  };
}

export interface GenerationResult {
  text: string;
  promptTokens: number;
  completionTokens: number;
}

export interface GenerationOptions {
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  do_sample: boolean;
  repetition_penalty?: number;
  eos_token_id?: number[];
}
