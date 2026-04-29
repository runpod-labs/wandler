import { serve } from "@hono/node-server";
import type { ServerConfig } from "../../src/config.js";
import type { LoadedModels } from "../../src/models/manager.js";
import type { Tokenizer } from "../../src/models/tokenizer.js";
import { resetMetrics } from "../../src/routes/admin.js";
import { createApp } from "../../src/server.js";

/** Minimal mock tokenizer compatible with transformers.js TextStreamer */
function createMockTokenizer(): Tokenizer {
  const tokenizer = function (text: string, _opts: Record<string, unknown>) {
    const tokens = text.split(/\s+/);
    return {
      input_ids: {
        dims: [1, tokens.length],
      },
    };
  };

  tokenizer.apply_chat_template = (
    messages: Array<{ role: string; content: string | null }>,
    _opts: Record<string, unknown>,
  ): string => {
    return (
      messages.map((m) => `<|${m.role}|>\n${m.content}`).join("\n") +
      "\n<|assistant|>\n"
    );
  };

  tokenizer.batch_decode = (
    _ids: unknown,
    _opts: Record<string, unknown>,
  ): string[] => {
    return ["Hello! I'm a mock assistant response."];
  };

  tokenizer.all_special_ids = [];
  tokenizer.decode = (
    _ids: unknown,
    _opts: Record<string, unknown>,
  ): string => {
    return "mock token ";
  };

  return tokenizer as unknown as Tokenizer;
}

interface Streamer {
  put(value: bigint[][]): void;
  end(): void;
  callback_function?: (text: string) => void;
}

/**
 * Mock model whose `.generate()` emits a fixed 7-token burst through the
 * real `TextStreamer.put()` protocol. Used by the default test server.
 */
export function createMockModels(): LoadedModels {
  const tokenizer = createMockTokenizer();

  const model = {
    async generate(opts: Record<string, unknown>) {
      const streamer = opts.streamer as Streamer | undefined;
      if (streamer) {
        const mockTokenIds = [1n, 2n, 3n, 4n, 5n, 6n, 7n];
        streamer.put([mockTokenIds]);
        for (const tokenId of mockTokenIds) {
          streamer.put([[tokenId]]);
        }
        streamer.end();
      }

      const inputDims = (opts.input_ids as { dims: number[] })?.dims;
      const promptLen = inputDims?.[1] ?? 10;
      return {
        dims: [1, promptLen + 7],
        slice(_dim: unknown, _range: unknown) {
          return { data: [1, 2, 3, 4, 5, 6, 7] };
        },
      };
    },
    async dispose() {},
  };

  const transcriber = async (_input: Float32Array) => ({
    text: " This is a mock transcription. ",
  });

  const embedder = async (_input: string, _opts: Record<string, unknown>) => ({
    data: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5]),
  });

  return {
    tokenizer,
    chatTemplate: null,
    processor: null,
    isVision: false,
    model,
    transcriber,
    embedder,
    maxContextLength: 2048,
    vocabSize: 32000,
    generationDiagnostics: {
      numLogitsToKeepInput: false,
      numLogitsToKeepPatchedSessions: [],
    },
    attentionHeads: 8,
  };
}

/**
 * Mock model whose `.generate()` emits a scripted sequence of pre-decoded
 * text chunks by invoking `streamer.callback_function` directly. Bypasses
 * the real TextStreamer's token-cache / space-boundary machinery so a test
 * can pin down exactly what content the chat route sees.
 *
 * Used to drive the streaming-tool-call e2e tests where we need to control
 * the raw text flow (e.g. a Qwen-style `<tool_call>…</tool_call>` split
 * across chunks).
 */
export function createMockModelsWithChunks(chunks: string[]): LoadedModels {
  const tokenizer = createMockTokenizer();

  const model = {
    async generate(opts: Record<string, unknown>) {
      const streamer = opts.streamer as Streamer | undefined;
      if (streamer?.callback_function) {
        // Bypass TextStreamer's decode/buffer layer — feed the callback the
        // pre-decoded chunks one at a time so we can test the downstream
        // stream-tools state machine without having to reverse-engineer
        // TextStreamer's space-boundary emission rules.
        for (const chunk of chunks) streamer.callback_function(chunk);
      }

      const inputDims = (opts.input_ids as { dims: number[] })?.dims;
      const promptLen = inputDims?.[1] ?? 10;
      return {
        dims: [1, promptLen + chunks.length],
        slice(_dim: unknown, _range: unknown) {
          return { data: chunks.map((_, i) => i + 1) };
        },
      };
    },
    async dispose() {},
  };

  return {
    tokenizer,
    chatTemplate: null,
    processor: null,
    isVision: false,
    model: model as unknown as LoadedModels["model"],
    transcriber: null,
    embedder: null,
  } as LoadedModels;
}

/**
 * Same as `startTestServer` but uses the scripted mock. Every request the
 * server handles will replay `chunks` verbatim through the streamer.
 */
export async function startTestServerWithChunks(
  chunks: string[],
  configOverrides?: Partial<ServerConfig>,
): Promise<TestServer> {
  const config = createTestConfig(configOverrides);
  const models = createMockModelsWithChunks(chunks);
  const app = createApp(config, models);

  return new Promise((resolve) => {
    const server = serve({ fetch: app.fetch, port: 0 }, (info) => {
      const baseUrl = `http://localhost:${info.port}`;
      resolve({
        baseUrl,
        close: () =>
          new Promise<void>((res) => {
            server.close(() => res());
          }),
      });
    });
  });
}

export function createTestConfig(
  overrides?: Partial<ServerConfig>,
): ServerConfig {
  return {
    port: 0,
    host: "127.0.0.1",
    modelId: "mock-model/test",
    modelDtype: "q4",
    device: "cpu",
    sttModelId: "mock-stt/test",
    sttDtype: "q4",
    embeddingModelId: "mock-embedding/test",
    embeddingDtype: "q8",
    apiKey: "",
    corsOrigin: "*",
    maxTokens: 2048,
    maxConcurrent: 1,
    timeout: 120000,
    logLevel: "info",
    hfToken: "",
    cacheDir: "",
    ...overrides,
  };
}

export interface TestServer {
  baseUrl: string;
  close: () => Promise<void>;
}

export async function startTestServer(
  configOverrides?: Partial<ServerConfig>,
): Promise<TestServer> {
  const config = createTestConfig(configOverrides);
  const models = createMockModels();
  return startTestServerWithLoadedModels(models, config);
}

export async function startTestServerWithLoadedModels(
  models: LoadedModels,
  config: ServerConfig,
): Promise<TestServer> {
  resetMetrics();
  const app = createApp(config, models);

  return new Promise((resolve) => {
    const server = serve({ fetch: app.fetch, port: 0 }, (info) => {
      const baseUrl = `http://localhost:${info.port}`;
      resolve({
        baseUrl,
        close: () =>
          new Promise<void>((res) => {
            server.close(() => res());
          }),
      });
    });
  });
}

export async function postJson(
  url: string,
  body: unknown,
): Promise<{ status: number; body: unknown }> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await res.text();
  try {
    return { status: res.status, body: JSON.parse(text) };
  } catch {
    return { status: res.status, body: text };
  }
}

export async function collectSSE(
  url: string,
  body: unknown,
): Promise<{ status: number; events: unknown[] }> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  const events: unknown[] = [];

  for (const line of text.split("\n")) {
    if (line.startsWith("data: ") || line.startsWith("data:")) {
      const data = line.startsWith("data: ") ? line.slice(6) : line.slice(5);
      if (data === "[DONE]") break;
      try {
        events.push(JSON.parse(data));
      } catch {
        // skip
      }
    }
  }

  return { status: res.status, events };
}

export interface NamedSSEEvent {
  event: string;
  data: unknown;
}

/**
 * Collect named SSE events (used by the Responses API).
 * Each event has `event: <type>\ndata: <json>\n\n`.
 */
export async function collectNamedSSE(
  url: string,
  body: unknown,
): Promise<{ status: number; events: NamedSSEEvent[] }> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  const events: NamedSSEEvent[] = [];
  let currentEvent = "";

  for (const line of text.split("\n")) {
    if (line.startsWith("event: ") || line.startsWith("event:")) {
      currentEvent = line.startsWith("event: ") ? line.slice(7) : line.slice(6);
    } else if (line.startsWith("data: ") || line.startsWith("data:")) {
      const data = line.startsWith("data: ") ? line.slice(6) : line.slice(5);
      try {
        events.push({ event: currentEvent, data: JSON.parse(data) });
      } catch {
        // skip
      }
      currentEvent = "";
    }
  }

  return { status: res.status, events };
}
