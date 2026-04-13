import { serve } from "@hono/node-server";
import type { ServerConfig } from "../../src/config.js";
import type { LoadedModels } from "../../src/models/manager.js";
import type { Tokenizer } from "../../src/models/tokenizer.js";
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
}

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

  return { tokenizer, chatTemplate: null, processor: null, isVision: false, model, transcriber, embedder };
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
