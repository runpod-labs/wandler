import http from "node:http";
import type { ServerConfig } from "../../src/config.js";
import type { LoadedModels } from "../../src/models/manager.js";
import type { Tokenizer } from "../../src/models/tokenizer.js";
import { createServer } from "../../src/server.js";

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

  // TextStreamer needs these for initialization and token decoding
  tokenizer.all_special_ids = [];
  tokenizer.decode = (
    _ids: unknown,
    _opts: Record<string, unknown>,
  ): string => {
    return "mock token ";
  };

  return tokenizer as unknown as Tokenizer;
}

// Streamer interface matching TextStreamer's put/end contract
interface Streamer {
  put(value: bigint[][]): void;
  end(): void;
}

/** Create mock models that don't require downloading real weights */
export function createMockModels(): LoadedModels {
  const tokenizer = createMockTokenizer();

  const model = {
    async generate(opts: Record<string, unknown>) {
      // Call the streamer using the TextStreamer API (put/end) if present
      const streamer = opts.streamer as Streamer | undefined;
      if (streamer) {
        // Simulate generating 7 tokens by calling put() with mock token IDs
        const mockTokenIds = [1n, 2n, 3n, 4n, 5n, 6n, 7n];
        // First put is the prompt (will be skipped by skip_prompt)
        streamer.put([mockTokenIds]);
        // Then individual generated tokens
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

  return { tokenizer, model, transcriber, embedder };
}

export function createTestConfig(
  overrides?: Partial<ServerConfig>,
): ServerConfig {
  return {
    port: 0,
    modelId: "mock-model/test",
    modelDtype: "q4",
    device: "cpu",
    sttModelId: "mock-stt/test",
    sttDtype: "q4",
    embeddingModelId: "mock-embedding/test",
    embeddingDtype: "q8",
    ...overrides,
  };
}

export interface TestServer {
  baseUrl: string;
  server: http.Server;
  close: () => Promise<void>;
}

/** Start a test server on a random port and return the base URL */
export async function startTestServer(
  configOverrides?: Partial<ServerConfig>,
): Promise<TestServer> {
  const config = createTestConfig(configOverrides);
  const models = createMockModels();
  const server = createServer(config, models);

  return new Promise((resolve) => {
    server.listen(0, () => {
      const addr = server.address() as { port: number };
      const baseUrl = `http://localhost:${addr.port}`;
      resolve({
        baseUrl,
        server,
        close: () =>
          new Promise<void>((res) => {
            server.close(() => res());
          }),
      });
    });
  });
}

/** Simple fetch helper for JSON requests */
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

/** Parse SSE stream into data events */
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
    if (line.startsWith("data: ")) {
      const data = line.slice(6);
      if (data === "[DONE]") break;
      try {
        events.push(JSON.parse(data));
      } catch {
        // skip unparseable
      }
    }
  }

  return { status: res.status, events };
}
