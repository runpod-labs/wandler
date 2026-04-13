#!/usr/bin/env node
import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { loadConfig } from "./config.js";
import { loadModels } from "./models/manager.js";
import { startServer } from "./server.js";

const { values } = parseArgs({
  options: {
    llm: { type: "string", short: "l" },
    embedding: { type: "string", short: "e" },
    stt: { type: "string", short: "s" },
    device: { type: "string", short: "d" },
    port: { type: "string", short: "p" },
    host: { type: "string" },
    "api-key": { type: "string", short: "k" },
    "cors-origin": { type: "string" },
    "max-tokens": { type: "string" },
    "max-concurrent": { type: "string" },
    timeout: { type: "string" },
    "log-level": { type: "string" },
    "hf-token": { type: "string" },
    "cache-dir": { type: "string" },
    "no-stt": { type: "boolean" },
    version: { type: "boolean", short: "v" },
    help: { type: "boolean", short: "h" },
  },
  strict: true,
  allowPositionals: false,
});

if (values.version) {
  const __dirname = dirname(fileURLToPath(import.meta.url));
  const pkg = JSON.parse(readFileSync(resolve(__dirname, "..", "package.json"), "utf-8"));
  console.log(pkg.version);
  process.exit(0);
}

if (values.help) {
  console.log(`
wandler — inference server for transformers.js

Usage:
  wandler --llm org/repo[:precision] [options]

Model:
  -l, --llm <id>              LLM model (default: onnx-community/gemma-4-E4B-it-ONNX:q4)
  -e, --embedding <id>        Embedding model (disabled by default)
  -s, --stt <id>              STT model (default: onnx-community/whisper-tiny:q4)
      --no-stt                Disable STT
  -d, --device <type>         Device: auto, webgpu, cpu, wasm (default: auto)
      --hf-token <token>      HuggingFace token for gated models
      --cache-dir <path>      Model cache directory

Server:
  -p, --port <number>         Port (default: 8000)
      --host <addr>           Bind address (default: 127.0.0.1)
  -k, --api-key <key>         API key for auth (or WANDLER_API_KEY)
      --cors-origin <origin>  Allowed CORS origin (default: *)
      --max-tokens <n>        Max tokens per request (default: 2048)
      --max-concurrent <n>    Max concurrent requests (default: 1)
      --timeout <ms>          Request timeout in ms (default: 120000)
      --log-level <level>     debug, info, warn, error (default: info)

Info:
  -v, --version               Show version
  -h, --help                  Show this help

Precision suffixes: q4, q8, fp16, fp32 (default: q4)

Environment variables:
  WANDLER_LLM, WANDLER_STT, WANDLER_EMBEDDING, WANDLER_DEVICE,
  WANDLER_PORT, WANDLER_HOST, WANDLER_API_KEY, WANDLER_CORS_ORIGIN,
  WANDLER_MAX_TOKENS, WANDLER_MAX_CONCURRENT, WANDLER_TIMEOUT,
  WANDLER_LOG_LEVEL, WANDLER_CACHE_DIR, HF_TOKEN

Examples:
  wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4
  wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX:fp16 --device cpu --port 3000
  wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX --api-key secret123
  wandler --llm onnx-community/gemma-4-E4B-it-ONNX --embedding Xenova/all-MiniLM-L6-v2:q8
`);
  process.exit(0);
}

// CLI flags override env vars
const config = loadConfig({
  WANDLER_LLM: values.llm ?? process.env.WANDLER_LLM ?? process.env.MODEL_ID,
  WANDLER_STT: values["no-stt"] ? "" : (values.stt ?? process.env.WANDLER_STT ?? process.env.STT_MODEL_ID),
  WANDLER_EMBEDDING: values.embedding ?? process.env.WANDLER_EMBEDDING ?? process.env.EMBEDDING_MODEL_ID,
  WANDLER_DEVICE: values.device ?? process.env.WANDLER_DEVICE ?? process.env.DEVICE,
  WANDLER_PORT: values.port ?? process.env.WANDLER_PORT ?? process.env.PORT,
  WANDLER_HOST: values.host ?? process.env.WANDLER_HOST,
  WANDLER_API_KEY: values["api-key"] ?? process.env.WANDLER_API_KEY,
  WANDLER_CORS_ORIGIN: values["cors-origin"] ?? process.env.WANDLER_CORS_ORIGIN,
  WANDLER_MAX_TOKENS: values["max-tokens"] ?? process.env.WANDLER_MAX_TOKENS,
  WANDLER_MAX_CONCURRENT: values["max-concurrent"] ?? process.env.WANDLER_MAX_CONCURRENT,
  WANDLER_TIMEOUT: values.timeout ?? process.env.WANDLER_TIMEOUT,
  WANDLER_LOG_LEVEL: values["log-level"] ?? process.env.WANDLER_LOG_LEVEL,
  HF_TOKEN: values["hf-token"] ?? process.env.HF_TOKEN,
  WANDLER_CACHE_DIR: values["cache-dir"] ?? process.env.WANDLER_CACHE_DIR,
});

const models = await loadModels(config);
startServer(config, models);

console.log(`[wandler] http://${config.host}:${config.port}`);
console.log(`[wandler] LLM: ${config.modelId} (${config.modelDtype}, ${config.device})`);
if (config.sttModelId) console.log(`[wandler] STT: ${config.sttModelId} (${config.sttDtype})`);
if (config.embeddingModelId) console.log(`[wandler] Embedding: ${config.embeddingModelId} (${config.embeddingDtype})`);
if (config.apiKey) console.log(`[wandler] API key auth enabled`);
