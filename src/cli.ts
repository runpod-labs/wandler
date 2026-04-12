#!/usr/bin/env node
import { parseArgs } from "node:util";
import { loadConfig } from "./config.js";
import { loadModels } from "./models/manager.js";
import { createServer } from "./server.js";

const { values } = parseArgs({
  options: {
    llm: { type: "string", short: "l", description: "LLM model ID from HuggingFace" },
    embedding: { type: "string", short: "e", description: "Embedding model ID" },
    stt: { type: "string", short: "s", description: "Speech-to-text model ID" },
    device: { type: "string", short: "d", description: "Device: webgpu, cpu, wasm" },
    dtype: { type: "string", description: "Quantization: q4, q8, fp16, fp32" },
    port: { type: "string", short: "p", description: "Server port" },
    "api-key": { type: "string", short: "k", description: "API key for authentication" },
    "no-stt": { type: "boolean", description: "Disable speech-to-text model" },
    help: { type: "boolean", short: "h" },
  },
  strict: true,
  allowPositionals: false,
});

if (values.help) {
  console.log(`
wandler — OpenAI-compatible inference server powered by transformers.js

Usage:
  wandler --llm <model-id> [options]

Options:
  -l, --llm <id>        LLM model ID (default: onnx-community/gemma-4-E4B-it-ONNX)
  -e, --embedding <id>  Embedding model ID (disabled by default)
  -s, --stt <id>        STT model ID (default: onnx-community/whisper-tiny)
      --no-stt          Disable STT model
  -d, --device <type>   Device: webgpu, cpu, wasm (default: webgpu)
      --dtype <type>    Quantization: q4, q8, fp16, fp32 (default: q4)
  -p, --port <number>   Server port (default: 8000)
  -k, --api-key <key>   API key for auth (or WANDLER_API_KEY env var)
  -h, --help            Show this help

Examples:
  wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX
  wandler --llm LiquidAI/LFM2.5-350M-ONNX --device cpu --port 3000
  wandler --llm onnx-community/Qwen3.5-0.8B-Text-ONNX --api-key secret123
  wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX --embedding Xenova/all-MiniLM-L6-v2
`);
  process.exit(0);
}

// CLI flags override env vars
const config = loadConfig({
  MODEL_ID: values.llm ?? process.env.MODEL_ID,
  DTYPE: values.dtype ?? process.env.DTYPE,
  DEVICE: values.device ?? process.env.DEVICE,
  PORT: values.port ?? process.env.PORT,
  STT_MODEL_ID: values["no-stt"] ? "" : (values.stt ?? process.env.STT_MODEL_ID),
  STT_DTYPE: process.env.STT_DTYPE,
  EMBEDDING_MODEL_ID: values.embedding ?? process.env.EMBEDDING_MODEL_ID,
  EMBEDDING_DTYPE: process.env.EMBEDDING_DTYPE,
  WANDLER_API_KEY: values["api-key"] ?? process.env.WANDLER_API_KEY,
});

const models = await loadModels(config);
const server = createServer(config, models);

server.listen(config.port, () => {
  console.log(`[wandler] http://localhost:${config.port}`);
  if (config.apiKey) {
    console.log(`[wandler] API key auth enabled`);
  }
});
