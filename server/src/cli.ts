#!/usr/bin/env node
import { Command } from "commander";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { loadConfig } from "./config.js";
import { loadModels } from "./models/manager.js";
import { startServer } from "./server.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(readFileSync(resolve(__dirname, "..", "package.json"), "utf-8"));

const program = new Command()
  .name("wandler")
  .description("transformers.js inference server")
  .version(pkg.version, "-v, --version");

// ── wandler model ls ────────────────────────────────────────────────────────

const model = program.command("model").description("Model management");

model
  .command("ls")
  .description("List available models from the catalog")
  .option("-t, --type <type>", "filter by type: llm, embedding, stt")
  .action(async (opts: { type?: string }) => {
    const { runModelLsCommand } = await import("./catalog/models-command.js");
    await runModelLsCommand(opts);
  });

// ── wandler (server mode, default) ─────────────────────────────────────────

program
  .option("-l, --llm <id>", "LLM model")
  .option("-e, --embedding <id>", "Embedding model")
  .option("-s, --stt <id>", "STT model")
  .option("-d, --device <type>", "Device: auto, cpu, cuda, coreml, dml, webgpu, wasm")
  .option("-p, --port <number>", "Port")
  .option("--host <addr>", "Bind address")
  .option("-k, --api-key <key>", "API key for auth")
  .option("--cors-origin <origin>", "Allowed CORS origin")
  .option("--max-tokens <n>", "Max tokens per request (default: model's max context)")
  .option("--max-concurrent <n>", "Max concurrent requests")
  .option("--timeout <ms>", "Request timeout in ms")
  .option("--log-level <level>", "debug, info, warn, error")
  .option("--hf-token <token>", "HuggingFace token for gated models")
  .option("--cache-dir <path>", "Model cache directory (default: ~/.cache/huggingface)")
  .action(async (opts) => {
    const config = loadConfig({
      WANDLER_LLM: opts.llm ?? process.env.WANDLER_LLM ?? process.env.MODEL_ID,
      WANDLER_STT: opts.stt ?? process.env.WANDLER_STT ?? process.env.STT_MODEL_ID,
      WANDLER_EMBEDDING: opts.embedding ?? process.env.WANDLER_EMBEDDING ?? process.env.EMBEDDING_MODEL_ID,
      WANDLER_DEVICE: opts.device ?? process.env.WANDLER_DEVICE ?? process.env.DEVICE,
      WANDLER_PORT: opts.port ?? process.env.WANDLER_PORT ?? process.env.PORT,
      WANDLER_HOST: opts.host ?? process.env.WANDLER_HOST,
      WANDLER_API_KEY: opts.apiKey ?? process.env.WANDLER_API_KEY,
      WANDLER_CORS_ORIGIN: opts.corsOrigin ?? process.env.WANDLER_CORS_ORIGIN,
      WANDLER_MAX_TOKENS: opts.maxTokens ?? process.env.WANDLER_MAX_TOKENS,
      WANDLER_MAX_CONCURRENT: opts.maxConcurrent ?? process.env.WANDLER_MAX_CONCURRENT,
      WANDLER_TIMEOUT: opts.timeout ?? process.env.WANDLER_TIMEOUT,
      WANDLER_LOG_LEVEL: opts.logLevel ?? process.env.WANDLER_LOG_LEVEL,
      HF_TOKEN: opts.hfToken ?? process.env.HF_TOKEN,
      WANDLER_CACHE_DIR: opts.cacheDir ?? process.env.WANDLER_CACHE_DIR,
      HF_HOME: process.env.HF_HOME,
      XDG_CACHE_HOME: process.env.XDG_CACHE_HOME,
    });

    if (!config.modelId && !config.embeddingModelId && !config.sttModelId) {
      console.error("[wandler] Error: at least one model is required (--llm, --embedding, or --stt)");
      process.exit(1);
    }

    const models = await loadModels(config);
    startServer(config, models);

    console.log(`[wandler] http://${config.host}:${config.port}`);
    if (config.modelId) console.log(`[wandler] LLM: ${config.modelId} (${config.modelDtype}, ${config.device})`);
    if (config.sttModelId) console.log(`[wandler] STT: ${config.sttModelId} (${config.sttDtype})`);
    if (config.embeddingModelId) console.log(`[wandler] Embedding: ${config.embeddingModelId} (${config.embeddingDtype})`);
    console.log(`[wandler] Cache: ${config.cacheDir}`);
    if (config.apiKey) console.log(`[wandler] API key auth enabled`);
  });

program.parse();
