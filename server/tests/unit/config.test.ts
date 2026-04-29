import { homedir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadConfig, parseModelRef } from "../../src/config.js";

describe("parseModelRef", () => {
  it("parses org/repo:precision", () => {
    expect(parseModelRef("onnx-community/gemma:q4", "q4")).toEqual({
      id: "onnx-community/gemma",
      dtype: "q4",
    });
  });

  it("parses org/repo:fp16", () => {
    expect(parseModelRef("LiquidAI/LFM2.5-1.2B-Instruct-ONNX:fp16", "q4")).toEqual({
      id: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX",
      dtype: "fp16",
    });
  });

  it("uses default dtype when no suffix", () => {
    expect(parseModelRef("onnx-community/gemma-4-E4B-it-ONNX", "q4")).toEqual({
      id: "onnx-community/gemma-4-E4B-it-ONNX",
      dtype: "q4",
    });
  });

  it("handles bare model name without org", () => {
    expect(parseModelRef("local-model", "fp32")).toEqual({
      id: "local-model",
      dtype: "fp32",
    });
  });

  it("does not split on colon that appears before slash", () => {
    // Edge case: no slash in the string, colon present
    expect(parseModelRef("model:q8", "q4")).toEqual({
      id: "model:q8",
      dtype: "q4",
    });
  });
});

describe("loadConfig", () => {
  it("returns defaults when no env vars set", () => {
    const config = loadConfig({});
    expect(config.port).toBe(8000);
    expect(config.host).toBe("127.0.0.1");
    expect(config.modelId).toBe("");
    expect(config.modelDtype).toBe("q4");
    expect(config.backend).toBe("wandler");
    expect(config.device).toBe("auto");
    expect(config.sttModelId).toBe("");
    expect(config.sttDtype).toBe("q4");
    expect(config.embeddingModelId).toBe("");
    expect(config.embeddingDtype).toBe("q8");
    expect(config.apiKey).toBe("");
    expect(config.corsOrigin).toBe("*");
    // null means "no server-side cap" — the effective cap is the loaded
    // model's max_position_embeddings, resolved in buildGenOpts.
    expect(config.maxTokens).toBeNull();
    expect(config.maxConcurrent).toBe(1);
    expect(config.timeout).toBe(120000);
    expect(config.logLevel).toBe("info");
    expect(config.quiet).toBe(false);
    expect(config.hfToken).toBe("");
    expect(config.cacheDir).toBe(join(homedir(), ".cache", "huggingface"));
    expect(config.prefillChunkSize).toBe("auto");
    expect(config.warmupTokens).toBe(0);
    expect(config.warmupMaxNewTokens).toBe(8);
  });

  it("uses HF_HOME as cache dir when set", () => {
    const config = loadConfig({ HF_HOME: "/custom/hf/home" });
    expect(config.cacheDir).toBe("/custom/hf/home");
  });

  it("uses XDG_CACHE_HOME/huggingface when HF_HOME is not set", () => {
    const config = loadConfig({ XDG_CACHE_HOME: "/custom/xdg/cache" });
    expect(config.cacheDir).toBe(join("/custom/xdg/cache", "huggingface"));
  });

  it("WANDLER_CACHE_DIR takes priority over HF_HOME", () => {
    const config = loadConfig({
      WANDLER_CACHE_DIR: "/explicit/cache",
      HF_HOME: "/hf/home",
    });
    expect(config.cacheDir).toBe("/explicit/cache");
  });

  it("reads WANDLER_ prefixed env vars", () => {
    const config = loadConfig({
      WANDLER_LLM: "my-org/my-model:fp16",
      WANDLER_BACKEND: "transformersjs",
      WANDLER_PORT: "3000",
      WANDLER_HOST: "0.0.0.0",
      WANDLER_DEVICE: "cpu",
      WANDLER_API_KEY: "secret",
      WANDLER_CORS_ORIGIN: "https://example.com",
      WANDLER_MAX_TOKENS: "4096",
      WANDLER_MAX_CONCURRENT: "4",
      WANDLER_TIMEOUT: "60000",
      WANDLER_LOG_LEVEL: "debug",
      WANDLER_QUIET: "true",
      WANDLER_CACHE_DIR: "/tmp/models",
      WANDLER_PREFILL_CHUNK_SIZE: "1024",
      WANDLER_WARMUP_TOKENS: "2048",
      WANDLER_WARMUP_MAX_NEW_TOKENS: "4",
      HF_TOKEN: "hf_abc123",
    });
    expect(config.port).toBe(3000);
    expect(config.host).toBe("0.0.0.0");
    expect(config.modelId).toBe("my-org/my-model");
    expect(config.modelDtype).toBe("fp16");
    expect(config.backend).toBe("transformersjs");
    expect(config.device).toBe("cpu");
    expect(config.apiKey).toBe("secret");
    expect(config.corsOrigin).toBe("https://example.com");
    expect(config.maxTokens).toBe(4096);
    expect(config.maxConcurrent).toBe(4);
    expect(config.timeout).toBe(60000);
    expect(config.logLevel).toBe("debug");
    expect(config.quiet).toBe(true);
    expect(config.cacheDir).toBe("/tmp/models");
    expect(config.hfToken).toBe("hf_abc123");
    expect(config.prefillChunkSize).toBe("1024");
    expect(config.warmupTokens).toBe(2048);
    expect(config.warmupMaxNewTokens).toBe(4);
  });

  it("reads legacy env vars as fallback", () => {
    const config = loadConfig({
      MODEL_ID: "legacy-org/legacy-model",
      DTYPE: "q8",
      DEVICE: "cpu",
      PORT: "3000",
      STT_MODEL_ID: "legacy-stt/model",
      STT_DTYPE: "q8",
    });
    expect(config.modelId).toBe("legacy-org/legacy-model");
    expect(config.modelDtype).toBe("q8");
    expect(config.device).toBe("cpu");
    expect(config.port).toBe(3000);
    expect(config.sttModelId).toBe("legacy-stt/model");
    expect(config.sttDtype).toBe("q8");
  });

  it("WANDLER_ vars take priority over legacy vars", () => {
    const config = loadConfig({
      WANDLER_LLM: "new-org/new-model:fp16",
      MODEL_ID: "old-org/old-model",
      DTYPE: "q4",
    });
    expect(config.modelId).toBe("new-org/new-model");
    expect(config.modelDtype).toBe("fp16");
  });

  it("parses precision from model references", () => {
    const config = loadConfig({
      WANDLER_LLM: "org/llm:q8",
      WANDLER_STT: "org/stt:fp16",
      WANDLER_EMBEDDING: "org/emb:q4",
    });
    expect(config.modelId).toBe("org/llm");
    expect(config.modelDtype).toBe("q8");
    expect(config.sttModelId).toBe("org/stt");
    expect(config.sttDtype).toBe("fp16");
    expect(config.embeddingModelId).toBe("org/emb");
    expect(config.embeddingDtype).toBe("q4");
  });

  it("handles partial env vars", () => {
    const config = loadConfig({ WANDLER_PORT: "9090" });
    expect(config.port).toBe(9090);
    expect(config.modelId).toBe("");
  });

  it("disables STT when empty string", () => {
    const config = loadConfig({ WANDLER_STT: "" });
    expect(config.sttModelId).toBe("");
  });
});
