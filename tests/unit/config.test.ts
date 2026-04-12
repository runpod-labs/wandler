import { describe, expect, it } from "vitest";
import { loadConfig } from "../../src/config.js";

describe("loadConfig", () => {
  it("returns defaults when no env vars set", () => {
    const config = loadConfig({});
    expect(config.port).toBe(8000);
    expect(config.modelId).toBe("onnx-community/gemma-4-E4B-it-ONNX");
    expect(config.modelDtype).toBe("q4");
    expect(config.device).toBe("webgpu");
    expect(config.sttModelId).toBe("onnx-community/whisper-tiny");
    expect(config.sttDtype).toBe("q4");
    expect(config.embeddingModelId).toBe("");
    expect(config.embeddingDtype).toBe("q8");
  });

  it("reads from env vars", () => {
    const config = loadConfig({
      PORT: "3000",
      MODEL_ID: "my-model",
      DTYPE: "fp16",
      DEVICE: "cpu",
      STT_MODEL_ID: "my-stt",
      STT_DTYPE: "q8",
    });
    expect(config.port).toBe(3000);
    expect(config.modelId).toBe("my-model");
    expect(config.modelDtype).toBe("fp16");
    expect(config.device).toBe("cpu");
    expect(config.sttModelId).toBe("my-stt");
    expect(config.sttDtype).toBe("q8");
  });

  it("handles partial env vars", () => {
    const config = loadConfig({ PORT: "9090" });
    expect(config.port).toBe(9090);
    expect(config.modelId).toBe("onnx-community/gemma-4-E4B-it-ONNX"); // default
  });
});
