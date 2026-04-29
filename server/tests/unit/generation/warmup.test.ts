import { describe, expect, it } from "vitest";
import { warmupLLM } from "../../../src/generation/warmup.js";
import { createMockModels, createTestConfig } from "../../e2e/helpers.js";

describe("warmupLLM", () => {
  it("is disabled by default", async () => {
    await expect(warmupLLM(createTestConfig(), createMockModels())).resolves.toEqual({
      enabled: false,
    });
  });

  it("runs a small non-sampling generation when enabled", async () => {
    const result = await warmupLLM(
      createTestConfig({ warmupTokens: 8, warmupMaxNewTokens: 3 }),
      createMockModels(),
    );

    expect(result.enabled).toBe(true);
    expect(result.error).toBeUndefined();
    expect(result.promptTokens).toBeGreaterThan(0);
    expect(result.completionTokens).toBe(7);
    expect(result.totalMs).toBeGreaterThanOrEqual(0);
  });

  it("caps the warmup prompt below the configured context budget", async () => {
    const result = await warmupLLM(
      createTestConfig({ maxTokens: 32, warmupTokens: 10_000, warmupMaxNewTokens: 4 }),
      createMockModels(),
    );

    expect(result.enabled).toBe(false);
  });
});
