import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildSessionOptions } from "../../../src/models/manager.js";

describe("buildSessionOptions — TurboQuant plumbing", () => {
  const originalBoundary = process.env.WANDLER_KV_CACHE_BOUNDARY;
  beforeEach(() => {
    delete process.env.WANDLER_KV_CACHE_BOUNDARY;
  });
  afterEach(() => {
    if (originalBoundary === undefined) {
      delete process.env.WANDLER_KV_CACHE_BOUNDARY;
    } else {
      process.env.WANDLER_KV_CACHE_BOUNDARY = originalBoundary;
    }
  });

  it("returns undefined when nothing is requested (cpu + default dtype)", () => {
    const opts = buildSessionOptions("cpu", "default");
    expect(opts).toBeUndefined();
  });

  it("does not set TurboQuant extras for non-TQ dtypes", () => {
    const opts = buildSessionOptions("webgpu", "q4f16");
    expect(opts).toBeDefined();
    expect(opts).not.toHaveProperty("extra");
  });

  // The core of the contract: every TQ preset routes to the same two
  // session-config entries that the patched ORT graph rewriter reads.
  // If this test changes, the wandler→ORT bridge has shifted and the
  // browser/Node/CUDA benches need re-running.
  it.each([
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
  ])("sets ORT session extras for preset %s", (preset) => {
    const opts = buildSessionOptions("webgpu", preset);
    expect(opts?.extra).toEqual({
      "optimization.turboquant_kv_method": preset,
      "optimization.turboquant_kv_boundary": "0",
    });
  });

  it("honours WANDLER_KV_CACHE_BOUNDARY override", () => {
    process.env.WANDLER_KV_CACHE_BOUNDARY = "2";
    const opts = buildSessionOptions("webgpu", "turboquant_4bit_nc");
    expect((opts?.extra as Record<string, string>)["optimization.turboquant_kv_boundary"]).toBe("2");
  });

  it("combines TQ extras with the webgpu execution provider", () => {
    const opts = buildSessionOptions("webgpu", "turboquant_4bit_nc");
    expect(opts?.executionProviders).toEqual(["webgpu", "cpu"]);
    expect(opts?.extra).toBeDefined();
  });

  it("combines TQ extras with the cuda execution provider", () => {
    const opts = buildSessionOptions("cuda", "turboquant_4bit_nc");
    const eps = opts?.executionProviders as Array<unknown>;
    expect(Array.isArray(eps)).toBe(true);
    expect(eps[0]).toMatchObject({ name: "cuda" });
    expect(eps[eps.length - 1]).toBe("cpu");
    expect(opts?.extra).toMatchObject({
      "optimization.turboquant_kv_method": "turboquant_4bit_nc",
    });
  });

  it("ignores unknown presets (no extra set)", () => {
    // If a user passes a typo'd preset, we don't silently inject the wrong
    // option. Only known turboquant_* prefixes flip the extras on.
    const opts = buildSessionOptions("webgpu", "turbo_5bit");
    expect(opts?.extra).toBeUndefined();
  });
});
