import { describe, expect, it, vi, beforeEach } from "vitest";
import { runModelLsCommand } from "../../../src/catalog/models-command.js";

vi.mock("../../../src/catalog/catalog.js", () => ({
  loadCatalog: vi.fn().mockResolvedValue({
    version: "1",
    models: [
      {
        id: "org/llm-model",
        name: "Test LLM",
        type: "llm",
        size: "1B",
        precisions: ["q4", "q8"],
        defaultPrecision: "q4",
        capabilities: ["chat", "tool-calling"],
        status: "stable",
      },
      {
        id: "org/embed-model",
        name: "Test Embedding",
        type: "embedding",
        size: "22M",
        precisions: ["q8"],
        defaultPrecision: "q8",
        capabilities: ["embedding"],
        dimensions: 384,
        status: "stable",
      },
      {
        id: "org/stt-model",
        name: "Test STT",
        type: "stt",
        size: "39M",
        precisions: ["q4"],
        defaultPrecision: "q4",
        capabilities: ["transcription"],
        status: "stable",
      },
    ],
  }),
}));

describe("wandler model ls", () => {
  let output: string[];

  beforeEach(() => {
    output = [];
    vi.spyOn(console, "log").mockImplementation((...args: unknown[]) => {
      output.push(args.join(" "));
    });
  });

  it("lists all models when no type filter", async () => {
    await runModelLsCommand({});
    const joined = output.join("\n");
    expect(joined).toContain("org/llm-model:q4");
    expect(joined).toContain("org/embed-model:q8");
    expect(joined).toContain("org/stt-model:q4");
    expect(joined).toContain("3 model(s) found");
  });

  it("filters by type llm", async () => {
    await runModelLsCommand({ type: "llm" });
    const joined = output.join("\n");
    expect(joined).toContain("org/llm-model:q4");
    expect(joined).not.toContain("org/embed-model");
    expect(joined).not.toContain("org/stt-model");
    expect(joined).toContain("1 model(s) found");
  });

  it("filters by type embedding", async () => {
    await runModelLsCommand({ type: "embedding" });
    const joined = output.join("\n");
    expect(joined).toContain("org/embed-model:q8");
    expect(joined).not.toContain("org/llm-model");
    expect(joined).toContain("1 model(s) found");
  });

  it("filters by type stt", async () => {
    await runModelLsCommand({ type: "stt" });
    const joined = output.join("\n");
    expect(joined).toContain("org/stt-model:q4");
    expect(joined).not.toContain("org/llm-model");
    expect(joined).toContain("1 model(s) found");
  });

  it("shows capabilities in output", async () => {
    await runModelLsCommand({ type: "llm" });
    const joined = output.join("\n");
    expect(joined).toContain("chat, tool-calling");
  });

  it("exits with error for unknown type", async () => {
    const mockExit = vi.spyOn(process, "exit").mockImplementation(() => undefined as never);
    const mockError = vi.spyOn(console, "error").mockImplementation(() => {});
    await runModelLsCommand({ type: "bogus" });
    expect(mockError).toHaveBeenCalledWith(expect.stringContaining('unknown type "bogus"'));
    expect(mockExit).toHaveBeenCalledWith(1);
    mockExit.mockRestore();
    mockError.mockRestore();
  });
});
