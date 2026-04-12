/**
 * Comprehensive eval/benchmark suite for wandler.
 *
 * Runs diverse scenarios across models with multiple iterations,
 * warmup, and proper statistics (avg/min/max/p50/p95).
 *
 * Usage:
 *   npm run benchmark                              # all models, webgpu
 *   DEVICE=cpu npm run benchmark                   # all models, cpu
 *   MODEL_ID=LiquidAI/LFM2.5-350M-ONNX npm run benchmark  # single model
 *   RUNS=5 npm run benchmark                       # 5 runs per scenario
 */
import OpenAI from "openai";
import { loadConfig } from "./config.js";
import { loadModels } from "./models/manager.js";
import { startServer } from "./server.js";

// ── Config ──────────────────────────────────────────────────────────────────

const RUNS = parseInt(process.env.RUNS || "10", 10);
const WARMUP_RUNS = 2;
const DEVICE = process.env.DEVICE || "webgpu";

interface ModelDef {
  id: string;
  name: string;
  supportsTools: boolean;
  supportsMultiTurn: boolean;
}

const ALL_MODELS: ModelDef[] = [
  { id: "LiquidAI/LFM2.5-350M-ONNX", name: "LFM2.5-350M", supportsTools: false, supportsMultiTurn: true },
  { id: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX", name: "LFM2.5-1.2B", supportsTools: true, supportsMultiTurn: true },
  { id: "onnx-community/Qwen3.5-0.8B-Text-ONNX", name: "Qwen3.5-0.8B", supportsTools: true, supportsMultiTurn: true },
  { id: "onnx-community/gemma-4-E4B-it-ONNX", name: "Gemma-4-E4B", supportsTools: true, supportsMultiTurn: true },
];

// ── Scenarios ───────────────────────────────────────────────────────────────

interface Scenario {
  name: string;
  category: string;
  messages: OpenAI.ChatCompletionMessageParam[];
  maxTokens: number;
  stream: boolean;
  tools?: OpenAI.ChatCompletionTool[];
  requiresTools?: boolean;
  requiresMultiTurn?: boolean;
}

const SCENARIOS: Scenario[] = [
  // Simple / Baseline
  {
    name: "Hello world",
    category: "baseline",
    messages: [{ role: "user", content: "Say hello." }],
    maxTokens: 20,
    stream: false,
  },
  {
    name: "Hello world (stream)",
    category: "baseline",
    messages: [{ role: "user", content: "Say hello." }],
    maxTokens: 20,
    stream: true,
  },

  // Math
  {
    name: "Simple arithmetic",
    category: "math",
    messages: [{ role: "user", content: "What is 17 * 23? Just the number." }],
    maxTokens: 20,
    stream: false,
  },
  {
    name: "Word problem",
    category: "math",
    messages: [
      { role: "user", content: "A train travels at 60 mph for 2.5 hours. How far does it go? Answer with just the distance." },
    ],
    maxTokens: 30,
    stream: false,
  },

  // Code generation
  {
    name: "JS function",
    category: "code",
    messages: [
      { role: "system", content: "You are a coding assistant. Write concise, working code." },
      { role: "user", content: "Write a JavaScript function that reverses a string. Just the function, no explanation." },
    ],
    maxTokens: 100,
    stream: false,
  },
  {
    name: "HTML canvas game",
    category: "code",
    messages: [
      { role: "system", content: "You write concise HTML/JS code." },
      { role: "user", content: "Write a minimal HTML page with a canvas element that draws a bouncing red ball. Include the full HTML." },
    ],
    maxTokens: 300,
    stream: false,
  },

  // Reasoning
  {
    name: "Logical reasoning",
    category: "reasoning",
    messages: [
      { role: "user", content: "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain in one sentence." },
    ],
    maxTokens: 60,
    stream: false,
  },

  // Multi-turn
  {
    name: "Multi-turn conversation",
    category: "multi-turn",
    messages: [
      { role: "user", content: "What is the capital of France?" },
      { role: "assistant", content: "The capital of France is Paris." },
      { role: "user", content: "What is its population?" },
    ],
    maxTokens: 40,
    stream: false,
    requiresMultiTurn: true,
  },

  // Tool calling
  {
    name: "Tool call: weather",
    category: "tools",
    messages: [
      { role: "system", content: "You have access to tools. Use them when appropriate." },
      { role: "user", content: "What is the weather in San Francisco?" },
    ],
    maxTokens: 100,
    stream: false,
    requiresTools: true,
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get the current weather for a location",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string", description: "City name" },
              unit: { type: "string", enum: ["celsius", "fahrenheit"] },
            },
            required: ["location"],
          },
        },
      },
    ],
  },

  // Long generation (streaming)
  {
    name: "Long generation (stream)",
    category: "long",
    messages: [
      { role: "user", content: "Explain how a CPU works in detail." },
    ],
    maxTokens: 200,
    stream: true,
  },
];

// ── Stats ───────────────────────────────────────────────────────────────────

interface RunResult {
  tokPerSec: number;
  ttftMs: number;
  totalMs: number;
  promptTokens: number;
  completionTokens: number;
  output: string;
}

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)]!;
}

function stats(values: number[]): { avg: number; min: number; max: number; p50: number; p95: number } {
  const sorted = [...values].sort((a, b) => a - b);
  const avg = values.reduce((s, v) => s + v, 0) / values.length;
  return {
    avg: Math.round(avg * 10) / 10,
    min: Math.round(sorted[0]! * 10) / 10,
    max: Math.round(sorted[sorted.length - 1]! * 10) / 10,
    p50: Math.round(percentile(sorted, 50) * 10) / 10,
    p95: Math.round(percentile(sorted, 95) * 10) / 10,
  };
}

// ── Runner ──────────────────────────────────────────────────────────────────

async function runScenario(
  client: OpenAI,
  modelId: string,
  scenario: Scenario,
): Promise<RunResult> {
  const t0 = Date.now();

  if (scenario.stream) {
    const stream = await client.chat.completions.create({
      model: modelId,
      messages: scenario.messages,
      max_tokens: scenario.maxTokens,
      temperature: 0.3,
      stream: true,
      tools: scenario.tools,
    });

    let text = "";
    let ttft: number | null = null;
    let completionTokens = 0;
    let promptTokens = 0;

    for await (const chunk of stream) {
      if (ttft === null) ttft = Date.now() - t0;
      const content = chunk.choices[0]?.delta?.content ?? "";
      text += content;
      if (content) completionTokens++;
      if (chunk.usage) {
        promptTokens = chunk.usage.prompt_tokens;
        completionTokens = chunk.usage.completion_tokens;
      }
    }

    const totalMs = Date.now() - t0;
    const genMs = totalMs - (ttft ?? 0);
    const tokPerSec = completionTokens > 1 ? (completionTokens - 1) / (genMs / 1000) : 0;

    return { tokPerSec, ttftMs: ttft ?? 0, totalMs, promptTokens, completionTokens, output: text };
  } else {
    const completion = await client.chat.completions.create({
      model: modelId,
      messages: scenario.messages,
      max_tokens: scenario.maxTokens,
      temperature: 0.3,
      tools: scenario.tools,
    });

    const totalMs = Date.now() - t0;
    const usage = completion.usage;
    const promptTokens = usage?.prompt_tokens ?? 0;
    const completionTokens = usage?.completion_tokens ?? 0;
    const tokPerSec = completionTokens > 0 ? completionTokens / (totalMs / 1000) : 0;
    const text = completion.choices[0]?.message?.content ?? "";
    const toolCalls = completion.choices[0]?.message?.tool_calls;
    const output = toolCalls ? JSON.stringify(toolCalls) : text;

    return { tokPerSec, ttftMs: totalMs, totalMs, promptTokens, completionTokens, output };
  }
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const singleModel = process.env.MODEL_ID;
  const models = singleModel
    ? ALL_MODELS.filter((m) => m.id === singleModel)
    : ALL_MODELS;

  if (singleModel && models.length === 0) {
    // Custom model not in the list — add it
    models.push({ id: singleModel, name: singleModel.split("/").pop()!, supportsTools: true, supportsMultiTurn: true });
  }

  console.log(`\n${"=".repeat(70)}`);
  console.log(`wandler benchmark — ${RUNS} runs/scenario, ${WARMUP_RUNS} warmup, device=${DEVICE}`);
  console.log("=".repeat(70));

  for (const modelDef of models) {
    console.log(`\n${"─".repeat(70)}`);
    console.log(`Loading: ${modelDef.name} (${modelDef.id})`);
    console.log("─".repeat(70));

    const config = loadConfig({
      MODEL_ID: modelDef.id,
      DTYPE: "q4",
      DEVICE: DEVICE,
      STT_MODEL_ID: "",
      EMBEDDING_MODEL_ID: "",
    });

    const memBefore = process.memoryUsage();
    const t0 = Date.now();
    const loadedModels = await loadModels(config);
    const loadMs = Date.now() - t0;
    const memAfter = process.memoryUsage();
    const memUsedMB = Math.round((memAfter.rss - memBefore.rss) / 1024 / 1024);
    console.log(`Loaded in ${(loadMs / 1000).toFixed(1)}s | RSS +${memUsedMB}MB (total ${Math.round(memAfter.rss / 1024 / 1024)}MB)`);

    const server = startServer({ ...config, port: 0 }, loadedModels);
    const addr = server.address() as { port: number };
    const port = addr.port;
    const client = new OpenAI({ baseURL: `http://localhost:${port}/v1`, apiKey: "-" });

    const scenarioResults: Array<{
      scenario: string;
      category: string;
      tps: ReturnType<typeof stats>;
      ttft: ReturnType<typeof stats>;
      totalMs: ReturnType<typeof stats>;
      avgPromptTok: number;
      avgCompTok: number;
      sampleOutput: string;
    }> = [];

    for (const scenario of SCENARIOS) {
      // Skip scenarios the model can't handle
      if (scenario.requiresTools && !modelDef.supportsTools) continue;
      if (scenario.requiresMultiTurn && !modelDef.supportsMultiTurn) continue;

      process.stdout.write(`  ${scenario.name}...`);

      // Warmup
      for (let i = 0; i < WARMUP_RUNS; i++) {
        try { await runScenario(client, modelDef.id, scenario); } catch { /* ignore warmup errors */ }
      }

      // Measured runs
      const runs: RunResult[] = [];
      for (let i = 0; i < RUNS; i++) {
        try {
          const result = await runScenario(client, modelDef.id, scenario);
          runs.push(result);
        } catch (e) {
          console.error(` run ${i} failed: ${(e as Error).message}`);
        }
      }

      if (runs.length === 0) {
        console.log(" SKIPPED (all runs failed)");
        continue;
      }

      const tpsStats = stats(runs.map((r) => r.tokPerSec));
      const ttftStats = stats(runs.map((r) => r.ttftMs));
      const totalStats = stats(runs.map((r) => r.totalMs));
      const avgPromptTok = Math.round(runs.reduce((s, r) => s + r.promptTokens, 0) / runs.length);
      const avgCompTok = Math.round(runs.reduce((s, r) => s + r.completionTokens, 0) / runs.length);

      scenarioResults.push({
        scenario: scenario.name,
        category: scenario.category,
        tps: tpsStats,
        ttft: ttftStats,
        totalMs: totalStats,
        avgPromptTok,
        avgCompTok,
        sampleOutput: runs[0]!.output.slice(0, 120).replace(/\n/g, "\\n"),
      });

      console.log(` ${tpsStats.avg} tok/s (p50=${tpsStats.p50}) | TTFT p50=${ttftStats.p50}ms | ${avgCompTok} tok`);
    }

    // Summary table for this model
    console.log(`\n  ${"─".repeat(66)}`);
    console.log(
      "  " +
      "Scenario".padEnd(30) +
      "tok/s".padEnd(10) +
      "p50".padEnd(8) +
      "p95".padEnd(8) +
      "TTFT p50".padEnd(10) +
      "Tokens",
    );
    console.log(`  ${"─".repeat(66)}`);
    for (const r of scenarioResults) {
      console.log(
        "  " +
        r.scenario.padEnd(30) +
        `${r.tps.avg}`.padEnd(10) +
        `${r.tps.p50}`.padEnd(8) +
        `${r.tps.p95}`.padEnd(8) +
        `${r.ttft.p50}ms`.padEnd(10) +
        `${r.avgPromptTok}p/${r.avgCompTok}c`,
      );
    }

    // Sample outputs
    console.log(`\n  Sample outputs:`);
    for (const r of scenarioResults) {
      console.log(`  [${r.scenario}] ${r.sampleOutput}`);
    }

    server.close();

    // Free model memory before loading the next one
    try { await loadedModels.model.dispose?.(); } catch { /* ignore */ }

    // Force garbage collection to actually reclaim memory between models.
    // Without this, the old model's weights stay in memory while the next loads,
    // potentially doubling memory usage and killing the system.
    if (global.gc) {
      global.gc();
      console.log("  [gc] Memory released");
    }
  }

  console.log(`\n${"=".repeat(70)}`);
  console.log("Done.");
  console.log("=".repeat(70));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
