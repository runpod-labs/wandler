/**
 * Agent-focused benchmark suite for wandler.
 *
 * Supports:
 * - local in-process server mode (default)
 * - external server mode via BASE_URL
 * - Hermes-like long-context and tool-heavy scenarios
 * - parallel client fan-out
 * - machine-readable JSON output
 *
 * Usage examples:
 *   npm run benchmark
 *   PROFILE=agent DEVICE=cuda RUNS=5 npm run benchmark
 *   BASE_URL=http://127.0.0.1:8000 PROFILE=agent PARALLEL=1,2,4 OUTPUT_JSON=bench.json npm run benchmark
 */
import { writeFileSync } from "node:fs";
import OpenAI from "openai";
import { loadConfig } from "./config.js";
import { loadModels } from "./models/manager.js";
import { startServer } from "./server.js";

// ── Config ──────────────────────────────────────────────────────────────────

const RUNS = parseInt(process.env.RUNS || "5", 10);
const WARMUP_RUNS = parseInt(process.env.WARMUP_RUNS || "2", 10);
const DEVICE = process.env.DEVICE || "webgpu";
const PROFILE = process.env.PROFILE || "agent";
const DTYPE = process.env.DTYPE || "q4";
const OUTPUT_JSON = process.env.OUTPUT_JSON || "";
const BASE_URL = process.env.BASE_URL || "";
const API_KEY = process.env.API_KEY || "-";
const MAX_TOKENS = parseInt(process.env.MAX_TOKENS || "128", 10);
const PARALLEL_LEVELS = parseNumberList(process.env.PARALLEL || "1,2,4");
const CONTEXT_TURNS = parseNumberList(process.env.CONTEXT_TURNS || "8,16");

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

type ChatMessage = OpenAI.ChatCompletionMessageParam;
type ChatTool = OpenAI.ChatCompletionTool;

interface Scenario {
  name: string;
  category: string;
  messages: ChatMessage[];
  maxTokens: number;
  stream: boolean;
  tools?: ChatTool[];
  parallelism: number;
  contextTurns: number;
  requiresTools?: boolean;
  requiresMultiTurn?: boolean;
}

const AGENT_TOOLS: ChatTool[] = [
  {
    type: "function",
    function: {
      name: "search_docs",
      description: "Search local project documentation for a query",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          limit: { type: "integer", description: "Maximum number of results" },
        },
        required: ["query"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read a file from the workspace",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Workspace-relative file path" },
        },
        required: ["path"],
      },
    },
  },
];

function buildConversation(turns: number): ChatMessage[] {
  const messages: ChatMessage[] = [
    {
      role: "system",
      content: "You are a coding agent. Be concise, use tools when useful, and keep state across turns.",
    },
  ];

  for (let i = 0; i < turns; i++) {
    messages.push({
      role: "user",
      content: `Turn ${i + 1}: inspect the codebase area for issue ${i + 1} and summarize the next step.`,
    });
    if (i < turns - 1) {
      messages.push({
        role: "assistant",
        content: `Summary ${i + 1}: inspected module ${i + 1}, found a likely bottleneck, and will continue with the next file.`,
      });
    }
  }

  return messages;
}

function buildScenarios(): Scenario[] {
  const scenarios: Scenario[] = [];
  const shortMessages = buildConversation(1);

  if (PROFILE === "standard" || PROFILE === "all") {
    scenarios.push(
      {
        name: "baseline-short",
        category: "baseline",
        messages: [{ role: "user", content: "Say hello in one sentence." }],
        maxTokens: 24,
        stream: false,
        parallelism: 1,
        contextTurns: 1,
      },
      {
        name: "baseline-stream",
        category: "baseline",
        messages: [{ role: "user", content: "Explain what a CPU does in one paragraph." }],
        maxTokens: 96,
        stream: true,
        parallelism: 1,
        contextTurns: 1,
      },
    );
  }

  if (PROFILE === "agent" || PROFILE === "all") {
    scenarios.push(
      {
        name: "agent-direct-short",
        category: "agent",
        messages: shortMessages,
        maxTokens: 64,
        stream: false,
        parallelism: 1,
        contextTurns: 1,
      },
      {
        name: "agent-direct-stream",
        category: "agent",
        messages: shortMessages,
        maxTokens: 64,
        stream: true,
        parallelism: 1,
        contextTurns: 1,
      },
      {
        name: "agent-tools-short",
        category: "tools",
        messages: shortMessages,
        maxTokens: 64,
        stream: false,
        tools: AGENT_TOOLS,
        parallelism: 1,
        contextTurns: 1,
        requiresTools: true,
      },
      {
        name: "agent-tools-stream",
        category: "tools",
        messages: shortMessages,
        maxTokens: 64,
        stream: true,
        tools: AGENT_TOOLS,
        parallelism: 1,
        contextTurns: 1,
        requiresTools: true,
      },
    );

    for (const turns of CONTEXT_TURNS) {
      scenarios.push(
        {
          name: `agent-context-${turns}`,
          category: "context",
          messages: buildConversation(turns),
          maxTokens: MAX_TOKENS,
          stream: false,
          parallelism: 1,
          contextTurns: turns,
          requiresMultiTurn: true,
        },
        {
          name: `agent-tools-context-${turns}`,
          category: "context-tools",
          messages: buildConversation(turns),
          maxTokens: MAX_TOKENS,
          stream: false,
          tools: AGENT_TOOLS,
          parallelism: 1,
          contextTurns: turns,
          requiresTools: true,
          requiresMultiTurn: true,
        },
      );
    }

    const heaviestTurns = CONTEXT_TURNS[CONTEXT_TURNS.length - 1] ?? 16;
    for (const parallelism of PARALLEL_LEVELS.filter((level) => level > 1)) {
      scenarios.push(
        {
          name: `parallel-direct-x${parallelism}`,
          category: "parallel",
          messages: shortMessages,
          maxTokens: 64,
          stream: false,
          parallelism,
          contextTurns: 1,
        },
        {
          name: `parallel-tools-x${parallelism}`,
          category: "parallel-tools",
          messages: shortMessages,
          maxTokens: 64,
          stream: false,
          tools: AGENT_TOOLS,
          parallelism,
          contextTurns: 1,
          requiresTools: true,
        },
        {
          name: `parallel-tools-context-${heaviestTurns}-x${parallelism}`,
          category: "parallel-tools-context",
          messages: buildConversation(heaviestTurns),
          maxTokens: MAX_TOKENS,
          stream: false,
          tools: AGENT_TOOLS,
          parallelism,
          contextTurns: heaviestTurns,
          requiresTools: true,
          requiresMultiTurn: true,
        },
      );
    }
  }

  return scenarios;
}

const SCENARIOS = buildScenarios();

// ── Stats ───────────────────────────────────────────────────────────────────

interface SingleRequestResult {
  ttftMs: number;
  totalMs: number;
  promptTokens: number;
  completionTokens: number;
  output: string;
}

interface RunResult {
  tokPerSec: number;
  ttftMs: number;
  totalMs: number;
  promptTokens: number;
  completionTokens: number;
  sampleOutput: string;
}

interface MetricsSnapshot {
  total_requests: number;
  total_prompt_tokens: number;
  total_tokens_generated: number;
  streamed_requests: number;
  tool_requests: number;
  average_latency_ms: number;
  active_requests?: number;
  peak_active_requests?: number;
}

function percentile(sorted: number[], p: number): number {
  const idx = Math.ceil((p / 100) * sorted.length) - 1;
  return sorted[Math.max(0, idx)]!;
}

function stats(values: number[]): { avg: number; min: number; max: number; p50: number; p95: number } {
  const sorted = [...values].sort((a, b) => a - b);
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    avg: round1(avg),
    min: round1(sorted[0]!),
    max: round1(sorted[sorted.length - 1]!),
    p50: round1(percentile(sorted, 50)),
    p95: round1(percentile(sorted, 95)),
  };
}

function round1(value: number): number {
  return Math.round(value * 10) / 10;
}

function parseNumberList(value: string): number[] {
  return value
    .split(",")
    .map((part) => parseInt(part.trim(), 10))
    .filter((part) => Number.isFinite(part) && part > 0);
}

function normalizeBaseUrl(baseUrl: string): { clientBaseUrl: string; rootUrl: string } {
  const trimmed = baseUrl.replace(/\/+$/, "");
  if (trimmed.endsWith("/v1")) {
    return {
      clientBaseUrl: trimmed,
      rootUrl: trimmed.slice(0, -3),
    };
  }
  return {
    clientBaseUrl: `${trimmed}/v1`,
    rootUrl: trimmed,
  };
}

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    headers: API_KEY ? { Authorization: `Bearer ${API_KEY}` } : undefined,
  });
  if (!res.ok) {
    throw new Error(`Request failed (${res.status}) for ${url}`);
  }
  return await res.json() as T;
}

async function fetchMetrics(rootUrl: string): Promise<MetricsSnapshot | null> {
  try {
    return await fetchJson<MetricsSnapshot>(`${rootUrl}/admin/metrics`);
  } catch {
    return null;
  }
}

function diffMetrics(
  before: MetricsSnapshot | null,
  after: MetricsSnapshot | null,
): MetricsSnapshot | null {
  if (!before || !after) return null;
  return {
    total_requests: after.total_requests - before.total_requests,
    total_prompt_tokens: after.total_prompt_tokens - before.total_prompt_tokens,
    total_tokens_generated: after.total_tokens_generated - before.total_tokens_generated,
    streamed_requests: after.streamed_requests - before.streamed_requests,
    tool_requests: after.tool_requests - before.tool_requests,
    average_latency_ms: after.average_latency_ms,
    active_requests: after.active_requests,
    peak_active_requests: after.peak_active_requests,
  };
}

async function waitForServerPort(
  server: ReturnType<typeof startServer>,
): Promise<number> {
  for (let attempt = 0; attempt < 20; attempt++) {
    const address = server.address() as { port?: number } | null;
    if (address?.port) return address.port;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("Server did not expose a bound port in time");
}

// ── Runner ──────────────────────────────────────────────────────────────────

async function runSingleRequest(
  client: OpenAI,
  modelId: string,
  scenario: Scenario,
): Promise<SingleRequestResult> {
  const t0 = Date.now();

  if (scenario.stream) {
    const stream = await client.chat.completions.create({
      model: modelId,
      messages: scenario.messages,
      max_tokens: scenario.maxTokens,
      temperature: 0,
      stream: true,
      tools: scenario.tools,
    });

    let text = "";
    let ttftMs: number | null = null;
    let completionTokens = 0;
    let promptTokens = 0;

    for await (const chunk of stream) {
      if (ttftMs === null) ttftMs = Date.now() - t0;
      const content = chunk.choices[0]?.delta?.content ?? "";
      text += content;
      if (content) completionTokens++;
      if (chunk.usage) {
        promptTokens = chunk.usage.prompt_tokens;
        completionTokens = chunk.usage.completion_tokens;
      }
    }

    return {
      ttftMs: ttftMs ?? 0,
      totalMs: Date.now() - t0,
      promptTokens,
      completionTokens,
      output: text,
    };
  }

  const completion = await client.chat.completions.create({
    model: modelId,
    messages: scenario.messages,
    max_tokens: scenario.maxTokens,
    temperature: 0,
    tools: scenario.tools,
  });

  const totalMs = Date.now() - t0;
  const usage = completion.usage;
  const promptTokens = usage?.prompt_tokens ?? 0;
  const completionTokens = usage?.completion_tokens ?? 0;
  const text = completion.choices[0]?.message?.content ?? "";
  const toolCalls = completion.choices[0]?.message?.tool_calls;

  return {
    ttftMs: totalMs,
    totalMs,
    promptTokens,
    completionTokens,
    output: toolCalls ? JSON.stringify(toolCalls) : text,
  };
}

async function runScenario(
  client: OpenAI,
  modelId: string,
  scenario: Scenario,
): Promise<RunResult> {
  const startedAt = Date.now();
  const requests = Array.from(
    { length: scenario.parallelism },
    () => runSingleRequest(client, modelId, scenario),
  );
  const results = await Promise.all(requests);
  const wallMs = Date.now() - startedAt;

  const totalPromptTokens = results.reduce((sum, result) => sum + result.promptTokens, 0);
  const totalCompletionTokens = results.reduce((sum, result) => sum + result.completionTokens, 0);
  const avgTtftMs = results.reduce((sum, result) => sum + result.ttftMs, 0) / results.length;
  const throughputTokPerSec = totalCompletionTokens > 0 ? totalCompletionTokens / (wallMs / 1000) : 0;

  return {
    tokPerSec: round1(throughputTokPerSec),
    ttftMs: round1(avgTtftMs),
    totalMs: wallMs,
    promptTokens: totalPromptTokens,
    completionTokens: totalCompletionTokens,
    sampleOutput: results[0]?.output ?? "",
  };
}

// ── Main ────────────────────────────────────────────────────────────────────

async function main() {
  const report: {
    generatedAt: string;
    device: string;
    dtype: string;
    profile: string;
    runs: number;
    warmupRuns: number;
    scenarios: string[];
    mode: "external" | "internal";
    results: unknown[];
  } = {
    generatedAt: new Date().toISOString(),
    device: DEVICE,
    dtype: DTYPE,
    profile: PROFILE,
    runs: RUNS,
    warmupRuns: WARMUP_RUNS,
    scenarios: SCENARIOS.map((scenario) => scenario.name),
    mode: BASE_URL ? "external" : "internal",
    results: [],
  };

  console.log(`\n${"=".repeat(78)}`);
  console.log(`wandler benchmark — profile=${PROFILE}, runs=${RUNS}, warmup=${WARMUP_RUNS}, device=${DEVICE}`);
  console.log(`parallel=${PARALLEL_LEVELS.join(",")} | context_turns=${CONTEXT_TURNS.join(",")} | dtype=${DTYPE}`);
  if (BASE_URL) console.log(`target=${BASE_URL}`);
  console.log("=".repeat(78));

  if (BASE_URL) {
    const { clientBaseUrl, rootUrl } = normalizeBaseUrl(BASE_URL);
    const modelsResponse = await fetchJson<{ data: Array<{ id: string }> }>(`${clientBaseUrl}/models`);
    const externalModelId = process.env.MODEL_ID || modelsResponse.data[0]?.id;
    if (!externalModelId) {
      throw new Error(`No model ID available from ${clientBaseUrl}/models`);
    }

    const client = new OpenAI({ baseURL: clientBaseUrl, apiKey: API_KEY });
    const metricsBefore = await fetchMetrics(rootUrl);
    const scenarioResults = await runScenarioMatrix(client, externalModelId);
    const metricsAfter = await fetchMetrics(rootUrl);
    const metricsDelta = diffMetrics(metricsBefore, metricsAfter);

    report.results.push({
      modelId: externalModelId,
      name: externalModelId.split("/").pop() ?? externalModelId,
      mode: "external",
      baseUrl: clientBaseUrl,
      metricsDelta,
      scenarios: scenarioResults,
    });
  } else {
    const singleModel = process.env.MODEL_ID;
    const selectedModels = singleModel
      ? ALL_MODELS.filter((model) => model.id === singleModel)
      : ALL_MODELS;

    if (singleModel && selectedModels.length === 0) {
      selectedModels.push({
        id: singleModel,
        name: singleModel.split("/").pop() ?? singleModel,
        supportsTools: true,
        supportsMultiTurn: true,
      });
    }

    for (const modelDef of selectedModels) {
      console.log(`\n${"─".repeat(78)}`);
      console.log(`Loading: ${modelDef.name} (${modelDef.id})`);
      console.log("─".repeat(78));

      const config = loadConfig({
        MODEL_ID: modelDef.id,
        DTYPE,
        DEVICE,
        STT_MODEL_ID: "",
        EMBEDDING_MODEL_ID: "",
      });

      const memBefore = process.memoryUsage();
      const loadStartedAt = Date.now();
      const loadedModels = await loadModels(config);
      const loadMs = Date.now() - loadStartedAt;
      const memAfter = process.memoryUsage();
      const memUsedMB = Math.round((memAfter.rss - memBefore.rss) / 1024 / 1024);
      console.log(`Loaded in ${(loadMs / 1000).toFixed(1)}s | RSS +${memUsedMB}MB (total ${Math.round(memAfter.rss / 1024 / 1024)}MB)`);

      const server = startServer({ ...config, port: 0 }, loadedModels);
      const port = await waitForServerPort(server);
      const rootUrl = `http://localhost:${port}`;
      const clientBaseUrl = `${rootUrl}/v1`;
      const client = new OpenAI({ baseURL: clientBaseUrl, apiKey: API_KEY });

      const metricsBefore = await fetchMetrics(rootUrl);
      const scenarioResults = await runScenarioMatrix(client, modelDef.id, modelDef);
      const metricsAfter = await fetchMetrics(rootUrl);
      const metricsDelta = diffMetrics(metricsBefore, metricsAfter);

      report.results.push({
        modelId: modelDef.id,
        name: modelDef.name,
        mode: "internal",
        loadMs,
        memUsedMB,
        metricsDelta,
        scenarios: scenarioResults,
      });

      await new Promise<void>((resolve) => server.close(() => resolve()));
      try {
        await loadedModels.model?.dispose?.();
      } catch {
        // Ignore dispose errors during benchmarking cleanup.
      }

      if (global.gc) {
        global.gc();
        console.log("  [gc] Memory released");
      }
    }
  }

  if (OUTPUT_JSON) {
    writeFileSync(OUTPUT_JSON, JSON.stringify(report, null, 2));
    console.log(`\nWrote benchmark JSON to ${OUTPUT_JSON}`);
  }

  console.log(`\n${"=".repeat(78)}`);
  console.log("Done.");
  console.log("=".repeat(78));
}

async function runScenarioMatrix(
  client: OpenAI,
  modelId: string,
  modelDef?: ModelDef,
) {
  const scenarioResults: Array<{
    scenario: string;
    category: string;
    parallelism: number;
    contextTurns: number;
    stream: boolean;
    hasTools: boolean;
    throughput: ReturnType<typeof stats>;
    ttft: ReturnType<typeof stats>;
    wallMs: ReturnType<typeof stats>;
    avgPromptTokPerRequest: number;
    avgCompTokPerRequest: number;
    sampleOutput: string;
  }> = [];

  for (const scenario of SCENARIOS) {
    if (scenario.requiresTools && modelDef && !modelDef.supportsTools) continue;
    if (scenario.requiresMultiTurn && modelDef && !modelDef.supportsMultiTurn) continue;

    process.stdout.write(`  ${scenario.name}...`);

    for (let i = 0; i < WARMUP_RUNS; i++) {
      try {
        await runScenario(client, modelId, scenario);
      } catch {
        // Ignore warmup failures.
      }
    }

    const runs: RunResult[] = [];
    for (let i = 0; i < RUNS; i++) {
      try {
        runs.push(await runScenario(client, modelId, scenario));
      } catch (error) {
        console.error(` run ${i + 1} failed: ${(error as Error).message}`);
      }
    }

    if (runs.length === 0) {
      console.log(" SKIPPED");
      continue;
    }

    const throughputStats = stats(runs.map((run) => run.tokPerSec));
    const ttftStats = stats(runs.map((run) => run.ttftMs));
    const wallStats = stats(runs.map((run) => run.totalMs));
    const avgPromptTokPerRequest = Math.round(
      runs.reduce((sum, run) => sum + run.promptTokens, 0) / runs.length / scenario.parallelism,
    );
    const avgCompTokPerRequest = Math.round(
      runs.reduce((sum, run) => sum + run.completionTokens, 0) / runs.length / scenario.parallelism,
    );

    scenarioResults.push({
      scenario: scenario.name,
      category: scenario.category,
      parallelism: scenario.parallelism,
      contextTurns: scenario.contextTurns,
      stream: scenario.stream,
      hasTools: Boolean(scenario.tools?.length),
      throughput: throughputStats,
      ttft: ttftStats,
      wallMs: wallStats,
      avgPromptTokPerRequest,
      avgCompTokPerRequest,
      sampleOutput: runs[0]!.sampleOutput.slice(0, 120).replace(/\n/g, "\\n"),
    });

    console.log(
      ` ${throughputStats.avg} tok/s | wall p50=${wallStats.p50}ms | TTFT p50=${ttftStats.p50}ms | ${avgPromptTokPerRequest}p/${avgCompTokPerRequest}c per req`,
    );
  }

  console.log(`\n  ${"─".repeat(74)}`);
  console.log(
    "  " +
    "Scenario".padEnd(32) +
    "tok/s".padEnd(10) +
    "wall p50".padEnd(10) +
    "TTFT p50".padEnd(10) +
    "parallel".padEnd(10) +
    "tokens",
  );
  console.log(`  ${"─".repeat(74)}`);
  for (const result of scenarioResults) {
    console.log(
      "  " +
      result.scenario.padEnd(32) +
      `${result.throughput.avg}`.padEnd(10) +
      `${result.wallMs.p50}ms`.padEnd(10) +
      `${result.ttft.p50}ms`.padEnd(10) +
      `${result.parallelism}`.padEnd(10) +
      `${result.avgPromptTokPerRequest}p/${result.avgCompTokPerRequest}c`,
    );
  }

  console.log(`\n  Sample outputs:`);
  for (const result of scenarioResults) {
    console.log(`  [${result.scenario}] ${result.sampleOutput}`);
  }

  return scenarioResults;
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
