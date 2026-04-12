/**
 * Smoke test: start server with a real model and make actual requests.
 * Run with: npx tsx tests/smoke.ts
 *
 * Downloads and loads a real ONNX model, then tests all endpoints.
 */
import { loadConfig } from "../src/config.js";
import { loadModels } from "../src/models/manager.js";
import { createServer } from "../src/server.js";
import OpenAI from "openai";

const MODEL_ID = process.env.MODEL_ID || "LiquidAI/LFM2.5-350M-ONNX";
const DEVICE = process.env.DEVICE || "cpu";

async function main() {
  console.log(`\n=== Smoke Test: ${MODEL_ID} (${DEVICE}) ===\n`);

  // Load config and models
  const config = loadConfig({
    MODEL_ID: MODEL_ID,
    DTYPE: "q4",
    DEVICE: DEVICE,
    STT_MODEL_ID: "", // skip STT for speed
    EMBEDDING_MODEL_ID: "",
  });

  console.log("[smoke] Loading model...");
  const t0 = Date.now();
  const models = await loadModels(config);
  const loadTime = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`[smoke] Model loaded in ${loadTime}s\n`);

  // Start server
  const server = createServer(config, models);
  await new Promise<void>((resolve) => server.listen(0, resolve));
  const port = (server.address() as { port: number }).port;
  const baseUrl = `http://localhost:${port}`;
  console.log(`[smoke] Server running at ${baseUrl}\n`);

  const client = new OpenAI({ baseURL: `${baseUrl}/v1`, apiKey: "-" });
  const results: Array<{ test: string; status: string; detail: string }> = [];

  // Test 1: Health
  try {
    const res = await fetch(`${baseUrl}/health`);
    const body = await res.json() as { status: string };
    results.push({ test: "GET /health", status: body.status === "ok" ? "PASS" : "FAIL", detail: JSON.stringify(body) });
  } catch (e) {
    results.push({ test: "GET /health", status: "FAIL", detail: (e as Error).message });
  }

  // Test 2: Models
  try {
    const models = await client.models.list();
    const list = [];
    for await (const m of models) list.push(m);
    results.push({ test: "GET /v1/models", status: list.length > 0 ? "PASS" : "FAIL", detail: `${list.length} models` });
  } catch (e) {
    results.push({ test: "GET /v1/models", status: "FAIL", detail: (e as Error).message });
  }

  // Test 3: Chat completion (non-streaming)
  try {
    const t1 = Date.now();
    const completion = await client.chat.completions.create({
      model: MODEL_ID,
      messages: [{ role: "user", content: "What is 2+2? Answer in one word." }],
      max_tokens: 20,
      temperature: 0.1,
    });
    const elapsed = Date.now() - t1;
    const text = completion.choices[0]?.message.content ?? "";
    const usage = completion.usage;
    const tps = usage ? (usage.completion_tokens / (elapsed / 1000)).toFixed(1) : "?";
    results.push({
      test: "POST /v1/chat/completions",
      status: text.length > 0 ? "PASS" : "FAIL",
      detail: `"${text.slice(0, 80)}" | ${usage?.prompt_tokens}p/${usage?.completion_tokens}c tok | ${tps} tok/s | ${elapsed}ms`,
    });
  } catch (e) {
    results.push({ test: "POST /v1/chat/completions", status: "FAIL", detail: (e as Error).message });
  }

  // Test 4: Chat completion (streaming)
  try {
    const t2 = Date.now();
    const stream = await client.chat.completions.create({
      model: MODEL_ID,
      messages: [{ role: "user", content: "Say hello in French." }],
      max_tokens: 20,
      temperature: 0.1,
      stream: true,
    });
    let text = "";
    let ttft: number | null = null;
    for await (const chunk of stream) {
      if (ttft === null) ttft = Date.now() - t2;
      text += chunk.choices[0]?.delta?.content ?? "";
    }
    const elapsed = Date.now() - t2;
    results.push({
      test: "POST /v1/chat/completions (stream)",
      status: text.length > 0 ? "PASS" : "FAIL",
      detail: `"${text.slice(0, 80)}" | TTFT ${ttft}ms | total ${elapsed}ms`,
    });
  } catch (e) {
    results.push({ test: "POST /v1/chat/completions (stream)", status: "FAIL", detail: (e as Error).message });
  }

  // Test 5: Text completion
  try {
    const completion = await client.completions.create({
      model: MODEL_ID,
      prompt: "The capital of France is",
      max_tokens: 10,
      temperature: 0.1,
    });
    const text = completion.choices[0]?.text ?? "";
    results.push({
      test: "POST /v1/completions",
      status: text.length > 0 ? "PASS" : "FAIL",
      detail: `"${text.slice(0, 80)}"`,
    });
  } catch (e) {
    results.push({ test: "POST /v1/completions", status: "FAIL", detail: (e as Error).message });
  }

  // Test 6: Tokenize
  try {
    const res = await fetch(`${baseUrl}/tokenize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: "Hello world" }),
    });
    const body = await res.json() as { count: number };
    results.push({ test: "POST /tokenize", status: body.count > 0 ? "PASS" : "FAIL", detail: `${body.count} tokens` });
  } catch (e) {
    results.push({ test: "POST /tokenize", status: "FAIL", detail: (e as Error).message });
  }

  // Print results
  console.log("\n" + "=".repeat(70));
  console.log(`RESULTS: ${MODEL_ID} (${DEVICE})`);
  console.log("=".repeat(70));
  let passed = 0;
  let failed = 0;
  for (const r of results) {
    const icon = r.status === "PASS" ? "✓" : "✗";
    console.log(`  ${icon} ${r.test}`);
    console.log(`    ${r.detail}`);
    if (r.status === "PASS") passed++;
    else failed++;
  }
  console.log(`\n${passed} passed, ${failed} failed\n`);

  server.close();
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
