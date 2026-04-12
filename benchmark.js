import { AutoTokenizer, AutoModelForCausalLM, TextStreamer } from "@huggingface/transformers"

const MODELS = [
  { id: "LiquidAI/LFM2.5-350M-ONNX", name: "LFM2.5-350M", dtype: "q4" },
  { id: "LiquidAI/LFM2.5-1.2B-Instruct-ONNX", name: "LFM2.5-1.2B", dtype: "q4" },
  { id: "onnx-community/Qwen3.5-0.8B-Text-ONNX", name: "Qwen3.5-0.8B", dtype: "q4" },
]

const SCENARIOS = [
  {
    name: "Quick factual Q&A",
    messages: [
      { role: "user", content: "What is the capital of Japan?" },
    ],
    maxTokens: 30,
  },
  {
    name: "System prompt + instruction",
    messages: [
      { role: "system", content: "You are a helpful coding assistant. Be concise and provide working code." },
      { role: "user", content: "Write a JavaScript function that reverses a string." },
    ],
    maxTokens: 150,
  },
  {
    name: "Multi-turn conversation",
    messages: [
      { role: "user", content: "I'm building a REST API with Node.js. What framework should I use?" },
      { role: "assistant", content: "For a REST API in Node.js, Express.js is the most popular choice due to its simplicity and large ecosystem. If you want something more modern with built-in TypeScript support, consider Fastify or Hono." },
      { role: "user", content: "I'll go with Hono. How do I set up a basic GET endpoint?" },
    ],
    maxTokens: 200,
  },
  {
    name: "Reasoning / explanation",
    messages: [
      { role: "user", content: "Explain the difference between WebSockets and Server-Sent Events. When would you use each? Give practical examples." },
    ],
    maxTokens: 300,
  },
  {
    name: "Creative writing",
    messages: [
      { role: "system", content: "You are a creative writer." },
      { role: "user", content: "Write a short story (3 paragraphs) about an AI that discovers it can dream." },
    ],
    maxTokens: 300,
  },
]

const DEVICES = ["cpu", "webgpu"]

// ── Run ─────────────────────────────────────────────────────────────────────
const results = []

for (const { id, name, dtype } of MODELS) {
  for (const device of DEVICES) {
    console.log(`\n${"=".repeat(60)}`)
    console.log(`${name} | ${device}`)
    console.log("=".repeat(60))

    const tLoad = Date.now()
    const tokenizer = await AutoTokenizer.from_pretrained(id)
    const model = await AutoModelForCausalLM.from_pretrained(id, { dtype, device })
    const loadSecs = (Date.now() - tLoad) / 1000

    // Warmup — first inference on GPU has extra overhead
    const warmupPrompt = tokenizer.apply_chat_template(
      [{ role: "user", content: "Hi" }],
      { tokenize: false, add_generation_prompt: true },
    )
    const warmupInputs = tokenizer(warmupPrompt, { return_tensors: "pt" })
    await model.generate({ ...warmupInputs, max_new_tokens: 5 })

    console.log(`Load: ${loadSecs.toFixed(1)}s (+ warmup done)\n`)

    for (const scenario of SCENARIOS) {
      const prompt = tokenizer.apply_chat_template(scenario.messages, {
        tokenize: false,
        add_generation_prompt: true,
      })
      const inputs = tokenizer(prompt, { return_tensors: "pt" })
      const promptTokens = inputs.input_ids.dims[1]

      let firstTokenMs = null
      let genTokens = 0
      const tGen = Date.now()

      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        callback_function: (text) => {
          genTokens++
          if (firstTokenMs === null) firstTokenMs = Date.now() - tGen
          process.stdout.write(text)
        },
      })

      await model.generate({
        ...inputs,
        max_new_tokens: scenario.maxTokens,
        temperature: 0.7,
        top_p: 0.95,
        do_sample: true,
      streamer,
      })

      const totalMs = Date.now() - tGen
      const ttft = firstTokenMs ?? 0
      const genMs = totalMs - ttft
      const tps = genTokens > 1 ? ((genTokens - 1) / (genMs / 1000)).toFixed(1) : "n/a"

      console.log()
      console.log(
        `  ↳ ${scenario.name}: ${promptTokens} prompt tok → ${genTokens} gen tok | ` +
        `TTFT ${ttft}ms | ${tps} tok/s | total ${(totalMs / 1000).toFixed(1)}s`
      )
      console.log()

      results.push({
        model: name,
        device,
        scenario: scenario.name,
        promptTokens,
        genTokens,
        ttftMs: ttft,
        tokPerSec: parseFloat(tps) || 0,
        totalMs,
      })
    }

    await model.dispose?.()
  }
}

// ── Summary table ───────────────────────────────────────────────────────────
console.log(`\n${"=".repeat(80)}`)
console.log("SUMMARY")
console.log("=".repeat(80))
console.log(
  "Model".padEnd(18) +
  "Device".padEnd(9) +
  "Scenario".padEnd(28) +
  "Prompt".padEnd(8) +
  "Gen".padEnd(6) +
  "TTFT".padEnd(8) +
  "tok/s".padEnd(8) +
  "Total"
)
console.log("-".repeat(80))

for (const r of results) {
  console.log(
    r.model.padEnd(18) +
    r.device.padEnd(9) +
    r.scenario.padEnd(28) +
    String(r.promptTokens).padEnd(8) +
    String(r.genTokens).padEnd(6) +
    `${r.ttftMs}ms`.padEnd(8) +
    `${r.tokPerSec}`.padEnd(8) +
    `${(r.totalMs / 1000).toFixed(1)}s`
  )
}

// Per-model averages
console.log("\n" + "-".repeat(50))
console.log("AVERAGES (tok/s across all scenarios)")
console.log("-".repeat(50))
const modelDevicePairs = [...new Set(results.map((r) => `${r.model}|${r.device}`))]
for (const pair of modelDevicePairs) {
  const [modelName, device] = pair.split("|")
  const rows = results.filter((r) => r.model === modelName && r.device === device)
  const avgTps = (rows.reduce((sum, r) => sum + r.tokPerSec, 0) / rows.length).toFixed(1)
  const avgTtft = Math.round(rows.reduce((sum, r) => sum + r.ttftMs, 0) / rows.length)
  console.log(`${modelName.padEnd(18)} ${device.padEnd(9)} avg ${avgTps} tok/s | avg TTFT ${avgTtft}ms`)
}
