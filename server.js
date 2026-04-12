import http from "node:http"
import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  pipeline,
} from "@huggingface/transformers"

// ── Config ──────────────────────────────────────────────────────────────────
const PORT = parseInt(process.env.PORT || "8000", 10)
const MODEL_ID = process.env.MODEL_ID || "onnx-community/gemma-4-E4B-it-ONNX"
const MODEL_DTYPE = process.env.DTYPE || "q4"
const DEVICE = process.env.DEVICE || "webgpu"
const STT_MODEL_ID = process.env.STT_MODEL_ID || "onnx-community/whisper-tiny"
const STT_DTYPE = process.env.STT_DTYPE || "q4"

// ── Gemma chat template (not included in ONNX tokenizer config) ─────────
function formatGemmaChat(messages) {
  let prompt = ""
  for (const msg of messages) {
    if (msg.role === "system") {
      // Gemma doesn't have a system role — prepend to first user message
      prompt += `<start_of_turn>user\n${msg.content}\n`
    } else if (msg.role === "user") {
      // If last char isn't a newline after system, we already started user turn
      if (prompt.endsWith(`\n`)) {
        prompt += msg.content + `<end_of_turn>\n`
      } else {
        prompt += `<start_of_turn>user\n${msg.content}<end_of_turn>\n`
      }
    } else if (msg.role === "assistant") {
      prompt += `<start_of_turn>model\n${msg.content}<end_of_turn>\n`
    }
  }
  prompt += `<start_of_turn>model\n`
  return prompt
}

// ── Generic chat template with fallback ─────────────────────────────────
function formatChat(tokenizer, messages, modelId, tools) {
  // Try built-in template first — pass tools if available
  try {
    const opts = {
      tokenize: false,
      add_generation_prompt: true,
    }
    if (tools?.length) {
      opts.tools = tools
    }
    return tokenizer.apply_chat_template(messages, opts)
  } catch {
    // Fallback for models without chat template (e.g. Gemma ONNX exports)
    if (modelId.toLowerCase().includes("gemma")) {
      return formatGemmaChat(messages)
    }
    // Generic fallback
    return messages.map((m) =>
      m.role === "assistant" ? `Assistant: ${m.content}` : `User: ${m.content}`
    ).join("\n") + "\nAssistant: "
  }
}

// ── Load models ─────────────────────────────────────────────────────────────
console.log(`[a2go-tjs] Loading LLM: ${MODEL_ID} (${MODEL_DTYPE}, ${DEVICE})`)
const t0 = Date.now()
const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID)
const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, { dtype: MODEL_DTYPE, device: DEVICE })
console.log(`[a2go-tjs] LLM ready in ${((Date.now() - t0) / 1000).toFixed(1)}s`)

console.log(`[a2go-tjs] Loading STT: ${STT_MODEL_ID} (${STT_DTYPE})`)
const t1 = Date.now()
const transcriber = await pipeline("automatic-speech-recognition", STT_MODEL_ID, { dtype: STT_DTYPE })
console.log(`[a2go-tjs] STT ready in ${((Date.now() - t1) / 1000).toFixed(1)}s`)

// ── Helpers ─────────────────────────────────────────────────────────────────
function makeId(prefix = "chatcmpl") {
  return `${prefix}-${Math.random().toString(36).slice(2, 14)}`
}

function json(res, status, body) {
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
  })
  res.end(JSON.stringify(body))
}

async function readBody(req) {
  const chunks = []
  for await (const chunk of req) chunks.push(chunk)
  return Buffer.concat(chunks)
}

// ── Tool call detection — parse model output into OpenAI format ─────────────
function parseToolCalls(text) {
  // Strip thinking blocks (Qwen outputs <think>...</think> before tool calls)
  const cleaned = text.replace(/<think>[\s\S]*?<\/think>/g, "").trim()

  // Strategy 1: LFM Pythonic format — <|tool_call_start|>[func_name(arg="val")]<|tool_call_end|>
  const lfmPythonic = cleaned.match(/\[(\w+)\(([^)]*)\)\]/)
  if (lfmPythonic) {
    const name = lfmPythonic[1]
    const argsStr = lfmPythonic[2]
    // Parse Python-style kwargs: key="value", key2="value2"
    const args = {}
    const kwargPattern = /(\w+)\s*=\s*"([^"]*)"/g
    let m
    while ((m = kwargPattern.exec(argsStr)) !== null) {
      args[m[1]] = m[2]
    }
    return [{
      id: `call_${Math.random().toString(36).slice(2, 10)}`,
      type: "function",
      function: { name, arguments: JSON.stringify(args) },
    }]
  }

  // Strategy 1b: LFM JSON format — [tool_calls [{...}]]
  const lfmJson = cleaned.match(/\[tool_calls\s*([\s\S]*?)(?:\]\s*$|\<\|tool_call_end\|>)/)
  if (lfmJson) {
    let inner = lfmJson[1].trim()
    const arrStart = inner.indexOf("[")
    if (arrStart >= 0) {
      let depth = 0, arrEnd = -1
      for (let i = arrStart; i < inner.length; i++) {
        if (inner[i] === "[" || inner[i] === "{") depth++
        if (inner[i] === "]" || inner[i] === "}") depth--
        if (depth === 0) { arrEnd = i; break }
      }
      if (arrEnd > arrStart) {
        try {
          const calls = JSON.parse(inner.substring(arrStart, arrEnd + 1))
          if (Array.isArray(calls)) {
            return calls.map((tc) => ({
              id: `call_${Math.random().toString(36).slice(2, 10)}`,
              type: "function",
              function: {
                name: tc.name,
                arguments: typeof tc.arguments === "string" ? tc.arguments : JSON.stringify(tc.arguments ?? {}),
              },
            }))
          }
        } catch {}
      }
    }
  }

  // Strategy 2: Qwen format — <tool_call>{"name": "...", "arguments": {...}}</tool_call>
  const qwenMatch = cleaned.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/)
  if (qwenMatch) {
    try {
      const call = JSON.parse(qwenMatch[1])
      return [{
        id: `call_${Math.random().toString(36).slice(2, 10)}`,
        type: "function",
        function: {
          name: call.name,
          arguments: typeof call.arguments === "string" ? call.arguments : JSON.stringify(call.arguments ?? {}),
        },
      }]
    } catch {}
  }

  // Strategy 3: OpenAI-style JSON — {"tool_calls": [...]}
  const jsonMatch = cleaned.match(/\{[\s\S]*"tool_calls"[\s\S]*\}/)
  if (!jsonMatch) return null

  try {
    const parsed = JSON.parse(jsonMatch[0])
    if (!Array.isArray(parsed.tool_calls)) return null

    return parsed.tool_calls.map((tc) => ({
      id: `call_${Math.random().toString(36).slice(2, 10)}`,
      type: "function",
      function: {
        name: tc.function?.name ?? tc.name,
        arguments: typeof tc.function?.arguments === "string"
          ? tc.function.arguments
          : JSON.stringify(tc.function?.arguments ?? tc.arguments ?? {}),
      },
    }))
  } catch {
    return null
  }
}

// ── LLM: Generate (non-streaming) ──────────────────────────────────────────
async function generate(messages, genOpts, tools) {
  const prompt = formatChat(tokenizer, messages, MODEL_ID, tools)
  const inputs = tokenizer(prompt, { return_tensors: "pt" })
  const outputIds = await model.generate({ ...inputs, ...genOpts })

  const promptTokens = inputs.input_ids.dims[1]
  const completionTokens = outputIds.dims[1] - promptTokens
  const newIds = outputIds.slice(null, [promptTokens, null])
  const text = tokenizer.batch_decode(newIds, { skip_special_tokens: true })[0]
  return { text, promptTokens, completionTokens }
}

// ── LLM: Generate (streaming) ──────────────────────────────────────────────
async function generateStream(res, messages, genOpts, id, created, tools) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "Access-Control-Allow-Origin": "*",
  })

  const sse = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`)

  sse({
    id, object: "chat.completion.chunk", created, model: MODEL_ID,
    choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
  })

  const prompt = formatChat(tokenizer, messages, MODEL_ID, tools)
  const inputs = tokenizer(prompt, { return_tensors: "pt" })
  const promptTokens = inputs.input_ids.dims[1]
  let completionTokens = 0

  const streamer = new TextStreamer(tokenizer, {
    skip_prompt: true,
    callback_function: (token) => {
      completionTokens++
      sse({
        id, object: "chat.completion.chunk", created, model: MODEL_ID,
        choices: [{ index: 0, delta: { content: token }, finish_reason: null }],
      })
    },
  })

  await model.generate({ ...inputs, ...genOpts, streamer })

  // Final chunk with usage info (OpenAI stream_options pattern)
  sse({
    id, object: "chat.completion.chunk", created, model: MODEL_ID,
    choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
    usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
  })
  res.write("data: [DONE]\n\n")
  res.end()
}

// ── STT: Parse multipart form data (minimal) ──────────────────────────────
function parseMultipart(buffer, boundary) {
  const parts = {}
  const boundaryBuf = Buffer.from(`--${boundary}`)
  let start = buffer.indexOf(boundaryBuf) + boundaryBuf.length

  while (start < buffer.length) {
    const nextBoundary = buffer.indexOf(boundaryBuf, start)
    if (nextBoundary === -1) break

    const part = buffer.subarray(start, nextBoundary)
    const headerEnd = part.indexOf("\r\n\r\n")
    if (headerEnd === -1) { start = nextBoundary + boundaryBuf.length; continue }

    const headers = part.subarray(0, headerEnd).toString()
    const body = part.subarray(headerEnd + 4, part.length - 2)

    const nameMatch = headers.match(/name="([^"]+)"/)
    if (nameMatch) {
      const name = nameMatch[1]
      const filenameMatch = headers.match(/filename="([^"]+)"/)
      parts[name] = filenameMatch ? { filename: filenameMatch[1], data: body } : body.toString()
    }

    start = nextBoundary + boundaryBuf.length
  }
  return parts
}

// ── HTTP Server ─────────────────────────────────────────────────────────────
const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type, Authorization",
    })
    return res.end()
  }

  if (req.method === "GET" && req.url === "/v1/models") {
    return json(res, 200, {
      object: "list",
      data: [
        { id: MODEL_ID, object: "model", created: Math.floor(Date.now() / 1000), owned_by: "a2go-transformers-js" },
        { id: STT_MODEL_ID, object: "model", created: Math.floor(Date.now() / 1000), owned_by: "a2go-transformers-js" },
      ],
    })
  }

  if (req.method === "POST" && req.url === "/v1/chat/completions") {
    let params
    try {
      params = JSON.parse((await readBody(req)).toString())
    } catch {
      return json(res, 400, { error: { message: "Invalid JSON" } })
    }

    const { messages, stream, max_tokens, temperature, top_p, tools, tool_choice } = params
    if (!messages?.length) {
      return json(res, 400, { error: { message: "messages is required" } })
    }

    const id = makeId()
    const created = Math.floor(Date.now() / 1000)
    const genOpts = {
      max_new_tokens: max_tokens || 2048,
      temperature: temperature ?? 0.7,
      top_p: top_p ?? 0.95,
      do_sample: (temperature ?? 0.7) > 0,
    }

    try {
      if (stream && !tools?.length) {
        return await generateStream(res, messages, genOpts, id, created, tools)
      }

      // Generate full text first (needed for tool call parsing)
      const result = await generate(messages, genOpts, tools)
      const toolCalls = tools?.length ? parseToolCalls(result.text) : null

      // If caller requested streaming, wrap the result as SSE
      if (stream) {
        res.writeHead(200, {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
          "Access-Control-Allow-Origin": "*",
        })
        const sse = (data) => res.write(`data: ${JSON.stringify(data)}\n\n`)

        if (toolCalls) {
          // Emit tool calls as a single chunk
          sse({
            id, object: "chat.completion.chunk", created, model: MODEL_ID,
            choices: [{ index: 0, delta: { role: "assistant", tool_calls: toolCalls }, finish_reason: null }],
          })
          sse({
            id, object: "chat.completion.chunk", created, model: MODEL_ID,
            choices: [{ index: 0, delta: {}, finish_reason: "tool_calls" }],
            usage: { prompt_tokens: result.promptTokens, completion_tokens: result.completionTokens, total_tokens: result.promptTokens + result.completionTokens },
          })
        } else {
          // Emit text content as chunks
          sse({
            id, object: "chat.completion.chunk", created, model: MODEL_ID,
            choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }],
          })
          sse({
            id, object: "chat.completion.chunk", created, model: MODEL_ID,
            choices: [{ index: 0, delta: { content: result.text }, finish_reason: null }],
          })
          sse({
            id, object: "chat.completion.chunk", created, model: MODEL_ID,
            choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
            usage: { prompt_tokens: result.promptTokens, completion_tokens: result.completionTokens, total_tokens: result.promptTokens + result.completionTokens },
          })
        }
        res.write("data: [DONE]\n\n")
        return res.end()
      }

      // Non-streaming response
      const message = toolCalls
        ? { role: "assistant", content: null, tool_calls: toolCalls }
        : { role: "assistant", content: result.text }

      return json(res, 200, {
        id, object: "chat.completion", created, model: MODEL_ID,
        choices: [{
          index: 0,
          message,
          finish_reason: toolCalls ? "tool_calls" : "stop",
        }],
        usage: {
          prompt_tokens: result.promptTokens,
          completion_tokens: result.completionTokens,
          total_tokens: result.promptTokens + result.completionTokens,
        },
      })
    } catch (e) {
      console.error("[a2go-tjs] LLM error:", e)
      return json(res, 500, { error: { message: e.message } })
    }
  }

  if (req.method === "POST" && req.url === "/v1/audio/transcriptions") {
    try {
      const body = await readBody(req)
      const contentType = req.headers["content-type"] || ""

      let audioBuffer
      if (contentType.includes("multipart/form-data")) {
        const boundary = contentType.split("boundary=")[1]
        const parts = parseMultipart(body, boundary)
        audioBuffer = parts.file?.data
      } else {
        audioBuffer = body
      }

      if (!audioBuffer?.length) {
        return json(res, 400, { error: { message: "No audio data provided" } })
      }

      const aligned = new ArrayBuffer(audioBuffer.byteLength)
      new Uint8Array(aligned).set(audioBuffer)
      const float32 = new Float32Array(aligned)

      const result = await transcriber(float32)
      return json(res, 200, { text: result.text.trim() })
    } catch (e) {
      console.error("[a2go-tjs] STT error:", e)
      return json(res, 500, { error: { message: e.message } })
    }
  }

  if (req.method === "GET" && (req.url === "/health" || req.url === "/")) {
    return json(res, 200, {
      status: "ok",
      engine: "transformers.js",
      device: DEVICE,
      models: { llm: MODEL_ID, stt: STT_MODEL_ID },
    })
  }

  json(res, 404, { error: { message: "Not found" } })
})

server.listen(PORT, () => {
  console.log(`[a2go-tjs] http://localhost:${PORT}`)
})
