# wandler

OpenAI-compatible inference server powered by [transformers.js](https://huggingface.co/docs/transformers.js). Run ONNX models locally with WebGPU acceleration or CPU — no Python, no CUDA required.

Think vLLM or llama.cpp, but for the TypeScript ecosystem.

## Quickstart

```bash
npm install
npm start          # loads default model, serves on :8000
```

```bash
# custom model + device
MODEL_ID=onnx-community/Qwen3.5-0.8B-Text-ONNX DEVICE=cpu npm start
```

```bash
# with embeddings
EMBEDDING_MODEL_ID=Xenova/all-MiniLM-L6-v2 npm start
```

Use it with the OpenAI SDK:

```typescript
import OpenAI from "openai";

const client = new OpenAI({ baseURL: "http://localhost:8000/v1", apiKey: "-" });

const response = await client.chat.completions.create({
  model: "onnx-community/gemma-4-E4B-it-ONNX",
  messages: [{ role: "user", content: "Hello!" }],
  stream: true,
});

for await (const chunk of response) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
```

## Endpoints

### LLM

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion (streaming + non-streaming) |
| `/v1/completions` | POST | Text completion (legacy) |
| `/v1/models` | GET | List loaded models |
| `/v1/models/{id}` | GET | Get model details |

### Embeddings

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/embeddings` | POST | Text embeddings (requires `EMBEDDING_MODEL_ID`) |

### Audio

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/transcriptions` | POST | Speech-to-text (Whisper) |

### Utilities

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/tokenize` | POST | Text to token IDs |
| `/detokenize` | POST | Token IDs to text |
| `/health` | GET | Server status |

## Parameters

### Chat & Text Completions

Standard OpenAI parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `messages` / `prompt` | array / string | required | Input |
| `temperature` | float | 0.7 | Sampling temperature (0 = greedy) |
| `top_p` | float | 0.95 | Nucleus sampling |
| `max_tokens` | int | 2048 | Max tokens to generate |
| `stream` | bool | false | Enable SSE streaming |
| `stop` | string \| string[] | — | Stop sequences |
| `presence_penalty` | float | 0 | Penalize token presence |
| `frequency_penalty` | float | 0 | Penalize token frequency |
| `response_format` | object | — | `{"type": "json_object"}` for JSON mode |
| `tools` | array | — | Function calling definitions |
| `stream_options` | object | — | `{"include_usage": true}` |

Extended parameters (vLLM/llama.cpp compatible):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `top_k` | int | — | Top-k sampling |
| `min_p` | float | — | Minimum probability threshold |
| `typical_p` | float | — | Locally typical sampling |
| `repetition_penalty` | float | — | Direct repetition penalty (> 1.0) |
| `no_repeat_ngram_size` | int | — | Prevent N-gram repetition |

Text completions also support `echo` and `suffix`.

### Embeddings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string \| string[] | required | Text to embed |
| `encoding_format` | string | "float" | "float" or "base64" |

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `MODEL_ID` | onnx-community/gemma-4-E4B-it-ONNX | LLM model from HuggingFace |
| `DTYPE` | q4 | Quantization: q4, q8, fp16, fp32 |
| `DEVICE` | webgpu | Runtime: webgpu, cpu, wasm |
| `STT_MODEL_ID` | onnx-community/whisper-tiny | Speech-to-text model |
| `STT_DTYPE` | q4 | STT quantization |
| `EMBEDDING_MODEL_ID` | — | Embedding model (empty = disabled) |
| `EMBEDDING_DTYPE` | q8 | Embedding quantization |

## Compatible Models

### LLMs

| Model | ID | Size | Notes |
|-------|-------|------|-------|
| Gemma 4 | `onnx-community/gemma-4-E4B-it-ONNX` | ~2B | Default, good general purpose |
| Qwen 3.5 | `onnx-community/Qwen3.5-0.8B-Text-ONNX` | 0.8B | Fast, good for tool calling |
| LFM 2.5 | `LiquidAI/LFM2.5-1.2B-Instruct-ONNX` | 1.2B | 82 tok/s on M3 WebGPU |
| LFM 2.5 | `LiquidAI/LFM2.5-350M-ONNX` | 350M | Ultra-fast, lightweight |
| SmolLM2 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | Good instruction following |
| Phi-2 | `microsoft/phi-2` | 2.7B | Strong reasoning |

Any ONNX model from the [onnx-community](https://huggingface.co/onnx-community) or [transformers.js compatible models](https://huggingface.co/models?library=transformers.js) should work.

### Embeddings

| Model | ID | Dimensions |
|-------|-------|------------|
| all-MiniLM-L6-v2 | `Xenova/all-MiniLM-L6-v2` | 384 |
| bge-small-en-v1.5 | `Xenova/bge-small-en-v1.5` | 384 |
| nomic-embed-text-v1.5 | `nomic-ai/nomic-embed-text-v1.5` | 768 |
| mxbai-embed-large-v1 | `mixedbread-ai/mxbai-embed-large-v1` | 1024 |

### Audio (Speech-to-Text)

| Model | ID | Notes |
|-------|-------|-------|
| Whisper Tiny | `onnx-community/whisper-tiny` | Default, fast |
| Whisper Small | `onnx-community/whisper-small` | Better accuracy |
| Whisper Base | `onnx-community/whisper-base` | Balance of speed/accuracy |

## Tool Calling

wandler parses tool calls from multiple model output formats:

- **LFM**: `[func_name(arg="val")]` and `[tool_calls [{...}]]`
- **Qwen**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **OpenAI JSON**: `{"tool_calls": [...]}`

Thinking blocks (`<think>...</think>`) are automatically stripped before parsing.

## Development

```bash
npm run dev         # watch mode with hot reload
npm test            # run all tests (81 tests)
npm run test:unit   # unit tests only
npm run test:e2e    # E2E tests only
npm run typecheck   # TypeScript type checking
npm run benchmark   # run model benchmarks
npm run build       # compile to dist/
```

## Architecture

```
src/
├── index.ts              # entry point
├── config.ts             # env var configuration
├── server.ts             # HTTP server + routing
├── routes/
│   ├── chat.ts           # /v1/chat/completions
│   ├── completions.ts    # /v1/completions
│   ├── embeddings.ts     # /v1/embeddings
│   ├── models.ts         # /v1/models
│   ├── audio.ts          # /v1/audio/transcriptions
│   ├── tokenize.ts       # /tokenize, /detokenize
│   └── health.ts         # /health
├── generation/
│   ├── generate.ts       # non-streaming generation
│   ├── stream.ts         # SSE streaming generation
│   └── options.ts        # sampling parameter mapping
├── models/
│   ├── manager.ts        # model loading (LLM, STT, embeddings)
│   └── tokenizer.ts      # chat template formatting
├── tools/
│   └── parser.ts         # multi-format tool call parser
├── types/
│   └── openai.ts         # OpenAI-compatible type definitions
└── utils/
    ├── http.ts           # HTTP helpers
    └── multipart.ts      # multipart form parser
```

## License

MIT
