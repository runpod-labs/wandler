# wandler

OpenAI-compatible inference server powered by [transformers.js](https://huggingface.co/docs/transformers.js). Run ONNX models locally with WebGPU acceleration or CPU — no Python, no CUDA required.

Think vLLM or llama.cpp, but for the ts crowd.

## Quickstart

```bash
npx wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4
```

```bash
# custom model, precision, device, port
npx wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX:fp16 --device cpu --port 3000
```

```bash
# with embeddings and STT
npx wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4 \
  --embedding Xenova/all-MiniLM-L6-v2:q8 \
  --stt onnx-community/whisper-tiny:q4
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

## CLI

```
wandler — transformers.js inference server

Usage:
  wandler --llm org/repo[:precision] [options]
  wandler model ls [--type <type>]

Commands:
  models                    List available models from the catalog

Model:
  -l, --llm <id>              LLM model
  -e, --embedding <id>        Embedding model
  -s, --stt <id>              STT model
  -d, --device <type>         Device: auto, cpu, cuda, coreml, dml, webgpu, wasm (default: auto)
      --hf-token <token>      HuggingFace token for gated models
      --cache-dir <path>      Model cache directory

Server:
  -p, --port <number>         Port (default: 8000)
      --host <addr>           Bind address (default: 127.0.0.1)
  -k, --api-key <key>         API key for auth (or WANDLER_API_KEY)
      --cors-origin <origin>  Allowed CORS origin (default: *)
      --max-tokens <n>        Max tokens per request (default: 2048)
      --max-concurrent <n>    Max concurrent requests (default: 1)
      --timeout <ms>          Request timeout in ms (default: 120000)
      --log-level <level>     debug, info, warn, error (default: info)
      --prefill-chunk-size <n>
                              Chunk size for long-prompt prefill; 0/off disables it
      --warmup-tokens <n>     Approximate prompt tokens to run once before serving
      --warmup-max-new-tokens <n>
                              Max new tokens for startup warmup

Info:
  -v, --version               Show version
  -h, --help                  Show this help
```

Precision suffixes: `q4`, `q8`, `fp16`, `fp32` (default: `q4`)

## Environment Variables

Every CLI flag has a corresponding environment variable:

| Variable | Default | Description |
|----------|---------|-------------|
| `WANDLER_LLM` | onnx-community/gemma-4-E4B-it-ONNX:q4 | LLM model with precision |
| `WANDLER_STT` | onnx-community/whisper-tiny:q4 | Speech-to-text model |
| `WANDLER_EMBEDDING` | — | Embedding model (disabled by default) |
| `WANDLER_DEVICE` | webgpu | Device: webgpu, cpu, wasm |
| `WANDLER_PORT` | 8000 | Server port |
| `WANDLER_HOST` | 127.0.0.1 | Bind address |
| `WANDLER_API_KEY` | — | API key for auth |
| `WANDLER_CORS_ORIGIN` | * | Allowed CORS origin |
| `WANDLER_MAX_TOKENS` | 2048 | Max tokens per request |
| `WANDLER_MAX_CONCURRENT` | 1 | Max concurrent requests |
| `WANDLER_TIMEOUT` | 120000 | Request timeout (ms) |
| `WANDLER_LOG_LEVEL` | info | Log level |
| `WANDLER_CACHE_DIR` | ~/.cache/huggingface | Model cache directory (also respects `HF_HOME`) |
| `WANDLER_PREFILL_CHUNK_SIZE` | 2048 | Chunk size for long-prompt prefill; set `0`/`off` to disable |
| `WANDLER_WARMUP_TOKENS` | 0 | Approximate prompt tokens to run once before serving |
| `WANDLER_WARMUP_MAX_NEW_TOKENS` | 8 | Max new tokens for startup warmup |
| `HF_TOKEN` | — | HuggingFace token for gated models |

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
| `/v1/embeddings` | POST | Text embeddings |

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
| `/admin/metrics` | GET | Request metrics |

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

### Embeddings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | string \| string[] | required | Text to embed |
| `encoding_format` | string | "float" | "float" or "base64" |

## Compatible Models

List all verified models with their capabilities:

```bash
wandler model ls
```

```
type      | size  | prec | capabilities             | repo:precision                                   | name
------------------------------------------------------------------------------------------------------------------------
llm       | 2B    | q4   | chat, tool-calling       | onnx-community/gemma-4-E4B-it-ONNX:q4            | Gemma 4 E4B
llm       | 1.2B  | q4   | chat, tool-calling       | LiquidAI/LFM2.5-1.2B-Instruct-ONNX:q4            | LFM 2.5 1.2B
llm       | 350M  | q4   | chat, tool-calling       | LiquidAI/LFM2.5-350M-ONNX:q4                     | LFM 2.5 350M
llm       | 0.8B  | q4   | chat, tool-calling       | onnx-community/Qwen3.5-0.8B-Text-ONNX:q4         | Qwen 3.5 0.8B
llm       | 1.7B  | q4   | chat                     | HuggingFaceTB/SmolLM2-1.7B-Instruct:q4           | SmolLM2 1.7B
embedding | 22M   | q8   | embedding                | Xenova/all-MiniLM-L6-v2:q8                       | all-MiniLM-L6-v2
embedding | 33M   | q8   | embedding                | Xenova/bge-small-en-v1.5:q8                      | BGE Small EN v1.5
embedding | 137M  | q8   | embedding                | nomic-ai/nomic-embed-text-v1.5:q8                | Nomic Embed Text v1.5
stt       | 39M   | q4   | transcription            | onnx-community/whisper-tiny:q4                   | Whisper Tiny
stt       | 74M   | q4   | transcription            | onnx-community/whisper-base:q4                   | Whisper Base
stt       | 244M  | q4   | transcription            | onnx-community/whisper-small:q4                  | Whisper Small
```

Filter by type:

```bash
wandler model ls --type llm
wandler model ls --type embedding
wandler model ls --type stt
```

Use the `repo:precision` value directly with `--llm`, `--embedding`, or `--stt`.

Any ONNX model from [onnx-community](https://huggingface.co/onnx-community) or [transformers.js compatible models](https://huggingface.co/models?library=transformers.js) should work beyond the verified catalog.

## Tool Calling

wandler parses tool calls from multiple model output formats:

- **LFM**: `[func_name(arg="val")]` and `[tool_calls [{...}]]`
- **Qwen**: `<tool_call>{"name": "...", "arguments": {...}}</tool_call>`
- **OpenAI JSON**: `{"tool_calls": [...]}`

Thinking blocks (`<think>...</think>`) are automatically stripped before parsing.

## License

MIT
