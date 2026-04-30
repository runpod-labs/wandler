---
name: wandler
description: Inference server powered by transformers.js.
metadata:
  author: runpod
---

# wandler

`npm install -g wandler` or `npx wandler --llm <org/repo:precision>`

```bash
# LLM
wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4
# LLM with the Wandler serving backend (default)
wandler --backend wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4 --device webgpu
# Direct transformers.js baseline for A/B testing
wandler --backend transformersjs --llm onnx-community/gemma-4-E4B-it-ONNX:q4 --device webgpu
# LLM on CPU with fp16
wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX:fp16 --device cpu
# LLM + embeddings
wandler --llm onnx-community/Qwen3.5-0.8B-Text-ONNX:q4 --embedding Xenova/all-MiniLM-L6-v2:q8
# LLM + embeddings + STT
wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4 --embedding Xenova/all-MiniLM-L6-v2:q8 --stt onnx-community/whisper-tiny:q4
# custom port, auth, listen on all interfaces
wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX:q4 --port 3000 --host 0.0.0.0 --api-key mysecret

# --llm <id>                         LLM model (env WANDLER_LLM)
# --backend <name>                   wandler | transformersjs (default: wandler; env WANDLER_BACKEND)
# --embedding <id>                   Embedding model (env WANDLER_EMBEDDING)
# --stt <id>                         STT model (env WANDLER_STT)
# --device <type>                    auto | cuda | coreml | dml | webgpu | cpu | wasm (default: auto)
# --port <n>                         Default: 8000
# --host <addr>                      Default: 127.0.0.1
# --api-key <key>                    Bearer auth (env WANDLER_API_KEY)
# --hf-token <token>                 HuggingFace token for gated models
# --cors-origin <o>                  Allowed CORS origin (default: *)
# --max-tokens <n>                   Max tokens per request (default: loaded model context)
# --max-concurrent <n>               Concurrent requests (default: 1)
# --timeout <ms>                     Request timeout (default: 120000)
# --log-level <l>                    debug | info | warn | error (default: info)
# --quiet                            Suppress non-error startup/profile logs (env WANDLER_QUIET)
# --cache-dir <path>                 Model cache directory (default: HF cache)
# --prefill-chunk-size <n>           auto | auto:<mb> | 0/off | integer chunk size
#                                    auto uses a 640MB WebGPU attention budget; other backends use 1024
# --decode-loop <mode>               auto | on | off (default: auto; on is experimental; env WANDLER_DECODE_LOOP)
# --prefix-cache <mode>              true | false (default: true; env WANDLER_PREFIX_CACHE)
# --prefix-cache-entries <n>         Prefix KV cache entries (default: 2)
# --prefix-cache-min-tokens <n>      Minimum prefix tokens to cache (default: 512)
# --warmup-tokens <n>                Approximate startup warmup prompt tokens (default: 0)
# --warmup-max-new-tokens <n>        Startup warmup max new tokens (default: 8)
# Precision suffixes:                q4 (default) | q8 | fp16 | fp32 | auto

# list all models from the wandler registry
# returns: type, size, precision, capabilities, repo:precision, name
# --type: llm | embedding | stt
wandler model ls
```

Server at `http://127.0.0.1:8000`.

## API (OpenAI-compatible)

- `POST /v1/chat/completions` — streaming + non-streaming
- `POST /v1/responses` — streaming + non-streaming
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/audio/transcriptions`
- `GET /v1/models`
- `POST /tokenize`
- `POST /detokenize`
- `GET /admin/metrics`
- `GET /health`

## Gotchas

- Use `WANDLER_LLM`, not `WANDLER_MODEL`.
- `--backend wandler` is the default optimized serving path; `--backend transformersjs` is the direct baseline for A/B testing.
- `--decode-loop auto` uses the safe transformers.js `generate()` path. Use `--decode-loop on` only to opt into Wandler's experimental owned text decode loop.
- Prefix KV cache is process-local and bounded by `WANDLER_PREFIX_CACHE_ENTRIES`; it helps repeated system/tool prefixes but is not true paged attention.
- Tool calls stream incrementally in Chat Completions and Responses; family-specific openers are buffered so partial tool-call markers do not leak as content.
- `stop` sequences only match on the last token. Multi-token stops won't match exactly.
