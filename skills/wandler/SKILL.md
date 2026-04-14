---
name: wandler
description: Inference server powered by transformers.js.
metadata:
  author: runpod
---

# wandler

`npm install -g wandler` or `npx wandler --llm <org/repo:precision>`

## Usage

```bash
wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4                          # default
wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4 --device cpu             # CPU only
wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4 --embedding Xenova/all-MiniLM-L6-v2:q8  # + embeddings
wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4 --embedding Xenova/all-MiniLM-L6-v2:q8 --stt onnx-community/whisper-tiny:q4  # all three
wandler --llm onnx-community/gemma-4-E4B-it-ONNX:fp16 --port 3000 --host 0.0.0.0 --api-key mysecret  # fp16, custom port, auth

# --llm <id>           LLM model (default: onnx-community/gemma-4-E4B-it-ONNX:q4)
# --embedding <id>     Embedding model (off by default)
# --stt <id>           STT/Whisper (default: onnx-community/whisper-tiny:q4)
# --no-stt             Disable STT
# --device <type>      auto | webgpu | cpu | wasm (default: auto)
# --port <n>           Default: 8000
# --host <addr>        Default: 127.0.0.1
# --api-key <key>      Bearer auth (or env WANDLER_API_KEY)
# --hf-token <token>   HuggingFace token for gated models
# --cors-origin <o>    Allowed CORS origin (default: *)
# --max-tokens <n>     Max tokens per request (default: 2048)
# --max-concurrent <n> Concurrent requests (default: 1)
# --timeout <ms>       Request timeout (default: 120000)
# --log-level <l>      debug | info | warn | error (default: info)
# --cache-dir <path>   Model cache directory
# Precision suffixes:  q4 (default) | q8 | fp16 | fp32
```

Server at `http://127.0.0.1:8000`. OpenAI-compatible — set `baseURL` in any OpenAI SDK.

## Endpoints

- `POST /v1/chat/completions` — chat (streaming + non-streaming)
- `POST /v1/completions` — text completions
- `POST /v1/embeddings` — requires `--embedding`
- `POST /v1/audio/transcriptions` — speech-to-text (Whisper)

## Gotchas

- `--embedding` is off by default — must pass it to enable `/v1/embeddings`.
- STT is on by default — use `--no-stt` to disable.
- Tool calling disables true streaming — full response generated first, then sent as SSE.
- `stop` sequences only match on the last token. Multi-token stops won't match exactly.
