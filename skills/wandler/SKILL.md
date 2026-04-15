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
# LLM on CPU with fp16
wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX:fp16 --device cpu
# LLM + embeddings
wandler --llm onnx-community/Qwen3.5-0.8B-Text-ONNX:q4 --embedding Xenova/all-MiniLM-L6-v2:q8
# LLM + embeddings + STT
wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4 --embedding Xenova/all-MiniLM-L6-v2:q8 --stt onnx-community/whisper-tiny:q4
# custom port, auth, listen on all interfaces
wandler --llm LiquidAI/LFM2.5-1.2B-Instruct-ONNX:q4 --port 3000 --host 0.0.0.0 --api-key mysecret

# list all models from the wandler registry
# returns: type, size, precision, capabilities, repo:precision, name
# --type: llm | embedding | stt
wandler model ls

# --llm <id>           LLM model
# --embedding <id>     Embedding model
# --stt <id>           STT model
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

Server at `http://127.0.0.1:8000`.

## API (OpenAI-compatible)

- `POST /v1/chat/completions` — streaming + non-streaming
- `POST /v1/completions`
- `POST /v1/embeddings`
- `POST /v1/audio/transcriptions`
- `GET /v1/models`
- `POST /tokenize`
- `POST /detokenize`
- `GET /admin/metrics`
- `GET /health`

## Gotchas

- Tool calling disables true streaming — full response generated first, then sent as SSE.
- `stop` sequences only match on the last token. Multi-token stops won't match exactly.
