---
"wandler": minor
---

Add transformers.js 4.2 support with 1-bit/2-bit dtypes and incremental streaming tool calls.

- Bump `@huggingface/transformers` from `^4.1.0` to `^4.2.0`
- Add `q1`, `q1f16`, `q2`, `q2f16` dtypes with `ModelRegistry`-based validation and auto-resolution
- Warn when 1-bit/2-bit dtypes are paired with cuda/coreml/dml (ternary kernels only ship with CPU/WebGPU)
- Replace the "generate-then-fake-stream" workaround with an incremental streaming tool-call parser — tokens now stream to the client in real time even when `tools` are present
- Add vision-first probe via `config.json` to avoid wasted downloads for text-only models
- Wire `env.logLevel` and `env.useWasmCache` from transformers.js
