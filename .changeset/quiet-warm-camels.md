---
"wandler": patch
---

Log when startup warmup begins, add `--quiet` / `WANDLER_QUIET`, introduce the explicit `wandler` LLM backend with a `transformersjs` baseline switch, make WebGPU prefill chunking budgeted (`auto:<mb>` customizes it), apply chunked prefill to streaming, and add prefix KV caching for repeated system/tool prefixes.
