---
"wandler": patch
---

Log when startup warmup begins, add `--quiet` / `WANDLER_QUIET`, make WebGPU prefill chunking budgeted (`auto:<mb>` customizes it), apply chunked prefill to streaming, and add prefix KV caching for repeated system/tool prefixes.
