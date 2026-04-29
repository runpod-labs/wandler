---
"wandler": patch
---

Log when startup warmup begins, add `--quiet` / `WANDLER_QUIET`, and make prefill chunking adaptive so WebGPU uses the fastest full/chunked path that fits a 640MB attention budget (`auto:<mb>` customizes it) while other backends keep 1024-token chunking.
