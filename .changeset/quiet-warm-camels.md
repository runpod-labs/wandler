---
"wandler": patch
---

Log when startup warmup begins, add `--quiet` / `WANDLER_QUIET`, and make prefill chunking adaptive so WebGPU defaults to the faster full-prompt path while other backends keep the safer 1024-token chunking default.
