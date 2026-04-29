---
"wandler": patch
---

Log when startup warmup begins, add `--quiet` / `WANDLER_QUIET`, and make prefill chunking adaptive so WebGPU uses the faster full-prompt path only for prompts up to 4096 tokens while long prompts keep the safer 1024-token chunking default.
