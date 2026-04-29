---
"wandler": patch
---

Enforce `WANDLER_MAX_CONCURRENT` for generation endpoints, reject prompts that exceed the loaded model context before generation, and fix the RunPod image to run the built repo server with cuDNN available for CUDA.
