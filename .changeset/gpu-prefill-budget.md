---
"wandler": patch
---

Use the automatic prefill attention-memory budget on CUDA, CoreML, and DML in addition to WebGPU, so medium prompts can stay on the faster full-prompt path when the estimated attention tensor fits the configured budget. Repeated tool/system prefixes now also reuse cached prefix token counts, and prefix-cache hits pass short request suffixes directly into `generate()` instead of doing an extra suffix prefill forward pass.
