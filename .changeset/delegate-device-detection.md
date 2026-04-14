---
"wandler": patch
---

Delegate device auto-detection to transformers.js instead of custom resolveDevice. With `--device auto` (the default), the best GPU backend is now selected per platform: CoreML on macOS, CUDA on Linux, DirectML on Windows, with WebGPU and CPU as fallbacks.
