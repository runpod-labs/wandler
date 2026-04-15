---
"wandler": patch
---

Fix device auto-detection crash in Docker and environments with missing GPU drivers. When `device=auto` and onnxruntime crashes on a provider (e.g. CUDA without libcudnn, Vulkan without ICD drivers), the server now tries each device individually in preference order instead of crashing. Unsupported devices for the current platform are skipped instantly.
