---
"wandler": patch
---

Add Wandler's owned text decode loop for supported generation requests. The Wandler backend can now run prefill and token-by-token decode through `forward()` without calling transformers.js `generate()`, while `--decode-loop off` / `WANDLER_DECODE_LOOP=off` keeps the direct `generate()` fallback available.
