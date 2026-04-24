---
"wandler": minor
---

Use the standard HuggingFace cache directory (`~/.cache/huggingface`) instead of `node_modules/`. Models are now persisted across reinstalls, `npx` runs, and fresh checkouts. Respects `HF_HOME` and `XDG_CACHE_HOME` environment variables. Override with `WANDLER_CACHE_DIR` or `--cache-dir`.
