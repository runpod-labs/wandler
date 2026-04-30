---
"wandler": patch
---

Keep `--decode-loop auto` on the safe transformers.js `generate()` path by default after release testing showed the experimental owned decode loop is much slower on CUDA. Use `--decode-loop on` to opt into the Wandler decode loop explicitly.
