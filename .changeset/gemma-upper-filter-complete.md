---
"wandler": patch
---

Fix `Cannot apply filter "upper" to type: UndefinedValue` crashes against real agent tool schemas on Gemma models. The 2.4.0 sanitizer only walked top-level tool properties; this extends it to every schema location the template touches.

Two problems in Gemma's `chat_template.jinja`:

1. **Nested descriptors without `type`.** The template walks nested `properties`, array `items`, `anyOf`/`oneOf`/`allOf`, and `additionalProperties` — a descriptor at any depth missing `type` crashed `| upper`. Reproduces on OpenClaw's `read` tool (`range.start`, `range.end` have only `description`).

2. **`type:"object"` descriptors without a `properties` key.** The template has a fallback branch that recursively iterates every *other* key in the descriptor as if it were a sub-property, crashing on non-descriptor values like `patternProperties` or boolean `additionalProperties`. Reproduces on OpenClaw's `exec.env` (`{"type":"object", "patternProperties":{"^.*$":{"type":"string"}}}`) and `video_generate.providerOptions`.

The sanitizer now:

- Recursively ensures every descriptor has a `type` (inferred as `"object"` when `properties` is present, `"array"` when `items` is present, otherwise `"string"`), walking into `properties`, `items`, `anyOf`/`oneOf`/`allOf`, and object-form `additionalProperties`.
- Injects `properties: {}` on any `type:"object"` descriptor missing the key, so the Jinja primary branch always handles it.

Verified end-to-end against `openclaw agent` with its full 27-tool schema on Gemma 4 E4B — zero template crashes.
