---
"wandler": patch
---

Fix Gemma chat-template crash on tool schemas without an explicit `type`.

`formatChat` now sanitizes tool parameter schemas before passing them to the tokenizer: every property inside `tools[].function.parameters.properties` gets its `type` defaulted to `"string"` when missing. Gemma's `chat_template.jinja` applies `value['type'] | upper` on every property and throws `Cannot apply filter "upper" to type: UndefinedValue` when the caller omits `type`, which JSON Schema and the OpenAI Chat Completions API both allow. Agent frameworks that emit such schemas (Hermes, OpenClaw) now work against `onnx-community/gemma-4-E4B-it-ONNX` and `onnx-community/gemma-4-E2B-it-ONNX`.

Upstream template fixes are filed against both Gemma ONNX repos; this workaround can be removed once they merge (tracked in https://github.com/runpod-labs/wandler/issues/18).
