<!-- Do not edit or remove this section -->
This document exists for non-obvious, error-prone shortcomings in the codebase, the model, or the tooling that an agent cannot figure out by reading the code alone. No architecture overviews, file trees, build commands, or standard behavior. When you encounter something that belongs here, first consider whether a code change could eliminate it and suggest that to the user. Only document it here if it can't be reasonably fixed.

---

- `TextStreamer` from `@huggingface/transformers` requires the tokenizer to have `all_special_ids` (array) and `decode(ids, opts)` — mocking the tokenizer without these will crash at construction time. See `tests/e2e/helpers.ts` for a working mock.
- `TextStreamer.put()` expects `bigint[][]`, not `number[][]`. The mock model's generate must call `streamer.put()` and `streamer.end()` — not `streamer.callback_function()` directly.
- `presence_penalty` / `frequency_penalty` from the OpenAI API don't map 1:1 to transformers.js. We approximate them via `repetition_penalty` (> 1.0). This is a lossy mapping — they aren't the same thing.
- Gemma ONNX exports don't include a chat template in the tokenizer config. The `formatGemmaChat` fallback in `src/models/tokenizer.ts` handles this, but new Gemma variants may need updates.
- Tool calls with streaming: when `tools` are provided, the server generates the full response first (non-streaming), then parses for tool calls, then wraps the result as SSE chunks. True token-by-token streaming is only used when no tools are specified.
- The `feature-extraction` pipeline returns `{ data: Float32Array }` — the embeddings route casts this to `Array.from()` for JSON serialization. Large embedding batches may be slow due to sequential processing.
- `stop` sequences are approximated by tokenizing each stop string and using the last token as an extra `eos_token_id`. Multi-token stop sequences won't match exactly — only the final token triggers stopping. A proper `StoppingCriteria` implementation would be needed for exact matching.
- The `/v1/completions` streaming path dynamically imports `TextStreamer` via `await import("@huggingface/transformers")` to avoid circular dependency issues with the generation module.
