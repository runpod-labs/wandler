# wandler

## 2.4.1

### Patch Changes

- 3a63ca5: Fix `Cannot apply filter "upper" to type: UndefinedValue` crashes against real agent tool schemas on Gemma models. The 2.4.0 sanitizer only walked top-level tool properties; this extends it to every schema location the template touches.

  Two problems in Gemma's `chat_template.jinja`:
  1. **Nested descriptors without `type`.** The template walks nested `properties`, array `items`, `anyOf`/`oneOf`/`allOf`, and `additionalProperties` — a descriptor at any depth missing `type` crashed `| upper`. Reproduces on OpenClaw's `read` tool (`range.start`, `range.end` have only `description`).
  2. **`type:"object"` descriptors without a `properties` key.** The template has a fallback branch that recursively iterates every _other_ key in the descriptor as if it were a sub-property, crashing on non-descriptor values like `patternProperties` or boolean `additionalProperties`. Reproduces on OpenClaw's `exec.env` (`{"type":"object", "patternProperties":{"^.*$":{"type":"string"}}}`) and `video_generate.providerOptions`.

  The sanitizer now:
  - Recursively ensures every descriptor has a `type` (inferred as `"object"` when `properties` is present, `"array"` when `items` is present, otherwise `"string"`), walking into `properties`, `items`, `anyOf`/`oneOf`/`allOf`, and object-form `additionalProperties`.
  - Injects `properties: {}` on any `type:"object"` descriptor missing the key, so the Jinja primary branch always handles it.

  Verified end-to-end against `openclaw agent` with its full 27-tool schema on Gemma 4 E4B — zero template crashes.

## 2.4.0

### Minor Changes

- ed2ad3c: Derive the default `max_new_tokens` from the loaded model's context length instead of capping every request at a hard-coded `2048`.

  **Behavior before:** the server capped every request at `2048` tokens (or whatever `--max-tokens` was set to). On a model with a 128K context window, a client asking for anything above 2048 was silently clipped.

  **Behavior now:** the effective ceiling is picked in this order of precedence:
  1. **Explicit `--max-tokens <n>`** (or `WANDLER_MAX_TOKENS=<n>`) — opt-in server cap, useful for shared deployments where the host needs to protect itself.
  2. **The loaded model's `max_position_embeddings`** — read from the model config at startup. Vision/multimodal models expose it under `text_config.max_position_embeddings`; text-only models expose it at the top level.
  3. **`FALLBACK_MAX_TOKENS` (2048)** — only reached when neither of the above is available.

  The client's `params.max_tokens` is capped at the effective ceiling; `undefined` or missing `max_tokens` defaults to the ceiling itself, so most clients get the model's full generation headroom automatically.

  Operator-facing changes:
  - `--max-tokens` is now optional; omitting it no longer forces 2048.
  - The server prints the detected context length on startup: `[wandler] Model context: <n> tokens`.

  Breaking-ish: `ServerConfig.maxTokens` changed from `number` (defaulting to `2048`) to `number | null` (defaulting to `null`, meaning "no explicit server cap"). External consumers of `loadConfig` that hardcoded `2048` will still get a valid server — but their returned value may now be `null`.

## 2.3.3

### Patch Changes

- 7b2ffc5: Fix Gemma chat-template crash on tool schemas without an explicit `type`.

  `formatChat` now sanitizes tool parameter schemas before passing them to the tokenizer: every property inside `tools[].function.parameters.properties` gets its `type` defaulted to `"string"` when missing. Gemma's `chat_template.jinja` applies `value['type'] | upper` on every property and throws `Cannot apply filter "upper" to type: UndefinedValue` when the caller omits `type`, which JSON Schema and the OpenAI Chat Completions API both allow. Agent frameworks that emit such schemas (Hermes, OpenClaw) now work against `onnx-community/gemma-4-E4B-it-ONNX` and `onnx-community/gemma-4-E2B-it-ONNX`.

  Upstream template fixes are filed against both Gemma ONNX repos; this workaround can be removed once they merge (tracked in https://github.com/runpod-labs/wandler/issues/18).

## 2.3.2

### Patch Changes

- 231cd87: update @huggingface/transformers from 4.0.1 to 4.1.0

## 2.3.1

### Patch Changes

- e4806a8: Fix device auto-detection crash in Docker and environments with missing GPU drivers. When `device=auto` and onnxruntime crashes on a provider (e.g. CUDA without libcudnn, Vulkan without ICD drivers), the server now tries each device individually in preference order instead of crashing. Unsupported devices for the current platform are skipped instantly.

## 2.3.0

### Minor Changes

- fbe802d: Add `wandler models` CLI command to list verified models with their capabilities. The catalog is built from a model registry, cached locally with a 1-hour TTL, and refreshed from GitHub. Agents can now query available LLM, embedding, and STT models without leaving the CLI.

## 2.2.0

### Minor Changes

- ce983d0: Remove default LLM and STT models. At least one model flag (`--llm`, `--embedding`, or `--stt`) is now required. Any combination is valid.

### Patch Changes

- a764328: Delegate device auto-detection to transformers.js instead of custom resolveDevice. With `--device auto` (the default), the best GPU backend is now selected per platform: CoreML on macOS, CUDA on Linux, DirectML on Windows, with WebGPU and CPU as fallbacks.

## 2.1.0

### Minor Changes

- 49cd72e: add device auto-detection, multimodal chat support, and structured output
  - device defaults to "auto" — detects WebGPU, falls back to cpu
  - chat messages support OpenAI multimodal content format (text + image_url arrays)
  - response_format supports json_schema with schema injection
  - text-only models gracefully handle multimodal input by extracting text

## 2.0.2

### Patch Changes

- 4172877: update readme with cli docs, env vars, and npx quickstart

## 2.0.1

### Patch Changes

- 8d0a6ba: test: verify oidc release flow

## 2.0.0

### Major Changes

- 04fa7b6: Add CLI for running wandler as an inference server via `npx wandler`
  - Support `org/repo:precision` syntax for model selection (q4, q8, fp16, fp32)
  - Add `--host`, `--cors-origin`, `--max-tokens`, `--max-concurrent`, `--timeout`, `--log-level` flags
  - Add `--hf-token` and `--cache-dir` for model management
  - Add `--version` flag
  - All flags have corresponding `WANDLER_` environment variables
  - Publish as `wandler` package on npm
