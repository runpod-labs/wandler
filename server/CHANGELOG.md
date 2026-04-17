# wandler

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
