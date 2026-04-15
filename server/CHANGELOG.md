# wandler

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
