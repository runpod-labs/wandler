# wandler

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
