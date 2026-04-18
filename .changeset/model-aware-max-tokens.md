---
"wandler": minor
---

Derive the default `max_new_tokens` from the loaded model's context length instead of capping every request at a hard-coded `2048`.

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
