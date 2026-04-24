---
"wandler": minor
---

Add OpenAI Responses API endpoint (`/v1/responses`) with full support for text generation, streaming with named SSE events, and function calling. This enables `provider("model")` in the Vercel AI SDK v6 without needing `.chat()`.
