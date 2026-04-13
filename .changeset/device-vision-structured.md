---
"wandler": minor
---

add device auto-detection, multimodal chat support, and structured output

- device defaults to "auto" — detects WebGPU, falls back to cpu
- chat messages support OpenAI multimodal content format (text + image_url arrays)
- response_format supports json_schema with schema injection
- text-only models gracefully handle multimodal input by extracting text
