---
"wandler": patch
---

Fix Gemma long-context generation memory usage by forcing generation logits to the final token, adding chunked prefill, and exposing generation memory diagnostics.

Adds RunPod WebGPU image build and verification tooling for validating the WebGPU backend.
