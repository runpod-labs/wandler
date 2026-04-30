---
"wandler": patch
---

Move text LLM generation into the Wandler backend engine instead of delegating through legacy generation helpers, and add tests proving the Wandler path performs chunked prefill with KV handoff while the `transformersjs` baseline sends the full prompt to `generate()`. The benchmark runner can now compare `BACKEND=wandler` and `BACKEND=transformersjs`.
