# Patches for ONNX Runtime

These patches add TurboQuant KV-cache quantization support to ONNX Runtime's CUDA execution provider, plus a graph-rewrite tool that lets stock HuggingFace ONNX exports use TurboQuant via a session option (no offline conversion needed).

## `turboquant-kv-cache.patch`

A single unified diff of 18 commits on top of upstream `microsoft/onnxruntime` commit `b81f3f855` (CUDA TurboQuant + WebGPU scaffold). To apply:

```bash
git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime
git checkout b81f3f855
git apply ../patches/turboquant-kv-cache.patch
```

The patch base (`b81f3f855`) is `fix: make sympy an optional runtime dependency (#28141)` from main as of 2026-05.

### What the patch adds

**TurboQuant CUDA kernels** (`onnxruntime/contrib_ops/cuda/bert/group_query_attention_turboquant_*`):

- Encode kernel: writes fresh K/V to packed cache (Walsh-Hadamard rotation + Lloyd-Max codebook for K, uniform asymmetric for V).
- Decode kernel: reads packed cache → fp16 K/V scratch.
- v4-lite fused FlashAttention kernel for decode step (`TQFlashAttentionKernel`).
- v5 q-tiled kernel for continuation-prefill (`TQFlashAttentionQTiledKernel`).
- v6 wmma tensor-core kernel for prefill scores (`TQFlashAttentionWmmaKernel`).
- Vectorised uint4 K/V smem loads (8-fp16 per HBM transaction).
- Option ε: prompt-step delegates to standard FlashAttention on fresh fp16 K/V.

**ONNX Runtime graph transformer** (`onnxruntime/core/optimizer/turboquant_kv_fusion.{cc,h}`):

- Runs at session-create when the session option `optimization.turboquant_kv_method` is set.
- Rewrites every `GroupQueryAttention` node to use TurboQuant KV-cache.
- Adds shared codebook + Hadamard graph initializers.
- Mutates past_key/past_value/present_key/present_value tensor types from fp16 to packed uint8.
- User downloads stock `model_q4f16.onnx` from HF — no offline `.onnx` conversion needed.

**Last-token logits patcher** (`onnxruntime/python/tools/quantization/turboquant_kv/last_token_logits.py`):

- Standalone graph-edit tool that slices the LM-head input to `[:, -1:, :]` so logits are computed only for the last position.
- Standard LLM-serving optimisation (HF transformers calls it `logits_to_keep=1`).
- Required for any HF causal-LM ONNX export at `S_q × vocab > 2³¹` because of an int32 element-count overflow in ORT's CUDA `Cast` kernel ([ORT issue #28385](https://github.com/microsoft/onnxruntime/issues/28385), filed by us).

**Schema extensions** (`onnxruntime/core/graph/contrib_ops/bert_defs.cc`, `onnxruntime/include/.../onnxruntime_session_options_config_keys.h`):

- New `GroupQueryAttention` attributes: `kv_quant_method`, `key_quant_bits`, `value_quant_bits`, `norm_correction`.
- New optional inputs at slots 14, 15: `k_codebook`, `hadamard`.
- New session option keys: `optimization.turboquant_kv_method`, `optimization.turboquant_kv_boundary`.

**Python TurboQuant tooling** (`onnxruntime/python/tools/quantization/turboquant_kv/`):

- `centroids.py`, `hadamard.py`, `packing.py`, `quantizer.py`: pure-NumPy reference implementation cross-validated bit-exact against vLLM.
- `onnx_rewriter.py`: offline alternative to the graph transformer.
- `validate.py`: 23/23 paper-validation tests.
- `benchmark.py`: standalone perf bench.

### Which commits

```
9f55c2f7d  feat(turboquant): end-to-end TurboQuant KV cache compression on CUDA
1f79c7e30  turboquant(cuda): drop bulk-dequant temp buffers in v1
ae6c1babe  turboquant(cuda): v2 — real attention math, validated on Llama-3.2-1B
9440913c8  turboquant(cuda): make v2 work end-to-end on dynamic-cache GQA models (LFM2-1.2B)
641e3d5dc  turboquant(cuda): v3 — on-device fp16 -> fp32 codebook conversion
9449f9020  turboquant(cuda): apply inline RoPE for do_rotary=1 GQA models
509f3469b  turboquant(cuda): v4-lite — fused FlashAttention-style kernel with online softmax
d3b013aab  turboquant(cuda): fix BSNH/BNSH stride mismatch in encode kernel — quality 0.18 -> 0.996
11e633b1d  turboquant(cuda): v5 q-tiled FlashAttention kernel (~2x prompt speedup at 4K)
363b6dfa8  turboquant: option A — runtime graph rewrite, no offline model conversion
b749d0fa9  turboquant(cuda): v6 — wmma tensor cores for Q*K^T (~20% extra prompt speedup)
b61ef5234  turboquant(cuda): v7 — skip lossy round-trip for new tokens + causal early-exit
490a019bb  turboquant(cuda): tune wmma kBlockQ back to 32 at long context
006e15411  turboquant(cuda): option ε — delegate prompt-step attention to FlashAttention
c29ea0318  turboquant: last_token_logits patcher — unlocks long context (>32K)
c26dd3e20  turboquant(cuda): vectorize K/V smem loads in v4-lite + v6 wmma decode kernels
fc7f206dc  turboquant(cuda): tried cp.async double-buffered K/V load, regressed at our tile size
29376c874  turboquant(webgpu): scaffold TurboQuant integration in webgpu/group_query_attention
```

**WebGPU scaffold** (commit `29376c874`):
Adds the TurboQuant code path to ORT's WebGPU EP — encode + decode WGSL templates, C++ Program wrappers, integration into `webgpu/GroupQueryAttention::ComputeInternal`, and Option-A graph rewrite scope extended to include WebGPU.  The patched ORT builds cleanly with `--use_webgpu`; fp16 baseline runs through the WebGPU provider (validated on Lavapipe / Vulkan-CPU).  The TurboQuant attention path itself currently fails inside Dawn's storage-buffer accounting at FlashAttention bind time — see commit message for diagnosis.  Apple Silicon Metal users may get past it depending on backend limits; the `.github/workflows/turboquant-mac-bench.yml` workflow is the right place to experiment.
