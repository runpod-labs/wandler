# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Generic TurboQuant benchmark.  Usable for any HuggingFace causal-LM ONNX
export (LFM2, Qwen3, Llama-3.x, etc.).  Auto-detects model architecture from
graph inputs (handles `position_ids`, `past_conv.X` for hybrid SSM models, etc.).

Set environment variables to control the run:
    MODEL_PATH    — path to the .onnx file (must be patched with last_token_logits)
    LABEL         — display label (model name)
    BENCH_MODE    — "fp16" (no TQ) or "tq" (TurboQuant via Option A graph rewrite)
    PROMPT_LEN    — initial prompt length in tokens
    DECODE_STEPS  — number of decode steps to time
    SLOT          — TurboQuant slot bytes (36 for hd=64 4-bit; 68 for hd=128 4-bit)
"""
import numpy as np
import onnxruntime as ort
import onnx
import time
import os
import sys


def main() -> int:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        print("ERROR: set MODEL_PATH=path/to/model_q4f16_lasttok.onnx", flush=True)
        return 1
    prompt_len = int(os.environ.get("PROMPT_LEN", "4096"))
    decode_steps = int(os.environ.get("DECODE_STEPS", "8"))
    label = os.environ.get("LABEL", "model")
    slot = int(os.environ.get("SLOT", "36"))
    mode = os.environ.get("BENCH_MODE", "")

    print(f"=== {label} | mode={mode} prompt={prompt_len} decode={decode_steps} ===", flush=True)

    m = onnx.load(model_path, load_external_data=False)
    specs = {
        i.name: (
            [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim],
            i.type.tensor_type.elem_type,
        )
        for i in m.graph.input
    }
    output_names = [o.name for o in m.graph.output]
    gqa_count = sum(1 for n in m.graph.node if n.op_type == "GroupQueryAttention")
    print(f"  GQA layers: {gqa_count}", flush=True)
    hd = None
    for i in m.graph.input:
        if "past_key_values" in i.name and ".key" in i.name:
            dims = i.type.tensor_type.shape.dim
            if len(dims) == 4 and dims[3].dim_value > 0:
                hd = dims[3].dim_value
                break
    print(f"  head_dim: {hd} | TQ slot bytes: {slot}", flush=True)

    def make_session(use_tq: bool) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        if use_tq:
            opts.add_session_config_entry(
                "optimization.turboquant_kv_method", "turboquant_4bit_nc"
            )
            opts.add_session_config_entry(
                "optimization.turboquant_kv_boundary", "0"
            )
        opts.log_severity_level = 3
        # Provider preference: webgpu first if built, else cuda, else cpu.
        avail = ort.get_available_providers()
        providers = []
        if "WebGpuExecutionProvider" in avail:
            providers.append("WebGpuExecutionProvider")
        if "CUDAExecutionProvider" in avail:
            providers.append(("CUDAExecutionProvider", {"device_id": 0}))
        providers.append("CPUExecutionProvider")
        return ort.InferenceSession(model_path, opts, providers=providers)

    def make_initial_inputs(plen: int, use_tq: bool) -> dict:
        np.random.seed(42)
        inputs = {
            "input_ids": np.random.randint(0, 32000, size=(1, plen), dtype=np.int64),
            "attention_mask": np.ones((1, plen), dtype=np.int64),
        }
        if "position_ids" in specs:
            inputs["position_ids"] = np.arange(plen, dtype=np.int64).reshape(1, plen)
        for name, (shape, elem) in specs.items():
            if name in inputs:
                continue
            concrete = []
            is_kv = "past_key_values" in name
            for d in shape:
                if isinstance(d, int):
                    concrete.append(d)
                elif d == "batch_size":
                    concrete.append(1)
                elif d == "sequence_length":
                    concrete.append(plen)
                elif d == "past_sequence_length":
                    concrete.append(0)
                elif d == "total_sequence_length":
                    concrete.append(plen)
                else:
                    concrete.append(0)
            if use_tq and is_kv:
                concrete[-1] = slot
                dtype = np.uint8
            else:
                dtype = {1: np.float32, 2: np.uint8, 7: np.int64, 10: np.float16}[elem]
            inputs[name] = np.zeros(concrete, dtype=dtype)
        return inputs

    def step_decode(prev_inputs: dict, prev_outputs: list, new_token: int) -> dict:
        past_kv: dict = {}
        for i, name in enumerate(output_names):
            if name.startswith("present.") and (".key" in name or ".value" in name):
                past_kv["past_key_values." + name[len("present."):]] = prev_outputs[i]
            elif name.startswith("present_conv."):
                past_kv["past_conv." + name[len("present_conv."):]] = prev_outputs[i]
        past_seq = next(
            iter(v for k, v in past_kv.items() if k.startswith("past_key_values."))
        ).shape[2]
        total_seq = past_seq + 1
        new_inputs: dict = {
            "input_ids": np.array([[new_token]], dtype=np.int64),
            "attention_mask": np.ones((1, total_seq), dtype=np.int64),
        }
        if "position_ids" in specs:
            new_inputs["position_ids"] = np.array([[past_seq]], dtype=np.int64)
        for name, (shape, elem) in specs.items():
            if name in new_inputs:
                continue
            if name in past_kv:
                new_inputs[name] = past_kv[name]
                continue
            new_inputs[name] = prev_inputs[name]
        return new_inputs

    def bench(use_tq: bool):
        sess = make_session(use_tq)
        inputs = make_initial_inputs(prompt_len, use_tq)
        t0 = time.time()
        outs = sess.run(output_names, inputs)
        t_prompt = (time.time() - t0) * 1000
        last_token = int(outs[0][0, -1].astype(np.float32).argmax())
        print(f"  prompt (TTFT): {t_prompt:.0f} ms", flush=True)
        decode_times = []
        for _ in range(decode_steps):
            inputs = step_decode(inputs, outs, last_token)
            t1 = time.time()
            outs = sess.run(output_names, inputs)
            decode_times.append((time.time() - t1) * 1000)
            last_token = int(outs[0][0, -1].astype(np.float32).argmax())
        steady = decode_times[1:] if len(decode_times) > 1 else decode_times
        avg = sum(steady) / len(steady)
        print(
            f"  decode ({decode_steps} steps): step1={decode_times[0]:.1f}ms "
            f"steady avg={avg:.1f} ms/tok",
            flush=True,
        )
        return t_prompt, avg

    if mode == "fp16":
        p, d = bench(use_tq=False)
    elif mode == "tq":
        p, d = bench(use_tq=True)
    else:
        print("set BENCH_MODE=fp16 or BENCH_MODE=tq", flush=True)
        return 1
    print(f"\nresult: prompt={p:.0f}ms decode={d:.1f}ms/tok\n", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
