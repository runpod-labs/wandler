# Recording TurboQuant comparison videos

Step-by-step for an agent. Two phases: (1) record raw decode timings + token streams, (2) render a video from the recording.

## 0. Prereqs (one-time, already set up on this machine)

- `node_modules/onnxruntime-node/bin/napi-v6/darwin/arm64/libonnxruntime.1.24.3.dylib` is a **symlink** to the custom-built ORT 1.27 dylib at `external/onnxruntime/build/MacOS/Release/libonnxruntime.1.27.0.dylib`. If missing, recreate:
  ```bash
  cd tq-mac-bench/node/node_modules/onnxruntime-node/bin/napi-v6/darwin/arm64
  ln -sf /Users/timpietrusky/.t3/worktrees/wandler/t3code-eb53f7f5/external/onnxruntime/build/MacOS/Release/libonnxruntime.1.27.0.dylib libonnxruntime.1.24.3.dylib
  ln -sf libonnxruntime.1.24.3.dylib libonnxruntime.1.dylib
  ln -sf libonnxruntime.1.dylib libonnxruntime.dylib
  ```
- Must run with **system node** (not bun), because the custom dylib needs NAPI v6 ABI:
  ```bash
  PATH="/Users/timpietrusky/.nvm/versions/node/v22.14.0/bin:$PATH"
  ```

## 1. Record

Two recorders, depending on model type:

| Recorder | Model variants it handles |
|---|---|
| `node/record-video.mjs` | Single-file ONNX with `lasttok` patched meta (lfm, qwen3, qwen35-0.8B) |
| `node/record-video-4b.mjs` | Two-file `decoder_model_merged` + `embed_tokens` (qwen35-4b) |

### Command shape

```bash
cd tq-mac-bench/node
PATH="/Users/timpietrusky/.nvm/versions/node/v22.14.0/bin:$PATH" \
  NO_EOS=1 \
  node record-video.mjs \
    <model-key> <doc-path> <max-new-tokens> <tag> \
    <temp> <top-p> <top-k> <rep-pen> <doc-bytes>
```

- `model-key`: one of `lfm`, `qwen3`, `qwen35` (use `record-video-4b.mjs` for qwen3.5-4B with no extra args except doc-path).
- `doc-path`: any text file. We use `../../TURBOQUANT-KV-CACHE-PLAN.md` for the comparison videos.
- `max-new-tokens`: `200` for ~10s pane.
- `tag`: filename suffix, e.g. `video`.
- Sampling defaults: `0.6 0.95 20 1.05` (T, top-p, top-k, rep-penalty) — matches Qwen's `generation_config.json` recommendation plus a light rep-penalty to prevent loop traps.
- `doc-bytes`: `30000` → ~10K-token prompt. `0` = no truncation.
- `NO_EOS=1`: don't stop on EOS, run the full `max-new-tokens`. Required so both panes have matched lengths.

### Canonical recording for the comparison video

```bash
# 0.8B / 4B / lfm: pick model-key, same other args
PATH="/Users/timpietrusky/.nvm/versions/node/v22.14.0/bin:$PATH" NO_EOS=1 \
  node record-video.mjs <model> ../../TURBOQUANT-KV-CACHE-PLAN.md 200 video 0.6 0.95 20 1.05 30000
```

### Output

Three files in `tq-mac-bench/browser/recordings/`:

- `real-fp16-<model>-real-n200-video.jsonl`
- `real-tq-<model>-real-n200-video.jsonl`
- `real-meta-<model>-real-n200-video.json` — totals, speedup, full text, KV cache bytes, hardware, runtime.

Each JSONL line: `{step, t_ms, dt_ms, token_id, token}`.

## 2. Render

```bash
# 1) Copy the chosen recording into remotion/public/
REC=tq-mac-bench/browser/recordings
PUB=tq-mac-bench/remotion/public
cp $REC/real-fp16-<model>-real-n200-video.jsonl $PUB/fp16.jsonl
cp $REC/real-tq-<model>-real-n200-video.jsonl   $PUB/tq.jsonl
cp $REC/real-meta-<model>-real-n200-video.json  $PUB/meta.json

# 2) Render
cd tq-mac-bench/remotion
bunx remotion render Comparison out/turboquant-comparison.mp4
```

Output: `tq-mac-bench/remotion/out/turboquant-comparison.mp4` (~1080p, 60fps, 1–3 MB).

### Live preview / props panel

```bash
cd tq-mac-bench/remotion && bunx remotion studio
```
Open <http://localhost:3000>. Adjust `speed` (default `3` = 3× playback), `holdFrames` (default `90`) in the right-side props panel.

## Things that are NOT obvious

- **Bench plumbing bug** to be aware of: hybrid linear/full attention models (LFM2.5, Qwen3.5) expose `present.X.{key,value}` **and** `present_conv.X` **and** `present_recurrent.X`. All three must be carried back as `past_*` between decode steps. Missing any → 75% of layers get zeroed every step → garbage output. The recorder handles this; if you fork or rewrite, do not omit `present_recurrent`.
- **Quality drift at long context with same sampling seed** is normal — TurboQuant per-step cosine sim is 0.99+, but free-running generation can diverge after enough tokens. The seeded-sampling we use keeps fp16 and TQ aligned for hundreds of tokens on a 4B model; on tiny models (≤1B) divergence can happen earlier.
- **Use `NO_EOS=1`** for the video. Without it one variant may EOS earlier and the side-by-side looks ragged.
- **Bun for the Remotion side, system node for the recorder.** Bun runs onnxruntime-node fine for synthetic benches but the recorder uses `@huggingface/transformers` which prefers node's module resolution; the npm-published binding is napi-v6 (ABI'd to node 22.x).
