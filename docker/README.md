# Wandler RunPod WebGPU Image

This image is for testing Wandler with ONNX Runtime Node WebGPU on RunPod.

Build:

```bash
docker build -f docker/runpod-webgpu.Dockerfile -t ghcr.io/runpod-labs/wandler-webgpu:dev .
docker push ghcr.io/runpod-labs/wandler-webgpu:dev
```

Build on GitHub with Blacksmith:

```bash
gh workflow run release.yml -f image_tag=dev -f push_webgpu_image=true
```

Default pushed image:

```bash
ghcr.io/<github-owner>/wandler-webgpu:dev
ghcr.io/<github-owner>/wandler-webgpu:sha-<commit>
ghcr.io/<github-owner>/wandler-webgpu:latest # only on main
```

RunPod pod requirements:

```bash
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

Smoke test inside the pod:

```bash
/opt/wandler/docker/webgpu-smoke.sh
```

Expected:

- `/dev/dri/renderD*` exists.
- `vulkaninfo --summary` shows an NVIDIA physical device, not only `llvmpipe`.
- `npm run verify:webgpu --workspace server` loads Gemma with `device: "webgpu"`.

Start Wandler manually:

```bash
wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4 --device webgpu --host 0.0.0.0 --port 8000
```
