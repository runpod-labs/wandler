FROM runpod/a2go:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV WANDLER_HOST=0.0.0.0
ENV WANDLER_PORT=8000
ENV WANDLER_DEVICE=webgpu
ENV WANDLER_PREFILL_CHUNK_SIZE=2048
ENV WANDLER_LOG_LEVEL=debug
ENV PATH=/opt/wandler/node_modules/.bin:$PATH

# Install Vulkan loader/tools only. Do not bake NVIDIA driver packages here:
# RunPod's NVIDIA runtime bind-mounts host-matched driver libraries at startup,
# and image-baked libnvidia-* versions can break NVML/CUDA with version mismatch.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libcudnn9-cuda-12 \
    libvulkan1 \
    vulkan-tools \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/wandler

COPY package.json package-lock.json turbo.json ./
COPY packages ./packages
COPY server/package.json ./server/package.json

RUN npm ci --workspace server --include-workspace-root=false

COPY server ./server
COPY docker ./docker
COPY README.md ./README.md

RUN npm run build --workspace server \
  && chmod +x /opt/wandler/docker/webgpu-smoke.sh \
  && chmod +x /opt/wandler/docker/runpod-wandler-entrypoint.sh

EXPOSE 8000

# Smoke-test after pod boot:
#   /opt/wandler/docker/webgpu-smoke.sh
# Start Wandler:
#   wandler --llm onnx-community/gemma-4-E4B-it-ONNX:q4 --device webgpu --host 0.0.0.0
ENTRYPOINT ["/opt/wandler/docker/runpod-wandler-entrypoint.sh"]
