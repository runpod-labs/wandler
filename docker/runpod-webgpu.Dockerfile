FROM runpod/a2go:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
ENV WANDLER_HOST=0.0.0.0
ENV WANDLER_PORT=8000
ENV WANDLER_DEVICE=webgpu
ENV WANDLER_PREFILL_CHUNK_SIZE=2048
ENV WANDLER_LOG_LEVEL=debug
ENV PATH=/opt/wandler/node_modules/.bin:$PATH

# Install the Vulkan loader/tools and NVIDIA's GL/Vulkan user-space ICD in the
# image layer. Installing libnvidia-gl in a running RunPod GPU container can fail
# because driver files are already bind-mounted by the NVIDIA runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    libvulkan1 \
    vulkan-tools \
    libcudnn9-cuda-12 \
    libnvidia-gl-570 \
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
