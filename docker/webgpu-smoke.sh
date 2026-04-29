#!/usr/bin/env bash
set -euo pipefail

echo "== NVIDIA =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

echo "== Devices =="
ls -l /dev/nvidia* 2>/dev/null || true
ls -l /dev/dri 2>/dev/null || true

echo "== Vulkan =="
vulkaninfo --summary

echo "== Wandler WebGPU =="
cd /opt/wandler
npm run verify:webgpu --workspace server
