#!/usr/bin/env bash
set -euo pipefail

# Keep the RunPod/A2GO image behavior that makes SSH available, then run Wandler
# as the foreground service for this image.
if [ -f /opt/a2go/entrypoint-common.sh ]; then
  # shellcheck source=/dev/null
  source /opt/a2go/entrypoint-common.sh
  oc_create_path_symlinks || true
  oc_setup_ssh_manual || true
elif command -v sshd >/dev/null 2>&1; then
  mkdir -p /run/sshd /root/.ssh
  if [ -n "${PUBLIC_KEY:-}" ]; then
    printf "%s\n" "${PUBLIC_KEY}" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
  fi
  ssh-keygen -A >/dev/null 2>&1 || true
  /usr/sbin/sshd || true
fi

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

exec wandler \
  --llm "${WANDLER_LLM:-onnx-community/gemma-4-E4B-it-ONNX:q4}" \
  --device "${WANDLER_DEVICE:-webgpu}" \
  --host "${WANDLER_HOST:-0.0.0.0}" \
  --port "${WANDLER_PORT:-8000}"
