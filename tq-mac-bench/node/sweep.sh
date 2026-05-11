#!/bin/bash
# Sweep 4 configs to find the sweet spot for the video.
# After all 4 finish, inspect recordings/real-meta-*.json to pick a winner.
set -e
cd "$(dirname "$0")"
export PATH="/Users/timpietrusky/.nvm/versions/node/v22.14.0/bin:$PATH"
DOC=../../TURBOQUANT-KV-CACHE-PLAN.md
N=200

echo "=== r1: full 67KB doc, T=0 rep_pen=1.3 (greedy+loop-break) ==="
node record-video.mjs qwen35 $DOC $N r1-greedy-full 0 1 0 1.3 0

echo "=== r2: 30KB doc (~10K tok), T=0 rep_pen=1.2 ==="
node record-video.mjs qwen35 $DOC $N r2-greedy-mid 0 1 0 1.2 30000

echo "=== r3: 12KB doc (~4K tok), T=0 rep_pen=1.15 ==="
node record-video.mjs qwen35 $DOC $N r3-greedy-small 0 1 0 1.15 12000

echo "=== r4: 30KB doc, T=0.6 top_p=0.95 top_k=20 rep_pen=1.15 (sampled) ==="
node record-video.mjs qwen35 $DOC $N r4-sampled-mid 0.6 0.95 20 1.15 30000

echo "=== sweep done. metas: ==="
ls -la ../browser/recordings/real-meta-*.json
