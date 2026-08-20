#!/usr/bin/env bash
# Gomoku inference service in the pytorch container on node09 (CUDA).
#
#   cd ~/h20_validation_20260724 && bash scripts/run_gomoku_serve.sh [gpu]
#
# Binds to node09's loopback only -- reach it from your machine through an
# SSH tunnel, after which the play page picks it up automatically:
#
#   ssh -N -L 8787:127.0.0.1:8787 node09
#
# (Stop a local serve_gomoku.py first: both want local port 8787.)
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-0}"
docker rm -f az_serve 2>/dev/null || true
docker run -d --name az_serve --gpus "\"device=${GPU}\"" \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --restart unless-stopped \
  -p 127.0.0.1:8787:8787 \
  -e SERVE_HOST=0.0.0.0 \
  -v "$PWD":/workspace -w /workspace \
  harbor.stonewise.cn/base/nvidia/pytorch:25.11-py3-cuda13.0-torch2.10 \
  python scripts/serve_gomoku.py
echo "az_serve starting on GPU ${GPU}; follow with: docker logs -f az_serve"
