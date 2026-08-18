#!/usr/bin/env bash
# AlphaZero Gomoku Phase 1 on node09: one arm, 15x15, sized from the 2026-07-31
# Phase-0 calibration (results/calib_*.jsonl):
#
#   - 9x9 is a draw at this strength (75-81% draws regardless of search depth),
#     so the board moves to 15x15 / freestyle five-in-a-row.
#   - Self-play is CPU-bound on the Python tree walk, so throughput comes from
#     worker count (7 GPUs x 4 procs = 28), not from batch size: 1,344 games
#     per iteration keeps a measured ~11 min self-play per iteration.
#   - GPU 5 is avoided: another user's job holds memory on it.
#
# Expected: ~14-16 min per iteration, 40 iterations ≈ one overnight run.
# Run from the workdir, which is mounted at /workspace.
set -euo pipefail
cd /workspace

export AZ_BOARD=${AZ_BOARD:-15}
export AZ_CH=${AZ_CH:-192}
export AZ_BLOCKS=${AZ_BLOCKS:-12}
export AZ_ITERS=${AZ_ITERS:-40}
export AZ_GAMES=${AZ_GAMES:-1344}
export AZ_SIMS=${AZ_SIMS:-800}
export AZ_TEMP_MOVES=${AZ_TEMP_MOVES:-20}
export AZ_BUFFER=${AZ_BUFFER:-1500000}
export AZ_TRAIN_STEPS=${AZ_TRAIN_STEPS:-700}
export AZ_BATCH=${AZ_BATCH:-512}
export AZ_GPUS=${AZ_GPUS:-0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,6,6,6,6,7,7,7,7}
export AZ_EVAL_EVERY=${AZ_EVAL_EVERY:-5}
export AZ_EVAL_SIMS=${AZ_EVAL_SIMS:-800}
export AZ_EVAL_GAMES=${AZ_EVAL_GAMES:-12}
export AZ_EVAL_GAMES_RAND=${AZ_EVAL_GAMES_RAND:-8}
export AZ_PURE_PLAYOUTS_HARD=${AZ_PURE_PLAYOUTS_HARD:-8000}
export AZ_ELO_SIMS=${AZ_ELO_SIMS:-200}
export AZ_ELO_GAMES=${AZ_ELO_GAMES:-8}
export AZ_ELO_TEMP=${AZ_ELO_TEMP:-0.3}
export AZ_ANCHOR_GAMES=${AZ_ANCHOR_GAMES:-12}
export AZ_ANCHOR_K=${AZ_ANCHOR_K:-3}
export AZ_ANCHOR_SIMS=${AZ_ANCHOR_SIMS:-200}
export AZ_ANCHOR_TEMP=${AZ_ANCHOR_TEMP:-0.3}
export AZ_SEED=${AZ_SEED:-42}
export AZ_BETA0=${AZ_BETA0:-0.0}

TAG=${AZ_TAG:-p15}

echo "=== env ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
PY

echo "=== phase-1: 15x15, tag=${TAG}, beta0=${AZ_BETA0} ==="
bash scripts/gpu_monitor.sh start 0 "results/gpu_log_gomoku_${TAG}.csv"
AZ_TAG="$TAG" \
  AZ_OUT="results/gomoku_${TAG}.json" AZ_CKPT="results/gomoku_ckpt_${TAG}" \
  python scripts/train_rl_gomoku_alphazero.py 2>&1 | tee "results/gomoku_${TAG}.log"
bash scripts/gpu_monitor.sh stop 0 "results/gpu_log_gomoku_${TAG}.csv"

echo "=== done ==="
ls -la results/ | grep "gomoku_${TAG}"
