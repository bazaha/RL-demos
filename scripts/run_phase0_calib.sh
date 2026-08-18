#!/usr/bin/env bash
# Phase-0 calibration, run inside the pytorch:25.11 container on node09.
#
#   B. self-play throughput matrix at 9x9: workers x games-per-worker
#      (needs 7 free GPUs, so it runs first, sequentially)
#   A. draw rate vs search depth on the trained 9x9 checkpoint
#      (3 single-GPU processes on GPUs 4/6/7, in the background)
#   C. one self-play iteration at the 15x15 Phase-1 candidate config,
#      two batch sizes (GPUs 0-3, concurrent with A)
#
# GPU 5 is avoided throughout: someone else's job holds 49 GB on it.
# Total expected wall time ~35-45 min.
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=results/gomoku_ckpt_pure/iter050.pt
[ -f "$CKPT" ] || { echo "missing $CKPT -- run the training demo first"; exit 1; }
mkdir -p results
: > results/calib_throughput.jsonl
: > results/calib_draw_vs_sims.jsonl

echo "=== B. throughput matrix (9x9, iter050 weights, 400 sims) ==="
b_point() { # gpu-list games note
  AZ_BOARD=9 AZ_CH=128 AZ_BLOCKS=8 AZ_SIMS=400 AZ_SEED=42 \
  AZ_GPUS="$1" AZ_GAMES="$2" CAL_CKPT="$CKPT" CAL_NOTE="$3" \
  CAL_OUT=results/calib_throughput.jsonl \
    python scripts/calib_selfplay_point.py
}
W8=0,0,1,1,2,2,3,3
W28=0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,6,6,6,6,7,7,7,7
b_point $W8  384  "9x9 8w x 48g (current config)"
b_point $W8  1536 "9x9 8w x 192g"
b_point $W28 1344 "9x9 28w x 48g"
b_point $W28 5376 "9x9 28w x 192g (Phase-1 worker/batch shape)"

echo "=== A. draw vs sims (bg, GPUs 4/6/7)  +  C. 15x15 probe (fg, GPUs 0-3) ==="
a_point() { # sims games dev
  AZ_BOARD=9 AZ_CH=128 AZ_BLOCKS=8 \
  CAL_CKPT="$CKPT" CAL_SIMS="$1" CAL_GAMES="$2" CAL_DEV="$3" \
  CAL_OUT=results/calib_draw_vs_sims.jsonl \
    python scripts/calib_draw_vs_sims.py
}
a_point 400  64 cuda:4 & A1=$!
a_point 1600 64 cuda:6 & A2=$!
a_point 6400 32 cuda:7 & A3=$!

c_point() { # games note
  AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 AZ_SIMS=800 AZ_SEED=42 \
  AZ_GPUS=0,1,2,3 AZ_GAMES="$1" CAL_NOTE="$2" \
  CAL_OUT=results/calib_throughput.jsonl \
    python scripts/calib_selfplay_point.py
}
c_point 192 "15x15 192ch/12blk 800sims 4w x 48g random-init"
c_point 768 "15x15 192ch/12blk 800sims 4w x 192g random-init"

wait $A1 $A2 $A3
echo "=== phase-0 calibration done ==="
echo "--- results/calib_throughput.jsonl ---"
cat results/calib_throughput.jsonl
echo "--- results/calib_draw_vs_sims.jsonl ---"
cat results/calib_draw_vs_sims.jsonl
