#!/usr/bin/env bash
# AlphaZero Gomoku on node09: runs two arms back to back with identical config
# and seed, differing only in the rule-guided cold start (AZ_BETA0).
#
#   arm A  pure    beta0=0     plain AlphaZero from random init
#   arm B  rules   beta0=0.6   tactical prior mixed into MCTS, annealed to 0
#
# Everything else is held fixed so the comparison is about the cold start only.
# Run from the workdir, which is mounted at /workspace.
set -euo pipefail
cd /workspace

export AZ_BOARD=${AZ_BOARD:-9}
export AZ_CH=${AZ_CH:-128}
export AZ_BLOCKS=${AZ_BLOCKS:-8}
export AZ_ITERS=${AZ_ITERS:-50}
export AZ_GAMES=${AZ_GAMES:-384}
export AZ_SIMS=${AZ_SIMS:-400}
export AZ_TRAIN_STEPS=${AZ_TRAIN_STEPS:-500}
export AZ_BATCH=${AZ_BATCH:-512}
export AZ_GPUS=${AZ_GPUS:-0,0,1,1,2,2,3,3}
export AZ_EVAL_EVERY=${AZ_EVAL_EVERY:-5}
export AZ_EVAL_SIMS=${AZ_EVAL_SIMS:-400}
export AZ_EVAL_GAMES=${AZ_EVAL_GAMES:-12}
export AZ_EVAL_GAMES_RAND=${AZ_EVAL_GAMES_RAND:-8}
export AZ_ELO_SIMS=${AZ_ELO_SIMS:-200}
export AZ_ELO_GAMES=${AZ_ELO_GAMES:-6}
export AZ_SEED=${AZ_SEED:-42}
export AZ_RULE_ITERS=${AZ_RULE_ITERS:-20}

echo "=== env ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
PY

run_arm () {  # $1 = tag, $2 = beta0
  local tag="$1" beta0="$2"
  echo "=== arm ${tag}: beta0=${beta0} ==="
  bash scripts/gpu_monitor.sh start 0 "results/gpu_log_gomoku_${tag}.csv"
  AZ_BETA0="$beta0" AZ_TAG="$tag" \
    AZ_OUT="results/gomoku_${tag}.json" AZ_CKPT="results/gomoku_ckpt_${tag}" \
    python scripts/train_rl_gomoku_alphazero.py 2>&1 | tee "results/gomoku_${tag}.log"
  bash scripts/gpu_monitor.sh stop 0 "results/gpu_log_gomoku_${tag}.csv"
}

run_arm pure  0.0
run_arm rules 0.6

echo "=== done ==="
ls -la results/ | grep gomoku
