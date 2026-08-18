#!/usr/bin/env bash
# Head-to-head eval of the two finished Gomoku arms, from checkpoints only.
#
# Runs after run_gomoku_in_container.sh has produced both arms. The board/net
# geometry must match what the arms were trained with, because AZNet() and
# State() read it from module-level globals -- eval_gomoku_cross_arm.py verifies
# this against each arm's JSON and refuses to run on a mismatch, so keep these
# in sync with run_gomoku_in_container.sh.
#
# Run from the workdir, which is mounted at /workspace.
#
# Sizing (measured on node09, 9x9 / 128ch / 8 blocks): a matched pair at 16
# openings x 400 sims takes ~115 s, an Elo pair at 6 openings x 200 sims ~80 s
# between two similarly-strong checkpoints (a blowout finishes in under 10 s).
# 11 + 10 checkpoints means 210 Elo pairs, so this is Elo-bound: ~22 min on 14
# workers. GPU 5 is left out because another tenant is using it.
set -euo pipefail
cd /workspace

export AZ_BOARD=${AZ_BOARD:-9}
export AZ_CH=${AZ_CH:-128}
export AZ_BLOCKS=${AZ_BLOCKS:-8}
export AZ_NIR=${AZ_NIR:-5}

export AZX_GPUS=${AZX_GPUS:-0,1,2,3,4,6,7}
export AZX_PROCS_PER_GPU=${AZX_PROCS_PER_GPU:-2}
export AZX_ELO_OPENINGS=${AZX_ELO_OPENINGS:-6}

echo "=== cross-arm eval: $(date -u +%FT%TZ) ==="
python scripts/eval_gomoku_cross_arm.py 2>&1 | tee results/gomoku_cross_arm.log
echo "=== done ==="
ls -la results/gomoku_cross_arm.json
