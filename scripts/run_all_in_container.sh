#!/usr/bin/env bash
# Orchestrates the full H20 validation inside the container on node09.
# Run from the workdir (~/h20_validation_20260724) which is mounted at /workspace.
set -euo pipefail
cd /workspace

echo "=== env check ==="
python - <<'PY'
import torch, torchvision, gymnasium
print("torch", torch.__version__, "| cuda", torch.version.cuda,
      "| gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0))
print("torchvision", torchvision.__version__, "| gymnasium", gymnasium.__version__)
PY

echo "=== prep CIFAR-10 ==="
if [ ! -f data/cifar10.npz ]; then python scripts/prep_cifar10.py data; fi

echo "=== DL training: ResNet-18 / CIFAR-10 ==="
bash scripts/gpu_monitor.sh start
python scripts/train_dl_cifar10.py
bash scripts/gpu_monitor.sh stop
mv results/gpu_log.csv results/gpu_log_dl.csv

echo "=== RL training: DQN / CartPole-v1 ==="
bash scripts/gpu_monitor.sh start
python scripts/train_rl_cartpole.py
bash scripts/gpu_monitor.sh stop
mv results/gpu_log.csv results/gpu_log_rl.csv

echo "=== done ==="
ls -la results/
