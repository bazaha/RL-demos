"""Phase-0 calibration B/C: time one self-play iteration at a given config.

Runs SelfPlayPool.run() once with the trainer's own worker code, so the number
it measures is exactly the per-iteration self-play cost of that config. All
shape knobs come from the usual AZ_* env vars (read by the module at import);
this script adds:

  CAL_CKPT  optional state_dict to load (must match AZ_BOARD/AZ_CH/AZ_BLOCKS);
            without it the net is random-init, which makes games longer than a
            trained net would play -- positions_per_s is the length-robust rate
  CAL_NOTE  free-form label copied into the output row
  CAL_OUT   jsonl file to append the result to
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_rl_gomoku_alphazero as az  # noqa: E402


def main():
    t0 = time.time()
    net = az.AZNet()
    ckpt = os.environ.get("CAL_CKPT", "")
    if ckpt:
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pool = az.SelfPlayPool(az.GPUS, az.CHANNELS, az.BLOCKS)
    try:
        t1 = time.time()
        X, PI, Z, logs, winners, lengths, worker_s = pool.run(
            net, az.GAMES_PER_ITER, az.N_SIMS, az.SEED, 0.0)
        t2 = time.time()
    finally:
        pool.close()

    row = {
        "note": os.environ.get("CAL_NOTE", ""),
        "board": az.BOARD, "channels": az.CHANNELS, "blocks": az.BLOCKS,
        "sims": az.N_SIMS, "workers": len(az.GPUS),
        "games": az.GAMES_PER_ITER,
        "games_per_worker": round(az.GAMES_PER_ITER / len(az.GPUS), 1),
        "ckpt": os.path.basename(ckpt) if ckpt else "random-init",
        "positions": int(len(X)),
        "avg_len": round(float(np.mean(lengths)), 1),
        "max_len": int(max(lengths)),
        "draws": int(sum(1 for w in winners if w == 0)),
        "selfplay_s": round(t2 - t1, 1),
        "startup_s": round(t1 - t0, 1),
        "worker_s_min": min(worker_s), "worker_s_max": max(worker_s),
        "games_per_s": round(az.GAMES_PER_ITER / (t2 - t1), 2),
        "positions_per_s": round(len(X) / (t2 - t1), 1),
    }
    out = os.environ.get("CAL_OUT", "results/calib_throughput.jsonl")
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("[throughput]", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
