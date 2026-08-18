"""Phase-0 calibration A: draw rate vs search depth for a trained checkpoint.

Self-play (same net on both sides), deterministic argmax moves with no root
noise, from CAL_GAMES *distinct* random CAL_OPEN_PLIES-ply openings. Play is
deterministic, so one opening yields exactly one distinct game and each game
is an independent observation of "under near-best play, is this drawn?".

The question this answers: is the 9x9 endgame draw collapse a property of the
game at this strength (draw rate stays high as sims grow -> switch boards), or
an artifact of shallow search (draw rate falls -> deepen search first)?

Env: the usual AZ_* (board/net must match the checkpoint), plus
  CAL_CKPT (required), CAL_SIMS, CAL_GAMES, CAL_OPEN_PLIES, CAL_SEED,
  CAL_DEV (e.g. cuda:4), CAL_OUT (jsonl, appended)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_rl_gomoku_alphazero as az  # noqa: E402

CKPT = os.environ["CAL_CKPT"]
SIMS = int(os.environ.get("CAL_SIMS", "400"))
GAMES = int(os.environ.get("CAL_GAMES", "64"))
OPEN_PLIES = int(os.environ.get("CAL_OPEN_PLIES", "2"))
SEED = int(os.environ.get("CAL_SEED", "20260731"))
DEV = os.environ.get("CAL_DEV", "cuda:0")
OUT = os.environ.get("CAL_OUT", "results/calib_draw_vs_sims.jsonl")


def main():
    net = az.AZNet().to(DEV)
    net.load_state_dict(torch.load(CKPT, map_location=DEV))
    net.eval()

    rng = np.random.default_rng(SEED)
    states, seen = [], set()
    while len(states) < GAMES:
        s = az.State()
        for _ in range(OPEN_PLIES):
            s.play(int(rng.choice(np.flatnonzero(s.legal_mask()))))
        key = s.board.tobytes()
        if key in seen:
            continue
        seen.add(key)
        states.append(s)

    trees = [az.Tree(s) for s in states]
    active = list(range(GAMES))
    t0 = time.time()
    while active:
        az.run_sims(net, [trees[i] for i in active], SIMS, DEV,
                    noise_rng=None, beta=0.0)
        for i in active:
            t = trees[i]
            t.advance(int(t.root.N.argmax()))
        active = [i for i in active if not trees[i].root.state.done]
    dt = time.time() - t0

    winners = [t.root.state.winner for t in trees]
    lengths = [t.root.state.n_moves for t in trees]  # includes opening plies
    n_draw = sum(1 for w in winners if w == 0)
    row = {
        "ckpt": os.path.basename(CKPT), "sims": SIMS, "games": GAMES,
        "open_plies": OPEN_PLIES, "seed": SEED,
        "black": sum(1 for w in winners if w == 1),
        "white": sum(1 for w in winners if w == -1),
        "draws": n_draw,
        "draw_rate": round(n_draw / GAMES, 3),
        "avg_len": round(float(np.mean(lengths)), 1),
        "max_len": int(max(lengths)),
        "seconds": round(dt, 1),
    }
    with open(OUT, "a") as f:
        f.write(json.dumps(row) + "\n")
    print("[draw-vs-sims]", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
