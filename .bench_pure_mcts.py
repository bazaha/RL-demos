"""Local numpy-only benchmark of PureMCTSPlayer cost at 9x9 vs 15x15.

Extracts State/_wins + Node/Tree + PureMCTSPlayer from the trainer (no torch)
and times _move() at the playout counts tonight's run will use.
"""
import math
import re
import time

import numpy as np

SRC = open("scripts/train_rl_gomoku_alphazero.py").read()


def cut(start_pat, end_pat):
    m = re.search(start_pat, SRC)
    e = re.search(end_pat, SRC)
    assert m and e, (start_pat, end_pat)
    return SRC[m.start():e.start()]


seg_rules = cut(r"DIRS = ", r"_CENTER_W = ")
seg_tree = cut(r"class Node:", r"def run_sims")
seg_pure = cut(r"class PureMCTSPlayer", r"class AZPlayer")

for BOARD in (9, 15):
    ns = {"np": np, "math": math, "BOARD": BOARD, "N_IN_ROW": 5,
          "N_ACT": BOARD * BOARD, "C_PUCT": 3.0}
    exec(seg_rules, ns)
    exec(seg_tree, ns)
    exec(seg_pure, ns)
    State = ns["State"]
    Pure = ns["PureMCTSPlayer"]

    # early-game position: 4 stones near the centre (what eval games look like
    # after the AZ side has opened)
    s = State()
    c = BOARD // 2
    for (dr, dc) in ((0, 0), (0, 1), (1, 0), (1, 1)):
        s.play((c + dr) * BOARD + (c + dc))

    rng = np.random.default_rng(0)

    # rollout length + cost
    p400 = Pure(400)
    t0 = time.perf_counter()
    lens = []
    for _ in range(300):
        r = s.clone()
        empty = np.flatnonzero(r.legal_mask())
        rng.shuffle(empty)
        n = 0
        for a in empty:
            r.play(int(a))
            n += 1
            if r.done:
                break
        lens.append(n)
    t_roll = (time.perf_counter() - t0) / 300
    print(f"BOARD={BOARD}: rollout from 4-stone pos: mean len {np.mean(lens):.1f} "
          f"plies, {t_roll*1e3:.3f} ms/rollout")

    for n_pl in ((400, 1000) if BOARD == 9 else (400, 8000)):
        pl = Pure(n_pl)
        reps = 3 if n_pl <= 1000 else 1
        t0 = time.perf_counter()
        for _ in range(reps):
            pl._move(s, rng)
        dt = (time.perf_counter() - t0) / reps
        print(f"  BOARD={BOARD} playouts={n_pl}: {dt:.2f} s/move "
              f"({dt/n_pl*1e3:.3f} ms/playout)")

    # mid-game position (20 stones): rollouts shorter, tree denser
    s2 = State()
    rng2 = np.random.default_rng(7)
    for _ in range(20):
        a = int(rng2.choice(np.flatnonzero(s2.legal_mask())))
        if s2.done:
            break
        s2.play(a)
    if not s2.done:
        n_pl = 400 if BOARD == 9 else 8000
        pl = Pure(n_pl)
        t0 = time.perf_counter()
        pl._move(s2, rng)
        dt = time.perf_counter() - t0
        print(f"  BOARD={BOARD} playouts={n_pl} (20-stone pos): {dt:.2f} s/move")
