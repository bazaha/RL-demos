"""Head-to-head evaluation of the two Gomoku arms from their saved checkpoints.

Why this exists: the fixed baselines in training (random / rule-greedy /
pure-MCTS) and the tactical probes all saturate within ~15 iterations, so from
there on they cannot tell the two arms apart at all. Measuring the arms against
*each other* never saturates, and it is the only thing that actually answers
"did the rule-guided cold start shorten training".

Two measurements, both from checkpoints only -- no retraining:

  matched pairs   pure@k vs rules@k for every shared checkpoint k.
                  Score > 0.5 means the rules arm is ahead at equal compute.

  joint Elo       one Bradley-Terry fit over a round-robin that mixes both
                  arms, so every checkpoint lands on a single scale. Reading
                  the rules arm's Elo off the pure arm's Elo-vs-iteration curve
                  converts "stronger" into "worth N iterations of self-play",
                  which is the shortened-training claim in its measurable form.

Both play randomised openings from both sides rather than repeated games from
the empty board -- AZPlayer is deterministic, so the naive version silently
measures two distinct games and reports them as N. See play_pair().

Pairs are farmed out to worker processes, one CUDA context each, several per
GPU -- each pair is independent, and a single AZ-vs-AZ match leaves most of a
H20 idle.

Run inside the training container, from the workdir:
    AZX_GPUS=0,1,2,3 python scripts/eval_gomoku_cross_arm.py
Writes results/gomoku_cross_arm.json.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

import train_rl_gomoku_alphazero as az

RESULTS = os.environ.get("AZX_RESULTS", "results")
OUT_JSON = os.environ.get("AZX_OUT", f"{RESULTS}/gomoku_cross_arm.json")
ARMS = os.environ.get("AZX_ARMS", "pure,rules").split(",")
GPUS = [int(g) for g in os.environ.get("AZX_GPUS", "0").split(",") if g != ""]
PROCS_PER_GPU = int(os.environ.get("AZX_PROCS_PER_GPU", 2))

# Both measurements are counted in *openings*, not games: one opening is played
# twice, once from each side, and those two games together are one independent
# observation. See play_pair() for why games are not independent.
OPEN_PLIES = int(os.environ.get("AZX_OPEN_PLIES", 2))

# matched pairs: same iteration, both arms. More openings here than the Elo
# round-robin because this is the headline number.
MATCH_OPENINGS = int(os.environ.get("AZX_MATCH_OPENINGS", 16))
MATCH_SIMS = int(os.environ.get("AZX_MATCH_SIMS", 400))

# joint Elo: every checkpoint, fewer openings each, cheaper search
ELO_OPENINGS = int(os.environ.get("AZX_ELO_OPENINGS", 4))
ELO_SIMS = int(os.environ.get("AZX_ELO_SIMS", 200))
ELO_STRIDE = int(os.environ.get("AZX_ELO_STRIDE", 1))  # every Nth checkpoint
SEED = int(os.environ.get("AZX_SEED", 20260725))


def ckpts(arm):
    """[(iter, path)] for one arm, sorted by iteration."""
    d = f"{RESULTS}/gomoku_ckpt_{arm}"
    if not os.path.isdir(d):
        return []
    out = []
    for fn in os.listdir(d):
        if fn.startswith("iter") and fn.endswith(".pt"):
            try:
                out.append((int(fn[4:-3]), os.path.join(d, fn)))
            except ValueError:
                continue
    return sorted(out)


def arm_meta(arm):
    """Net/board config the arm was actually trained with, from its JSON."""
    p = f"{RESULTS}/gomoku_{arm}.json"
    if not os.path.exists(p):
        return None
    with open(p) as f:
        m = json.load(f)
    return {"channels": m["net"]["channels"], "blocks": m["net"]["blocks"],
            "board": m["cfg"]["board"], "n_in_row": m["cfg"]["n_in_row"],
            "beta0": m["cfg"]["beta0"], "seed": m["cfg"]["seed"],
            "iters": m["cfg"]["iters"]}


def check_config(metas):
    """Refuse to run if the checkpoints do not match this module's globals.

    AZNet() and State() read CHANNELS / BLOCKS / BOARD from the trainer's
    module-level globals, which come from AZ_* env vars. A mismatch would load
    state dicts into the wrong shape (loud) or play on the wrong board size
    (silent and wrong), so check both up front.
    """
    for arm, m in metas.items():
        if m is None:
            print(f"!! no results/gomoku_{arm}.json, cannot verify config",
                  file=sys.stderr)
            continue
        want = {"channels": az.CHANNELS, "blocks": az.BLOCKS,
                "board": az.BOARD, "n_in_row": az.N_IN_ROW}
        got = {k: m[k] for k in want}
        if got != want:
            raise SystemExit(
                f"config mismatch for arm {arm}: checkpoints are {got}, this "
                f"process is {want}. Set AZ_BOARD / AZ_CH / AZ_BLOCKS / AZ_NIR "
                f"to match (the run script's values).")


def same_file(pa, pb):
    """True if two checkpoints are byte-identical (same seed → same iter-0)."""
    if os.path.getsize(pa) != os.path.getsize(pb):
        return False
    h = []
    for p in (pa, pb):
        d = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                d.update(chunk)
        h.append(d.hexdigest())
    return h[0] == h[1]


def make_openings(n, rng, plies=OPEN_PLIES):
    """n distinct random openings, each a list of moves from the empty board.

    Two plies is enough to break the determinism described in play_pair without
    dragging the nets far off the distribution they trained on. 9x9 has 81*80
    two-ply openings so collisions are unlikely, but dedupe anyway.
    """
    seen, out = set(), []
    guard = 0
    while len(out) < n and guard < 1000 * n:
        guard += 1
        s = az.State()
        mv = []
        for _ in range(plies):
            m = int(rng.choice(np.flatnonzero(s.legal_mask())))
            s.play(m)
            mv.append(m)
        if s.done:                      # cannot happen at 2 plies; cheap guard
            continue
        key = tuple(mv)
        if key in seen:
            continue
        seen.add(key)
        out.append(mv)
    if len(out) < n:
        raise SystemExit(f"could not build {n} distinct {plies}-ply openings")
    return out


def play_pair(pA, pB, openings, sims_rng):
    """Play every opening twice, colours swapped. -> (wa, wb, dr, pair_scores).

    az.play_matches is not usable for AlphaZero vs AlphaZero. AZPlayer.move_batch
    takes the argmax of the visit counts with no root noise (it ignores the rng
    it is handed), so a move is a pure function of the position: every game that
    starts from the empty board with the same colour assignment is identical.
    Asking play_matches for 24 games really buys *two* distinct games replayed
    12 times each, which makes the score look sqrt(12) times more precise than
    it is -- the 0-0-24 and 0-12-12 results that exposed this are exactly that
    signature. Randomised openings give genuinely independent games, and playing
    each opening from both sides cancels its own first-player bias.

    pair_scores[i] is arm B's score over opening i across both colours, so each
    entry is one independent observation in {0, 0.25, 0.5, 0.75, 1}.
    """
    states, meta = [], []
    for oi, mv in enumerate(openings):
        for a_black in (True, False):
            s = az.State()
            for m in mv:
                s.play(m)
            states.append(s)
            meta.append((oi, a_black))
    active = [i for i, s in enumerate(states) if not s.done]
    while active:
        # a_idx: games where it is arm A's turn, i.e. the side to play is the
        # colour A was assigned. Same lockstep structure as az.play_matches.
        a_idx = [i for i in active if (states[i].to_play == 1) == meta[i][1]]
        a_set = set(a_idx)
        b_idx = [i for i in active if i not in a_set]
        for player, idxs in ((pA, a_idx), (pB, b_idx)):
            if not idxs:
                continue
            mv = player.move_batch([states[i] for i in idxs], sims_rng)
            for i, m in zip(idxs, mv):
                states[i].play(m)
        active = [i for i in active if not states[i].done]

    wa = wb = dr = 0
    per = {}
    for i, s in enumerate(states):
        oi, a_black = meta[i]
        w = s.winner
        if w == 0:
            dr += 1
            sb = 0.5
        elif (w == 1) == a_black:
            wa += 1
            sb = 0.0
        else:
            wb += 1
            sb = 1.0
        per.setdefault(oi, []).append(sb)
    pair_scores = [round(sum(v) / len(v), 4) for _, v in sorted(per.items())]
    return wa, wb, dr, pair_scores


def _play(job):
    """One match in a worker process. job = (idx, pathA, pathB, openings, sims, gpu, seed)."""
    idx, pa, pb, openings, sims, gpu, seed = job
    dev = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")
    nets = []
    for p in (pa, pb):
        m = az.AZNet().to(dev)
        m.load_state_dict(torch.load(p, map_location=dev))
        m.eval()
        nets.append(az.AZPlayer(m, sims, dev))
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        wa, wb, dr, ps = play_pair(nets[0], nets[1], openings, rng)
    return idx, wa, wb, dr, ps


def run_jobs(jobs, label):
    """Farm jobs out over GPUs; returns {idx: (wa, wb, dr, pair_scores)}."""
    if not jobs:
        return {}
    n_proc = min(len(jobs), max(1, len(GPUS) * PROCS_PER_GPU))
    t0 = time.time()
    print(f"[{label}] {len(jobs)} matches on {n_proc} workers "
          f"over GPUs {GPUS}", flush=True)
    out = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(n_proc) as pool:
        for k, (idx, wa, wb, dr, ps) in enumerate(pool.imap_unordered(_play, jobs), 1):
            out[idx] = (wa, wb, dr, ps)
            print(f"  [{label} {k}/{len(jobs)}] {idx}: {wa}-{wb}-{dr} "
                  f"({time.time() - t0:.0f}s elapsed)", flush=True)
    print(f"[{label}] done in {time.time() - t0:.0f}s", flush=True)
    return out


def interp_iter(curve, target):
    """Iteration at which `curve` [(iter, elo)] first reaches `target` Elo.

    Uses the running max so the answer is monotonic in target -- a checkpoint
    that happens to dip does not make a higher bar look reached earlier.
    Linear interpolation between the bracketing checkpoints; None if the curve
    never gets there.
    """
    pts, best = [], -float("inf")
    for it, e in curve:
        best = max(best, e)
        pts.append((it, best))
    if not pts or target <= pts[0][1]:
        return pts[0][0] if pts else None
    for (i0, e0), (i1, e1) in zip(pts, pts[1:]):
        if e1 >= target:
            if e1 == e0:
                return i1
            return round(i0 + (i1 - i0) * (target - e0) / (e1 - e0), 1)
    return None


def main():
    arms = {a: ckpts(a) for a in ARMS}
    arms = {a: c for a, c in arms.items() if c}
    if len(arms) < 2:
        raise SystemExit(f"need 2 arms with checkpoints, found {list(arms)}")
    a1, a2 = list(arms)[0], list(arms)[1]
    print(f"[init] {a1}: {len(arms[a1])} ckpts | {a2}: {len(arms[a2])} ckpts",
          flush=True)
    metas = {a: arm_meta(a) for a in arms}
    check_config(metas)

    M = {"arms": [a1, a2], "board": az.BOARD, "n_in_row": az.N_IN_ROW,
         "net": {"channels": az.CHANNELS, "blocks": az.BLOCKS},
         "arm_meta": metas,
         "cfg": {"match_openings": MATCH_OPENINGS, "match_sims": MATCH_SIMS,
                 "match_games": MATCH_OPENINGS * 2,
                 "elo_openings": ELO_OPENINGS, "elo_sims": ELO_SIMS,
                 "elo_games": ELO_OPENINGS * 2,
                 "open_plies": OPEN_PLIES,
                 "elo_stride": ELO_STRIDE, "seed": SEED},
         "started": time.time()}

    # One fixed opening set, reused by every matched pair, so differences along
    # the iteration axis are not confounded by different starting positions.
    orng = np.random.default_rng(SEED)
    match_open = make_openings(MATCH_OPENINGS, orng)
    elo_open = make_openings(ELO_OPENINGS, orng)
    M["openings"] = {"match": match_open, "elo": elo_open}
    print(f"[init] {MATCH_OPENINGS} matched openings, {ELO_OPENINGS} elo "
          f"openings, {OPEN_PLIES} plies each", flush=True)

    # ---- matched pairs: same iteration, arm vs arm ----------------------- #
    d1, d2 = dict(arms[a1]), dict(arms[a2])
    shared = sorted(set(d1) & set(d2))
    # iter 0 is the same random init in both arms (identical seed), so a match
    # there is self-play against a byte-identical opponent -- no information.
    skipped = [it for it in shared if same_file(d1[it], d2[it])]
    shared = [it for it in shared if it not in skipped]
    if skipped:
        print(f"[matched] skipping identical checkpoints at iters {skipped}",
              flush=True)
    M["matched_skipped_identical"] = skipped
    jobs = [(f"m{it}", d1[it], d2[it], match_open, MATCH_SIMS,
             GPUS[i % len(GPUS)], SEED + it)
            for i, it in enumerate(shared)]
    res = run_jobs(jobs, "matched")
    M["matched"] = []
    for it in shared:
        wa, wb, dr, ps = res.get(f"m{it}", (0, 0, 0, []))
        n = wa + wb + dr
        M["matched"].append({
            "iter": it, "games": n, "openings": len(ps),
            f"{a1}_win": wa, f"{a2}_win": wb, "draw": dr,
            # score from arm 2's point of view: > 0.5 means arm 2 is ahead
            "score": round((wb + 0.5 * dr) / n, 4) if n else None,
            # one entry per opening (both colours) -- the independent unit
            "pair_scores": ps})
        print(f"  iter {it:3d}: {a1} {wa} - {a2} {wb} - draw {dr}", flush=True)

    # ---- joint Elo over a mixed round-robin ----------------------------- #
    # One player list mixing both arms, so a single Bradley-Terry fit puts every
    # checkpoint on the same scale. The shared random init appears once (it is
    # byte-identical across arms) and anchors the scale at 0.
    grid = [it for k, it in enumerate(sorted(d1)) if k % ELO_STRIDE == 0]
    if sorted(d1)[-1] not in grid:
        grid.append(sorted(d1)[-1])
    players = [{"arm": a1, "iter": it, "path": d1[it]} for it in grid]
    players += [{"arm": a2, "iter": it, "path": d2[it]}
                for it in grid if it in d2 and it not in skipped]
    print(f"[elo] {len(players)} players, "
          f"{len(players) * (len(players) - 1) // 2} pairs", flush=True)
    jobs, pair_ix = [], []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            jobs.append((f"e{i}_{j}", players[i]["path"], players[j]["path"],
                         elo_open, ELO_SIMS, GPUS[len(jobs) % len(GPUS)],
                         SEED + 1000 + len(jobs)))
            pair_ix.append((i, j))
    res = run_jobs(jobs, "elo")
    rr = []
    for (i, j) in pair_ix:
        wa, wb, dr, _ = res.get(f"e{i}_{j}", (0, 0, 0, []))
        rr.append((i, j, wa, wb, dr))
    elo = az.fit_elo(len(players), rr)
    # anchor at the shared random-init checkpoint so both arms read off 0
    M["elo"] = [{"arm": p["arm"], "iter": p["iter"], "elo": elo[k]}
                for k, p in enumerate(players)]
    M["elo_pairs"] = [{"i": i, "j": j, "wi": wi, "wj": wj, "d": d}
                      for i, j, wi, wj, d in rr]

    # ---- convert Elo into "worth N iterations of the other arm" ---------- #
    curve1 = [(e["iter"], e["elo"]) for e in M["elo"] if e["arm"] == a1]
    M["equivalence"] = []
    for e in M["elo"]:
        if e["arm"] != a2:
            continue
        eq = interp_iter(curve1, e["elo"])
        # arm 2 reached this strength in e["iter"] iterations; arm 1 needed eq.
        # Positive = arm 2 got there sooner, i.e. iterations saved.
        M["equivalence"].append({
            "iter": e["iter"], "elo": e["elo"], f"{a1}_iters_equivalent": eq,
            "iters_saved": round(eq - e["iter"], 1) if eq is not None else None})
        print(f"  {a2}@{e['iter']} (Elo {e['elo']}) ≈ {a1}@{eq}", flush=True)

    M["seconds"] = round(time.time() - M["started"], 1)
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(M, f)
    os.replace(tmp, OUT_JSON)
    print(f"[done] wrote {OUT_JSON} in {M['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
