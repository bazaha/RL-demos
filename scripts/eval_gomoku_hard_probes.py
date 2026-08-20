"""Offline hard tactical probes over saved checkpoints (read-only).

The in-training probes saturated at iteration 5 on all tiers: their correct
answers all lie ON the most salient line, so a policy that merely attends to
the longest line solves them without calculating anything. Each hard probe
here separates salience from correctness -- it places a tempting line on the
board whose extension is wrong, and puts the winning/saving move elsewhere:

  HD1  先防后攻    own OPEN three begs to be extended, but the opponent has a
                   four: the only correct move is the block. The hardened
                   checker proves extending is not even a double threat
                   (the opponent's five lands first).
  HD2  识别跳三    the opponent's open jump-three must be defused (gap or
                   either outer end); own salient three is dead (both
                   relevant ends blocked) so extending it is pure salience.
  HV1  选对冲四    two look-alike half-open threes; both extensions make a
                   four, but only one continues into a VCF-2 win. Pure
                   calculation: local pattern identical, continuation differs.
  HV2  冲四诱饵    the flashy four is REFUTED (its forced block completes an
                   opponent four), the quiet crossing move wins by double
                   threat. The hardened checkers prove the refutation.

Good sets come from the same exact checkers the trainer uses
(_block_five_moves/_defense_moves/_vcf_starts/_double_threat_moves); every
probe asserts its traps are NOT in the good set at construction time.

Run in the container (read-only over checkpoints):
  AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 python scripts/eval_gomoku_hard_probes.py
Env: AZH_CKPT_DIR (default results/gomoku_ckpt_p15), AZH_SIMS (default 800,
matching the training eval), AZH_OUT (default results/gomoku_hard_probes.json),
AZH_DEVICE (default cuda:0), AZH_ARM_JSON for the config cross-check.
"""
import glob
import json
import os
import re
import sys
import time

os.environ.setdefault("AZ_BOARD", "15")
os.environ.setdefault("AZ_CH", "192")
os.environ.setdefault("AZ_BLOCKS", "12")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import train_rl_gomoku_alphazero as az  # noqa: E402

CKPT_DIR = os.environ.get("AZH_CKPT_DIR", "results/gomoku_ckpt_p15")
SIMS = int(os.environ.get("AZH_SIMS", "800"))
OUT = os.environ.get("AZH_OUT", "results/gomoku_hard_probes.json")
DEVICE = os.environ.get("AZH_DEVICE", "cuda:0")
ARM_JSON = os.environ.get("AZH_ARM_JSON", "results/gomoku_p15.json")


def hard_probes():
    """Constructed 15x15+ positions; returns list of probe dicts."""
    assert az.BOARD >= 15, "hard probes are laid out for a 15x15 board"

    def tr(cells):
        return [(c, r) for r, c in cells]

    base = [
        {"id": "hd1_defend_first", "family": "HD1", "kind": "defend",
         "desc": "己方活三很诱人，但白方已有冲四 —— 唯一正解是封堵",
         "black": [(7, 5), (7, 6), (7, 7), (11, 3)],
         "white": [(11, 4), (11, 5), (11, 6), (11, 7)],
         "traps": [(7, 4), (7, 8)],
         "check": lambda b: az._block_five_moves(b, 1),
         "expect": [(11, 8)]},
        {"id": "hd2_jump_three", "family": "HD2", "kind": "defend",
         "desc": "白方开放跳三必须拆解；己方三连两头被堵，延伸是纯显眼诱饵",
         "black": [(10, 6), (10, 7), (10, 8), (13, 0), (13, 13)],
         "white": [(4, 5), (4, 6), (4, 8), (10, 5), (10, 10)],
         "traps": [(10, 9)],
         "check": lambda b: az._defense_moves(b, 1),
         "expect": [(4, 4), (4, 7), (4, 9)]},
        {"id": "hv1_dead_four", "family": "HV1", "kind": "attack",
         "desc": "盘面最长的线是两头封死的死四（显眼但零价值）；唯一胜着是短线上的 VCF-2 起手",
         "black": [(2, 3), (2, 4), (2, 5), (2, 6), (8, 3), (8, 4), (8, 5),
                   (9, 6), (10, 6)],
         "white": [(2, 2), (2, 7), (8, 2), (14, 0), (14, 2), (14, 4),
                   (14, 10), (14, 12), (0, 14)],
         "traps": [(1, 4), (3, 5)],   # hugging the dead four
         "check": lambda b: az._vcf_starts(b, 1, 3),
         "expect": [(8, 6)]},
        {"id": "hv2_poisoned_four", "family": "HV2", "kind": "attack",
         "desc": "显眼的冲四被毒化（对方的挡子恰好连出己方冲四）；胜着是安静的双威胁交叉点,或先占毒点的保先冲四",
         "black": [(12, 8), (12, 9), (12, 10), (2, 10), (3, 10), (4, 10),
                   (5, 7), (5, 8), (5, 9)],
         "white": [(12, 7), (1, 10), (5, 6), (9, 12), (10, 12), (11, 12),
                   (14, 0), (14, 2), (0, 0)],
         "traps": [(12, 11)],
         # ALL forced wins, not just the designed one: the iter-25+ nets found
         # a second correct answer -- occupy the poison square (12,12) itself,
         # which is a tempo-keeping split four -- and a good set narrower than
         # "every objective win" misgrades the smarter move (learned the hard
         # way: this probe first shipped with _double_threat_moves only)
         "check": lambda b: az._vcf_starts(b, 1, 3),
         "expect": [(5, 10), (12, 12)]},
    ]

    pos = []
    for spec in base:
        for flip, xf in (("", lambda x: x), ("_tr", tr)):
            black, white = xf(spec["black"]), xf(spec["white"])
            assert len(black) == len(white), spec["id"]
            cells = black + white
            assert len(set(cells)) == len(cells), f"{spec['id']}: overlap"
            s = az._mk(black, white)
            good = sorted(int(a) for a in spec["check"](s.board))
            traps = [r * az.BOARD + c for r, c in xf(spec["traps"])]
            expect = sorted(r * az.BOARD + c for r, c in xf(spec["expect"]))
            # construction-time proofs: the probe tests what it claims to test
            assert not s.done and s.to_play == 1
            assert good == expect, (spec["id"] + flip, good, expect)
            assert not set(traps) & set(good), (spec["id"] + flip, "trap in good")
            assert not az._win_cells(s.board, 1), (spec["id"], "black wins now")
            if spec["family"] in ("HD1", "HD2"):
                # the tempting extension must genuinely fail the defense test
                assert not az._double_threat_moves(s.board, 1) or \
                    spec["family"] == "HD2", spec["id"]
            pos.append({"id": spec["id"] + flip, "family": spec["family"],
                        "kind": spec["kind"], "desc": spec["desc"],
                        "state": s, "good": good, "traps": traps})
    return pos


def check_config():
    """Board mismatch would silently probe garbage positions; catch it early.
    (Channel/block mismatches fail loudly at state_dict load time.)"""
    if not os.path.exists(ARM_JSON):
        print(f"!! {ARM_JSON} missing, skipping config cross-check",
              file=sys.stderr)
        return
    with open(ARM_JSON) as f:
        cfg = json.load(f)["cfg"]
    if cfg["board"] != az.BOARD:
        sys.exit(f"config mismatch: checkpoints are {cfg['board']}x"
                 f"{cfg['board']}, AZ_BOARD={az.BOARD}")


def main():
    import torch
    check_config()
    probes = hard_probes()
    legacy = az.tactical_positions()
    ckpts = sorted(glob.glob(f"{CKPT_DIR}/iter*.pt"))
    if not ckpts:
        sys.exit(f"no checkpoints under {CKPT_DIR}")
    print(f"[hard-probes] {len(probes)} hard + {len(legacy)} legacy probes, "
          f"{len(ckpts)} checkpoints, {SIMS} sims, {DEVICE}", flush=True)

    device = DEVICE if torch.cuda.is_available() else "cpu"
    net = az.AZNet().to(device)
    M = {"sims": SIMS, "ckpt_dir": CKPT_DIR, "board": az.BOARD,
         "probe_meta": [{"id": p["id"], "family": p["family"],
                         "kind": p["kind"], "desc": p["desc"],
                         "good": p["good"], "traps": p["traps"],
                         "board_cells": [int(v) for v in
                                         p["state"].board.reshape(-1)]}
                        for p in probes],
         "checkpoints": []}

    for path in ckpts:
        it = int(re.search(r"iter(\d+)\.pt", path).group(1))
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        pl = az.AZPlayer(net, SIMS, device)
        t0 = time.time()
        rec = {"iter": it, "hard": [], "legacy_raw": 0, "legacy_mcts": 0}
        for p in probes:
            pol, vis, val = pl.policy_and_value(p["state"])
            ra, ma = int(pol.argmax()), int(vis.argmax())
            rec["hard"].append({
                "id": p["id"], "raw_move": ra, "mcts_move": ma,
                "raw_ok": ra in p["good"], "mcts_ok": ma in p["good"],
                "raw_trap": ra in p["traps"], "mcts_trap": ma in p["traps"],
                "value": round(val, 4)})
        for p in legacy:
            pol, vis, _ = pl.policy_and_value(p["state"])
            rec["legacy_raw"] += int(pol.argmax()) in p["good"]
            rec["legacy_mcts"] += int(vis.argmax()) in p["good"]
        rec["legacy_n"] = len(legacy)
        rec["seconds"] = round(time.time() - t0, 1)
        M["checkpoints"].append(rec)
        h = rec["hard"]
        fam = {}
        for r, p in zip(h, probes):
            f = fam.setdefault(p["family"], [0, 0, 0])
            f[0] += 1
            f[1] += r["raw_ok"]
            f[2] += r["mcts_ok"]
        fams = " ".join(f"{k}:{v[1]}/{v[2]}" for k, v in sorted(fam.items()))
        print(f"  iter{it:03d}  hard raw {sum(r['raw_ok'] for r in h):2d}/"
              f"{len(h)}  mcts {sum(r['mcts_ok'] for r in h):2d}/{len(h)}  "
              f"traps r/m {sum(r['raw_trap'] for r in h)}/"
              f"{sum(r['mcts_trap'] for r in h)}  "
              f"[{fams}] (raw/mcts)  legacy {rec['legacy_raw']}/"
              f"{rec['legacy_mcts']}/{rec['legacy_n']}  "
              f"{rec['seconds']}s", flush=True)

    with open(OUT, "w") as f:
        json.dump(M, f)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
