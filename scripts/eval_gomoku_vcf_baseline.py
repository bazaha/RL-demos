"""Offline VCF-solver baseline vs the saved checkpoints (read-only).

pure-MCTS is retired at 15x15 (12-0 at every playout count from iteration 5:
random rollouts on 225 cells carry no evaluation signal). This baseline is the
opposite kind of opponent: zero positional judgment, perfect shallow tactics.

  VCFPlayer move priority (all sets computed by the trainer's exact checkers):
    1. complete an own five
    2. block the opponent's five (the block that removes every completion)
    3. play a proven victory-by-continuous-fours start (_vcf_starts, depth D)
    4. deny the opponent's double-threat follow-up (_defense_moves), breaking
       ties by the tactical rule prior
    5. otherwise: rule-greedy (rule_priors_batch argmax)

  It never misses a forced win or an unguarded loss within its horizon, so any
  game a checkpoint LOSES to it is a genuine tactical hole (VCF-blindness or
  worse), and the win rate over iterations is an absolute-ish tactical ruler
  that cannot be gamed by style.

The AZ side plays with a small sampling temperature (same convention as the
anchor ladder), so repeated games are distinct and scores carry error bars;
the VCF side is deterministic.

Run in the container:
  AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 python scripts/eval_gomoku_vcf_baseline.py
Env: AZV_CKPT_DIR (results/gomoku_ckpt_p15), AZV_GAMES (12), AZV_SIMS (400),
AZV_TEMP (0.3), AZV_DEPTH (5, VCF stones), AZV_OUT
(results/gomoku_vcf_baseline.json), AZV_DEVICE (cuda:0).
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

CKPT_DIR = os.environ.get("AZV_CKPT_DIR", "results/gomoku_ckpt_p15")
GAMES = int(os.environ.get("AZV_GAMES", "12"))
SIMS = int(os.environ.get("AZV_SIMS", "400"))
TEMP = float(os.environ.get("AZV_TEMP", "0.3"))
DEPTH = int(os.environ.get("AZV_DEPTH", "5"))
OUT = os.environ.get("AZV_OUT", "results/gomoku_vcf_baseline.json")
DEVICE = os.environ.get("AZV_DEVICE", "cuda:0")


class VCFPlayer:
    """Perfect shallow tactics, no positional judgment."""

    def __init__(self, depth=DEPTH):
        self.depth = depth
        self.name = f"vcf-{depth}"

    def _move(self, state):
        b = state.board.copy()
        me = int(state.to_play)
        win = az._win_cells(b, me)
        if win:
            return int(win[0])
        if az._win_cells(b, -me):
            blocks = az._block_five_moves(b, me)
            if blocks:
                return int(blocks[0])
            return int(az._win_cells(b, -me)[0])   # doomed; contest anyway
        vcf = az._vcf_starts(b, me, self.depth)
        if vcf:
            return int(vcf[0])
        # rule prior both picks the fallback move and breaks defense ties
        rp = az.rule_priors_batch(b[None], np.array([me], dtype=np.int8))[0]
        guard = az._defense_moves(b, me)
        if guard:
            return int(max(guard, key=lambda a: rp[a]))
        return int(rp.argmax())

    def move_batch(self, states, rng):
        return [self._move(s) for s in states]


def main():
    import torch
    probes = az.tactical_positions()   # selftest side effect: checkers sane
    del probes
    ckpts = sorted(glob.glob(f"{CKPT_DIR}/iter*.pt"))
    if not ckpts:
        sys.exit(f"no checkpoints under {CKPT_DIR}")
    device = DEVICE if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(20260819)
    net = az.AZNet().to(device)
    vcf = VCFPlayer()
    print(f"[vcf-baseline] {vcf.name} vs {len(ckpts)} checkpoints, "
          f"{GAMES} games each, AZ {SIMS} sims temp {TEMP}, {device}",
          flush=True)

    M = {"opponent": vcf.name, "depth": DEPTH, "games": GAMES, "sims": SIMS,
         "temp": TEMP, "board": az.BOARD, "rows": []}
    for path in ckpts:
        it = int(re.search(r"iter(\d+)\.pt", path).group(1))
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        pl = az.AZPlayer(net, SIMS, device, f"iter{it:03d}", temp=TEMP)
        t0 = time.time()
        wa, wb, dr, games = az.play_matches(pl, vcf, GAMES, rng, "vcf")
        lens = [len(g["moves"]) for g in games]
        # keep one lost game per checkpoint for post-mortem, if any
        lost = [g for g in games if g["result"] == "B"]
        row = {"iter": it, "az_win": wa, "vcf_win": wb, "draw": dr,
               "score": round((wa + 0.5 * dr) / GAMES, 4),
               "avg_len": round(float(np.mean(lens)), 1),
               "seconds": round(time.time() - t0, 1),
               "lost_game": ({"moves": lost[0]["moves"],
                              "az_is_black": lost[0]["a_is_black"]}
                             if lost else None)}
        M["rows"].append(row)
        print(f"  iter{it:03d}  {wa}W-{wb}L-{dr}D  score {row['score']:.2f}  "
              f"len {row['avg_len']}  {row['seconds']}s", flush=True)

    with open(OUT, "w") as f:
        json.dump(M, f)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
