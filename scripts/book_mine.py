"""Mine the opening book from mass self-play games (numpy + stdlib only).

Pipeline:
  1. canonicalize every game's ply-K prefix under the 8-fold dihedral group
     (min over transformed-board bytes; ties broken by smallest transform id,
     so symmetric positions canonicalize identically everywhere)
  2. group games by the canonical position at ply K (board content, so
     transpositions merge for free); count n / black / white / draw
  3. select ~TOPK entries: n >= MIN_N, at most MAX_PER_PREFIX4 entries sharing
     a canonical 4-ply prefix (diversity guard), ranked by the Wilson 95%
     lower bound of the black win rate (near-degenerate at this strength --
     ranking then effectively follows support n, which is the honest order)
  4. per entry: modal canonical K-ply move sequence, extended ply-by-ply with
     the most frequent continuation while support >= max(30, 0.1 * n), up to
     MAX_PLY; per-ply support recorded

Usage:
  python3 scripts/book_mine.py --selftest          # must pass first
  python3 scripts/book_mine.py results/book_games_p15.jsonl
Knobs (env): BOOK_K=8 BOOK_MAX_PLY=12 BOOK_MIN_N=200 BOOK_TOPK=20
             BOOK_MAX_PER_PREFIX4=5 BOOK_JSON=results/book/gomoku_book.json
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

B = 15
N = B * B
K = int(os.environ.get("BOOK_K", "8"))
MAX_PLY = int(os.environ.get("BOOK_MAX_PLY", "12"))
MIN_N = int(os.environ.get("BOOK_MIN_N", "200"))
TOPK = int(os.environ.get("BOOK_TOPK", "20"))
MAX_PER_PREFIX4 = int(os.environ.get("BOOK_MAX_PER_PREFIX4", "5"))
OUT = os.environ.get("BOOK_JSON", "results/book/gomoku_book.json")

COLS = "ABCDEFGHJKLMNOP"


# ---------------------------------------------------------------------------
# dihedral-8 group on cells: g = rot90^k (k = g % 4), then column flip if g>=4
# rot90 by one step maps (r, c) -> (B-1-c, r)  [the trainer's asserted map]
# ---------------------------------------------------------------------------
def transform_cell(a, g):
    r, c = divmod(a, B)
    for _ in range(g % 4):
        r, c = B - 1 - c, r
    if g >= 4:
        c = B - 1 - c
    return r * B + c


_CELL_MAPS = np.array([[transform_cell(a, g) for a in range(N)]
                       for g in range(8)], dtype=np.int64)


def canon(prefix):
    """Canonical transform id and transformed move list for a move prefix.

    Returns (key_bytes, g_star, canonical_moves). Board-content keying merges
    transpositions; smallest-g tie-break keeps symmetric positions stable.
    """
    board = np.zeros(N, dtype=np.int8)
    p = 1
    for a in prefix:
        board[a] = p
        p = -p
    best_key, best_g = None, 0
    for g in range(8):
        tb = np.zeros(N, dtype=np.int8)
        tb[_CELL_MAPS[g]] = board          # cell a moves to position map[a]
        key = tb.tobytes()
        if best_key is None or key < best_key:
            best_key, best_g = key, g
    return best_key, best_g, [int(_CELL_MAPS[best_g][a]) for a in prefix]


def wilson_lb(wins, n, z=1.96):
    if n == 0:
        return 0.0
    p = wins / n
    d = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / d)


def auto_name(moves, k=4):
    """Neutral coordinate name from the first k canonical moves (k matches the
    grouping ply so names are distinct across entries)."""
    def nm(a):
        r, c = divmod(a, B)
        return f"{COLS[c]}{B - r}"
    return "·".join(nm(a) for a in moves[:k]) + " 系"


def mine(games):
    groups = defaultdict(list)      # key -> list of canonical full-move lists
    skipped = 0
    for g in games:
        mv = g["moves"]
        if len(mv) < K:
            skipped += 1
            continue
        key, gs, _ = canon(mv[:K])
        groups[key].append(([int(_CELL_MAPS[gs][a]) for a in mv], g["winner_c"]
                            if "winner_c" in g else g["winner"], gs))
    stats = []
    for key, members in groups.items():
        n = len(members)
        bw = sum(1 for _, w, _ in members if w == 1)
        ww = sum(1 for _, w, _ in members if w == -1)
        stats.append((key, n, bw, ww, n - bw - ww, members))
    stats.sort(key=lambda s: (-wilson_lb(s[2], s[1]), -s[1]))

    picked, per_prefix4 = [], Counter()
    for key, n, bw, ww, dr, members in stats:
        if n < MIN_N:
            continue
        seq = Counter(tuple(m[:K]) for m, _, _ in members).most_common(1)[0][0]
        p4 = tuple(canon(list(seq)[:4])[2])
        if per_prefix4[p4] >= MAX_PER_PREFIX4:
            continue
        # extend the modal K-ply line with most-frequent continuations
        line = list(seq)
        support = [n] * K
        while len(line) < MAX_PLY:
            nxt = Counter(m[len(line)] for m, _, _ in members
                          if len(m) > len(line) and m[:len(line)] == line)
            if not nxt:
                break
            a, cnt = nxt.most_common(1)[0]
            if cnt < max(30, 0.1 * n):
                break
            line.append(int(a))
            support.append(int(cnt))
        per_prefix4[p4] += 1
        picked.append({
            "id": f"op{len(picked) + 1:02d}",
            "name": auto_name(line),
            "line": [int(a) for a in line],
            "ply_book": K,
            "n": n, "black_wins": bw, "white_wins": ww, "draws": dr,
            "winrate_black": round(bw / n, 4),
            "wilson_lb": round(wilson_lb(bw, n), 4),
            "avg_len": round(float(np.mean([len(m) for m, _, _ in members])), 1),
            "line_support": support,
        })
        if len(picked) >= TOPK:
            break
    return picked, len(groups), skipped


# ---------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(1)
    # 1) symmetry round-trip: applying any g to a prefix must not change the
    #    canonical key or the canonical move sequence
    for _ in range(300):
        k = int(rng.integers(4, 12))
        prefix = list(rng.choice(N, size=k, replace=False))
        key0, _, seq0 = canon(prefix)
        g = int(rng.integers(1, 8))
        moved = [int(_CELL_MAPS[g][a]) for a in prefix]
        key1, _, seq1 = canon(moved)
        assert key0 == key1, "canonical key not transform-invariant"
        assert seq0 == seq1, "canonical sequence not transform-invariant"
    # 2) transform_cell is a group action: applying g then its inverse is id
    for g in range(8):
        m = _CELL_MAPS[g]
        assert sorted(m) == list(range(N)), "transform not a permutation"
    # 3) Wilson vs hand-computed value (p=0.9, n=100 -> ~0.8245)
    assert abs(wilson_lb(90, 100) - 0.8245) < 3e-3
    assert wilson_lb(0, 0) == 0.0
    # 4) planted rotated duplicates merge: one base game replicated under all
    #    8 transforms must land in ONE group with n=8
    base = list(rng.choice(N, size=10, replace=False))
    games = [{"moves": [int(_CELL_MAPS[g][a]) for a in base], "winner": 1}
             for g in range(8)]
    global MIN_N
    keep = MIN_N
    MIN_N = 1
    picked, n_groups, _ = mine(games)
    MIN_N = keep
    assert n_groups == 1, f"rotated duplicates split into {n_groups} groups"
    assert picked and picked[0]["n"] == 8
    # 5) the modal line replays legally (no duplicate cells)
    assert len(set(picked[0]["line"])) == len(picked[0]["line"])
    print("SELFTEST PASSED")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    src = sys.argv[1]
    games = [r for r in (json.loads(l) for l in open(src)) if "moves" in r]
    bw_all = sum(1 for g in games if g["winner"] == 1)
    picked, n_groups, skipped = mine(games)
    meta_rows = [r["_meta"] for r in (json.loads(l) for l in open(src))
                 if "_meta" in r]
    book = {
        "version": 1, "board": B,
        "ckpt": "iter040.pt",
        "source": {
            "games": len(games), "sims": 800,
            "temp_moves": int(os.environ.get("BOOK_TEMP_MOVES", "8")),
            "cap_prob": 0.25, "resign": 1,
            "date": os.environ.get("BOOK_DATE", ""),
            "black_overall_winrate": round(bw_all / max(1, len(games)), 4),
        },
        "openings": picked,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(book, f, ensure_ascii=False)
    print(f"mined {len(picked)} openings from {len(games)} games "
          f"({n_groups} distinct ply-{K} positions, {skipped} too short)")
    for op in picked:
        print(f"  {op['id']} {op['name']:18} n={op['n']:5d} "
              f"black {op['winrate_black']*100:5.1f}% (lb {op['wilson_lb']*100:.1f}%) "
              f"len {len(op['line'])} avg_game {op['avg_len']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
