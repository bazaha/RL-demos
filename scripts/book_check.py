"""Quick field checks over a book JSONL (used by M1 smoke and M2 final)."""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
games = [r for r in rows if "moves" in r]
meta = [r for r in rows if "_meta" in r]
bw = sum(1 for g in games if g["winner"] == 1)
ww = sum(1 for g in games if g["winner"] == -1)
lens = [g["len"] for g in games]
print(f"CHECK: {len(games)} games, {len(meta)} meta rows")
print(f"  black {bw} ({bw/len(games)*100:.1f}%)  white {ww}  "
      f"draw {len(games)-bw-ww}")
print(f"  len min/avg/max {min(lens)}/{sum(lens)/len(lens):.1f}/{max(lens)}")
fr = sum(m["_meta"].get("false_resigns", 0) for m in meta)
nr = sum(m["_meta"].get("noresign_games", 0) for m in meta)
rs = sum(m["_meta"].get("resigned", 0) for m in meta)
print(f"  resigned {rs}  false_resigns {fr}/{nr}")
assert all(len(g["moves"]) == g["len"] for g in games)
assert all(0 <= a < 225 for g in games for a in g["moves"])
assert all(len(set(g["moves"])) == len(g["moves"]) for g in games), "dup moves"
print("  field checks OK")
