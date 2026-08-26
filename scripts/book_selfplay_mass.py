"""Mass self-play generation for the opening book (read-only over iter040).

Modeled on calib_selfplay_point.py: import the trainer, load the checkpoint,
build the worker pool ONCE, then loop pool.run in chunks and append one JSONL
row per game -- X/PI/Z are discarded (we only need move lists and outcomes).
Output is incremental and crash-safe: each chunk also writes a "_meta" row
with the self-play stats, so a killed run keeps everything finished so far.

Row formats:
  {"moves": [flat r*15+c, ...], "winner": 1|-1|0, "len": n}
  {"_meta": {"chunk": i, "games": g, "seconds": s, ...sp_stats}}

Run in the node09 container (see the plan; GPUs 1-3, never GPU 5):
  AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 AZ_SIMS=800 AZ_TEMP_MOVES=8 \
  AZ_CAP_PROB=0.25 AZ_RESIGN=1 AZ_RESIGN_MIN=16 AZ_RESIGN_KEEP=0.05 \
  AZ_DEAD_DRAW=1 AZ_GPUS=1,1,1,1,2,2,2,2,3,3,3,3 \
  python scripts/book_selfplay_mass.py
Knobs: BOOK_CKPT, BOOK_GAMES (total), BOOK_CHUNK, BOOK_OUT, BOOK_SEED.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import train_rl_gomoku_alphazero as az  # noqa: E402

CKPT = os.environ.get("BOOK_CKPT", "results/gomoku_ckpt_p15/iter040.pt")
GAMES = int(os.environ.get("BOOK_GAMES", "50000"))
CHUNK = int(os.environ.get("BOOK_CHUNK", "1200"))
OUT = os.environ.get("BOOK_OUT", "results/book_games_p15.jsonl")
SEED = int(os.environ.get("BOOK_SEED", "20260825"))


def main():
    net = az.AZNet()
    net.load_state_dict(torch.load(CKPT, map_location="cpu"))
    pool = az.make_pool(az.GPUS, az.CHANNELS, az.BLOCKS)
    print(f"[book-gen] {az.BOARD}x{az.BOARD} {os.path.basename(CKPT)} | "
          f"{GAMES} games in chunks of {CHUNK} | {pool.n} workers on {az.GPUS} | "
          f"sims {az.N_SIMS} temp_moves {az.TEMP_MOVES} cap {az.CAP_PROB} "
          f"resign {az.RESIGN}(min {az.RESIGN_MIN})", flush=True)
    done = 0
    t00 = time.time()
    try:
        with open(OUT, "a") as f:
            chunk_i = 0
            while done < GAMES:
                g = min(CHUNK, GAMES - done)
                t0 = time.time()
                _, _, _, logs, winners, lengths, stats, _ = pool.run(
                    net, g, az.N_SIMS, SEED + 1000 * chunk_i, 0.0)
                dt = time.time() - t0
                for mv, w, ln in zip(logs, winners, lengths):
                    f.write(json.dumps({"moves": [int(a) for a in mv],
                                        "winner": int(w), "len": int(ln)}) + "\n")
                f.write(json.dumps({"_meta": {"chunk": chunk_i, "games": g,
                                              "seconds": round(dt, 1),
                                              **{k: int(v) for k, v in stats.items()}}}) + "\n")
                f.flush()
                done += g
                chunk_i += 1
                bw = sum(1 for w in winners if w == 1)
                eta = (GAMES - done) / max(done / (time.time() - t00), 1e-9)
                print(f"  chunk {chunk_i}: {g} games in {dt:.0f}s "
                      f"({g/dt:.2f} g/s) black {bw}/{g} "
                      f"rsn {stats.get('resigned', 0)} "
                      f"fr {stats.get('false_resigns', 0)}/{stats.get('noresign_games', 0)} "
                      f"| total {done}/{GAMES} eta {eta/60:.0f} min", flush=True)
    finally:
        pool.close()
    print(f"[book-gen] wrote {done} games to {OUT} "
          f"in {(time.time()-t00)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
