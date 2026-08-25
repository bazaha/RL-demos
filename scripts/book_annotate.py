"""Annotate book lines with the model's value (black POV) at every ply.

Runs locally in .venv-serve (MPS): replays each opening line, forwards
iter040 after each move, converts the mover-POV value to black POV, and
writes v_black[] (same length as line) into the book JSON in place.

  .venv-serve/bin/python scripts/book_annotate.py [results/book/gomoku_book.json]
"""
import json
import os
import sys

os.environ.setdefault("AZ_BOARD", "15")
os.environ.setdefault("AZ_CH", "192")
os.environ.setdefault("AZ_BLOCKS", "12")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_rl_gomoku_alphazero as az  # noqa: E402

CKPT = os.environ.get("BOOK_CKPT", "results/gomoku_ckpt_p15/iter040.pt")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/book/gomoku_book.json"
    book = json.load(open(path))
    assert book["board"] == az.BOARD

    if torch.backends.mps.is_available():
        dev = "mps"
    elif torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    net = az.AZNet().to(dev)
    net.load_state_dict(torch.load(CKPT, map_location=dev))
    net.eval()

    n_fwd = 0
    for op in book["openings"]:
        s = az.State()
        vs = []
        for a in op["line"]:
            s.play(a)
            with torch.no_grad():
                x = torch.from_numpy(s.encode()[None]).to(dev)
                _, v = net(x)
            v_mover = float(v.float().cpu()[0])
            # value is from the (next) mover's POV; convert to black POV
            vs.append(round(v_mover if s.to_play == 1 else -v_mover, 4))
            n_fwd += 1
        op["v_black"] = vs
    with open(path, "w") as f:
        json.dump(book, f, ensure_ascii=False)
    print(f"annotated {len(book['openings'])} openings "
          f"({n_fwd} forwards on {dev}); v_black[last] range "
          f"[{min(o['v_black'][-1] for o in book['openings']):+.3f}, "
          f"{max(o['v_black'][-1] for o in book['openings']):+.3f}]")


if __name__ == "__main__":
    main()
