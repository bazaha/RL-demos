"""Local inference service for the Gomoku play page.

Reuses the trainer's AZNet/State/Tree/run_sims verbatim (import, no
reimplementation), so server play is bit-identical in semantics to training
and evaluation. The play page probes http://127.0.0.1:8787/health at boot and
routes AI moves here when available, falling back to its embedded WebGL2
engine otherwise.

Endpoints (all JSON):
  GET  /health            -> {ok, device, board, channels, blocks, ckpt, torch}
  POST /move    {moves:[int], sims:int}
                          -> {move, value, visits:{a:n}, elapsed_ms}
                             value is from the mover's (AI's) point of view;
                             sims=0 returns the raw-policy argmax
  POST /forward {moves:[int]}
                          -> {policy:[225], value}   (legal-masked softmax)

Run on a MacBook (Apple Silicon -> MPS):
  .venv-serve/bin/python scripts/serve_gomoku.py
Run in the node09 container (CUDA), then reach it via an SSH tunnel:
  see scripts/run_gomoku_serve.sh

Config via env: AZ_BOARD/AZ_CH/AZ_BLOCKS (must match the checkpoint),
SERVE_CKPT (default results/gomoku_ckpt_p15/iter040.pt),
SERVE_HOST (default 127.0.0.1), SERVE_PORT (default 8787).
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# the trainer reads its architecture from AZ_* env at import time -- pin the
# 15x15 shape BEFORE importing it, unless the caller already set them
os.environ.setdefault("AZ_BOARD", "15")
os.environ.setdefault("AZ_CH", "192")
os.environ.setdefault("AZ_BLOCKS", "12")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_rl_gomoku_alphazero as az  # noqa: E402

CKPT = os.environ.get("SERVE_CKPT", "results/gomoku_ckpt_p15/iter040.pt")
HOST = os.environ.get("SERVE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SERVE_PORT", "8787"))
MAX_SIMS = int(os.environ.get("SERVE_MAX_SIMS", "6400"))

if torch.cuda.is_available():
    DEVICE = "cuda:0"
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

NET = az.AZNet().to(DEVICE)
NET.load_state_dict(torch.load(CKPT, map_location=DEVICE))
NET.eval()
# one inference at a time: the net and (on MPS) the backend are not meant for
# concurrent forwards, and games are turn-based anyway
LOCK = threading.Lock()


def build_state(moves):
    s = az.State()
    for a in moves:
        a = int(a)
        if not (0 <= a < az.N_ACT) or s.board.reshape(-1)[a] != 0 or s.done:
            raise ValueError(f"illegal move in history: {a}")
        s.play(a)
    return s


def forward_one(state):
    with torch.no_grad():
        x = torch.from_numpy(state.encode()[None]).to(DEVICE)
        logits, v = NET(x)
        legal = state.legal_mask()
        p = torch.softmax(logits[0], dim=0).float().cpu().numpy() * legal
        p = p / max(p.sum(), 1e-12)
    return p, float(v[0])


def do_move(moves, sims):
    state = build_state(moves)
    if state.done:
        raise ValueError("game is already over")
    t0 = time.time()
    if sims <= 0:
        p, v = forward_one(state)
        a = int(p.argmax())
        visits = None
    else:
        tree = az.Tree(state)
        az.run_sims(NET, [tree], min(sims, MAX_SIMS), DEVICE)
        n = tree.root.N
        a = int(n.argmax())
        # backup flips sign per level: root W/N is already the mover's POV
        v = float(tree.root.W[a] / max(n[a], 1.0))
        visits = {int(i): int(n[i]) for i in np.flatnonzero(n > 0)}
    return {"move": a, "value": round(v, 4), "visits": visits,
            "elapsed_ms": round((time.time() - t0) * 1000)}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # the page is opened from file:// (origin "null"): allow it, plus the
        # Private Network Access preflight header newer Chrome requires
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self._send(204, {})

    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        self._send(200, {"ok": True, "device": DEVICE, "board": az.BOARD,
                         "channels": az.CHANNELS, "blocks": az.BLOCKS,
                         "ckpt": os.path.basename(CKPT),
                         "max_sims": MAX_SIMS,
                         "torch": torch.__version__})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            moves = req.get("moves", [])
            if self.path == "/move":
                with LOCK:
                    out = do_move(moves, int(req.get("sims", 400)))
                return self._send(200, out)
            if self.path == "/forward":
                with LOCK:
                    p, v = forward_one(build_state(moves))
                return self._send(200, {"policy": [round(float(x), 6) for x in p],
                                        "value": round(v, 4)})
            return self._send(404, {"error": "not found"})
        except ValueError as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:  # keep the server alive on bad input
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} "
              f"{fmt % args}", flush=True)


def main():
    # warm-up: the first MPS/CUDA forward compiles kernels and would otherwise
    # be billed to the first user request
    t0 = time.time()
    forward_one(az.State())
    print(f"[serve] {az.BOARD}x{az.BOARD} {az.CHANNELS}ch/{az.BLOCKS}blk "
          f"| {os.path.basename(CKPT)} on {DEVICE} (torch {torch.__version__}) "
          f"| warm-up {time.time() - t0:.2f}s", flush=True)
    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[serve] listening on http://{HOST}:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
