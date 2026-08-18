"""Export the trained Gomoku net for in-browser play.

Writes, under results/web_export/:
  weights_fp16.bin   every state_dict tensor, flattened, little-endian float16,
                     concatenated in state_dict order
  manifest.json      architecture constants + per-tensor {name, shape, offset}
  testvec.json       reference positions (as move lists) with the policy /
                     value the JS engine must reproduce. References are
                     computed with fp16-rounded weights in fp32 arithmetic --
                     exactly the browser's numeric model -- plus the original
                     fp32 outputs for drift bookkeeping.

Run inside the container: AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 \
  CAL_CKPT=results/gomoku_ckpt_p15/iter040.pt python scripts/export_gomoku_web.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_rl_gomoku_alphazero as az  # noqa: E402

CKPT = os.environ.get("CAL_CKPT", "results/gomoku_ckpt_p15/iter040.pt")
OUT = os.environ.get("WEB_OUT", "results/web_export")


def forward_ref(net, states):
    """Legal-masked softmax policy + value for a list of States."""
    xs = np.stack([s.encode() for s in states])
    with torch.no_grad():
        logits, v = net(torch.from_numpy(xs))
        out = []
        for i, s in enumerate(states):
            legal = s.legal_mask()
            p = torch.softmax(logits[i], dim=0).numpy() * legal
            p = p / max(p.sum(), 1e-12)
            out.append((p, float(v[i])))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    net = az.AZNet()
    net.load_state_dict(torch.load(CKPT, map_location="cpu"))
    net.eval()

    sd = net.state_dict()
    manifest = {"board": az.BOARD, "channels": az.CHANNELS, "blocks": az.BLOCKS,
                "n_in_row": az.N_IN_ROW, "gn_groups": 8, "gn_eps": 1e-5,
                "ckpt": os.path.basename(CKPT), "params": []}
    blobs = []
    off = 0
    for name, t in sd.items():
        a = t.numpy().astype(np.float16)
        manifest["params"].append({"name": name, "shape": list(t.shape),
                                   "offset": off, "numel": int(a.size)})
        blobs.append(a.tobytes())
        off += a.size * 2
    with open(f"{OUT}/weights_fp16.bin", "wb") as f:
        f.write(b"".join(blobs))
    manifest["bytes"] = off
    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump(manifest, f)
    print(f"weights: {off/1e6:.1f} MB fp16, {len(sd)} tensors")

    # test positions as move lists (JS replays them through its own State)
    B = az.BOARD
    mid = B // 2

    def mv(r, c):
        return r * B + c

    seqs = {
        "empty": [],
        "one_stone": [mv(mid, mid)],
        "open_three": [mv(7, 6), mv(6, 6), mv(7, 7), mv(6, 7), mv(7, 8),
                       mv(0, 0)],                       # black open three, black to move
        "must_block": [mv(7, 5), mv(0, 1), mv(7, 6), mv(0, 3), mv(7, 7),
                       mv(0, 5), mv(7, 8)],             # black four -> white must block
        "midgame": [mv(7, 7), mv(7, 8), mv(8, 8), mv(6, 6), mv(8, 6), mv(8, 7),
                    mv(6, 8), mv(9, 7), mv(5, 9), mv(9, 5), mv(9, 9), mv(10, 4)],
    }
    states = {}
    for k, moves in seqs.items():
        s = az.State()
        for a in moves:
            s.play(a)
        assert not s.done, k
        states[k] = s

    ref32 = forward_ref(net, list(states.values()))
    # the browser computes in fp32 on fp16-rounded weights: build that net
    net16 = az.AZNet()
    net16.load_state_dict({k: v.half().float() for k, v in sd.items()})
    net16.eval()
    ref16 = forward_ref(net16, list(states.values()))

    vecs = []
    for (k, moves), (p32, v32), (p16, v16) in zip(seqs.items(), ref32, ref16):
        drift = float(np.abs(p32 - p16).max())
        vecs.append({"name": k, "moves": moves,
                     "policy": [round(float(x), 7) for x in p16],
                     "value": round(v16, 6),
                     "argmax": int(p16.argmax()),
                     "fp16_policy_drift_vs_fp32": round(drift, 7),
                     "value32": round(v32, 6)})
        print(f"  {k:12} argmax={divmod(int(p16.argmax()), B)} "
              f"v={v16:+.4f} (fp32 {v32:+.4f})  fp16 drift={drift:.2e}")
        assert int(p16.argmax()) == int(p32.argmax()), \
            f"{k}: fp16 rounding flips the argmax -- quantization too lossy"
    with open(f"{OUT}/testvec.json", "w") as f:
        json.dump({"board": B, "vectors": vecs}, f)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
