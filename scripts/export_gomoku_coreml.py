"""Convert the trained Gomoku net to CoreML for the iOS app.

Produces ios/Gomoku15/Models/Gomoku15.mlpackage (fp16 mlprogram, batch 1,
outputs raw policy logits [1,225] + value [1]) and validates it ON THIS MAC
against the same reference vectors the web engine was validated with: the
Mac and the target devices run the same CoreML stack, so an argmax/tolerance
pass here locks the numerics before Xcode is even opened.

Run:  .venv-serve/bin/python scripts/export_gomoku_coreml.py
"""
import json
import os
import sys

os.environ.setdefault("AZ_BOARD", "15")
os.environ.setdefault("AZ_CH", "192")
os.environ.setdefault("AZ_BLOCKS", "12")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coremltools as ct  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_rl_gomoku_alphazero as az  # noqa: E402

CKPT = os.environ.get("CKPT", "results/gomoku_ckpt_p15/iter040.pt")
OUT = os.environ.get("OUT", "ios/Gomoku15/Models/Gomoku15.mlpackage")
TESTVEC = "results/web_export/testvec.json"


class Wrapped(torch.nn.Module):
    """(1,4,15,15) -> policy logits (1,225), value (1,)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        logits, v = self.net(x)
        return logits, v


def main():
    net = az.AZNet()
    net.load_state_dict(torch.load(CKPT, map_location="cpu"))
    net.eval()
    wrapped = Wrapped(net).eval()

    ex = torch.zeros(1, 4, az.BOARD, az.BOARD)
    traced = torch.jit.trace(wrapped, ex)
    ml = ct.convert(
        traced,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,
        inputs=[ct.TensorType(name="board", shape=ex.shape, dtype=np.float32)],
        # fp32 output casts: internals stay fp16 (ANE-eligible); fp16 OUTPUT
        # tensors came back all-zero from the iOS-simulator CPU path
        outputs=[ct.TensorType(name="policy_logits", dtype=np.float32),
                 ct.TensorType(name="value", dtype=np.float32)],
    )
    ml.author = "rl-demos"
    ml.short_description = ("AlphaZero Gomoku 15x15 policy/value net "
                            f"({os.path.basename(CKPT)}, fp16)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    ml.save(OUT)
    print(f"saved {OUT}")

    # ---- validation against the reference vectors (CoreML on this Mac) ----
    tv = json.load(open(TESTVEC))
    worst_p, worst_v = 0.0, 0.0
    for vec in tv["vectors"]:
        s = az.State()
        for a in vec["moves"]:
            s.play(a)
        x = s.encode()[None].astype(np.float32)
        out = ml.predict({"board": x})
        logits = np.asarray(out["policy_logits"]).reshape(-1)
        v = float(np.asarray(out["value"]).reshape(-1)[0])
        legal = s.legal_mask()
        e = np.exp(logits - logits[legal].max())
        e[~legal] = 0
        p = e / e.sum()
        am = int(p.argmax())
        dp = float(np.abs(p - np.array(vec["policy"])).max())
        dv = abs(v - vec["value"])
        worst_p, worst_v = max(worst_p, dp), max(worst_v, dv)
        ok = am == vec["argmax"]
        print(f"  {vec['name']:12} argmax {'OK' if ok else 'FAIL'}  "
              f"dPolicy={dp:.2e}  dValue={dv:.2e}")
        assert ok, vec["name"]
    assert worst_p < 5e-3 and worst_v < 2e-2, (worst_p, worst_v)
    print(f"CoreML validation PASSED (worst dPolicy {worst_p:.2e}, "
          f"dValue {worst_v:.2e})")


if __name__ == "__main__":
    main()
