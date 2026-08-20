"""Export the trained Gomoku net as a Core ML model for native iOS/macOS play.

The browser build (export_gomoku_web.py + gomoku_play_template.html) reaches
~78 ms per node evaluation because its WebGL2 conv is a naive texture gather.
Core ML on the Neural Engine does the same forward in ~0.66 ms on an M2 Ultra
(measured), which is what makes 400-simulation MCTS viable on an iPhone/iPad.

Writes, under results/coreml_export/:
  GomokuAZ_b<N>.mlpackage  fp16 mlprogram, one per batch size in CML_BATCHES
  testvec.json             copied from the web export so the app can ship the
                           same 5 reference positions as an on-device self-test
  coreml_report.json       acceptance record: per-compute-unit latency, the
                           testvec deltas, and the compute-plan device
                           histogram (see "why the plan matters" below)

Three things this script checks, all of which have bitten a port before:

  1. testvec parity. The references in results/web_export/testvec.json are
     legal-masked softmax + value computed with fp16-rounded weights in fp32.
     Core ML computes in fp16 throughout, so it drifts a little more; the
     thresholds here are the same ones the play page uses in its boot
     self-test (argmax must match, policy maxD <= 5e-3, value D <= 2e-2).
  2. Compute-plan placement. The net uses GroupNorm, which -- unlike
     BatchNorm -- cannot be folded into the conv weights and lowers to a
     reduce_mean/sub/square/sqrt/real_div chain. If any of those ops fall off
     the ANE, Core ML partitions the graph and every partition boundary costs
     latency. This script fails unless every compute op prefers the ANE.
  3. Compute-unit choice. With MLComputeUnits.all the planner picked the GPU
     path here (2.67 ms vs 0.66 ms on ANE). The exported model is only fast if
     the app asks for .cpuAndNeuralEngine explicitly -- the report records
     both so a regression is visible.

Run on a Mac (needs coremltools, which is not a default dependency):
  uv sync --group coreml
  AZ_BOARD=15 AZ_CH=192 AZ_BLOCKS=12 \
    uv run python scripts/export_gomoku_coreml.py

Env: CAL_CKPT (results/gomoku_ckpt_p15/iter040.pt), CML_OUT
(results/coreml_export), CML_BATCHES (comma list, default "1"),
CML_TESTVEC (results/web_export/testvec.json), CML_BENCH_ITERS (200),
CML_SKIP_BENCH (unset), CML_ALLOW_NON_ANE (unset -> non-ANE ops are fatal).
"""
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import train_rl_gomoku_alphazero as az  # noqa: E402

try:
    import coremltools as ct
except ImportError:  # pragma: no cover - environment guard
    sys.exit("coremltools is missing. On a Mac: uv sync --group coreml")

CKPT = os.environ.get("CAL_CKPT", "results/gomoku_ckpt_p15/iter040.pt")
OUT = os.environ.get("CML_OUT", "results/coreml_export")
BATCHES = [int(b) for b in os.environ.get("CML_BATCHES", "1").split(",") if b]
TESTVEC = os.environ.get("CML_TESTVEC", "results/web_export/testvec.json")
BENCH_ITERS = int(os.environ.get("CML_BENCH_ITERS", "200"))
SKIP_BENCH = bool(os.environ.get("CML_SKIP_BENCH"))
ALLOW_NON_ANE = bool(os.environ.get("CML_ALLOW_NON_ANE"))

# same acceptance thresholds the play page applies in its boot self-test
TOL_POLICY = 5e-3
TOL_VALUE = 2e-2

UNITS = [("ANE", ct.ComputeUnit.CPU_AND_NE),
         ("GPU", ct.ComputeUnit.CPU_AND_GPU),
         ("ALL", ct.ComputeUnit.ALL),
         ("CPU", ct.ComputeUnit.CPU_ONLY)]


def load_net():
    net = az.AZNet()
    net.load_state_dict(torch.load(CKPT, map_location="cpu"))
    net.eval()
    return net


def convert(net, batch, path):
    """Trace at a fixed batch shape and convert to a fp16 mlprogram.

    Fixed rather than flexible shapes on purpose: ranged shapes push a Core ML
    graph off the ANE, and enumerated shapes still recompile per shape. One
    small file per batch size is cheaper than either.
    """
    ex = torch.randn(batch, 4, az.BOARD, az.BOARD)
    with torch.no_grad():
        ts = torch.jit.trace(net, ex)
    m = ct.convert(
        ts,
        inputs=[ct.TensorType(name="x", shape=ex.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="policy_logits", dtype=np.float16),
                 ct.TensorType(name="value", dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    # the Swift side reads geometry back from the model instead of hardcoding it
    m.short_description = ("AlphaZero Gomoku trunk+heads. policy_logits are RAW "
                           "logits: the caller must apply the legal mask before "
                           "softmax. value is tanh, from the mover's viewpoint.")
    m.user_defined_metadata.update({
        "board": str(az.BOARD), "n_in_row": str(az.N_IN_ROW),
        "channels": str(az.CHANNELS), "blocks": str(az.BLOCKS),
        "batch": str(batch), "ckpt": os.path.basename(CKPT),
        "c_puct": str(az.C_PUCT),
    })
    m.save(path)
    return m


def masked_softmax(logits, legal):
    z = np.where(legal, logits.astype(np.float64), -np.inf)
    z = z - z.max()
    p = np.exp(z) * legal
    return p / max(p.sum(), 1e-12)


def replay(moves):
    s = az.State()
    for a in moves:
        s.play(a)
    return s


def check_testvec(model, batch):
    """Replay the web export's reference positions through the Core ML model."""
    if not os.path.exists(TESTVEC):
        print(f"  ! {TESTVEC} missing -- skipping parity check")
        return [], True
    vecs = json.load(open(TESTVEC))["vectors"]
    rows, ok = [], True
    for v in vecs:
        s = replay(v["moves"])
        x = np.zeros((batch, 4, az.BOARD, az.BOARD), dtype=np.float16)
        x[0] = s.encode()
        out = model.predict({"x": x})
        logits = np.asarray(out["policy_logits"]).reshape(batch, -1)[0]
        value = float(np.asarray(out["value"]).reshape(-1)[0])
        p = masked_softmax(logits, s.legal_mask())
        ref_p = np.asarray(v["policy"], dtype=np.float64)
        dp = float(np.abs(p - ref_p).max())
        dv = abs(value - v["value"])
        am = int(p.argmax())
        good = am == v["argmax"] and dp <= TOL_POLICY and dv <= TOL_VALUE
        ok &= good
        rows.append({"name": v["name"], "argmax": am, "ref_argmax": v["argmax"],
                     "argmax_ok": am == v["argmax"],
                     "max_delta_policy": round(dp, 6),
                     "delta_value": round(dv, 6), "pass": good})
        print(f"  {v['name']:12} argmax {am:>3} (ref {v['argmax']:>3}) "
              f"policy maxD {dp:.2e}  value D {dv:.2e}  {'ok' if good else 'FAIL'}")
    return rows, ok


def compute_plan(path):
    """Which device does Core ML prefer for each op? -> (histogram, per-op).

    Weight consts report no usage and are counted separately: only compute ops
    matter for partitioning.
    """
    from collections import Counter, defaultdict
    try:
        from coremltools.models.compute_device import (
            MLCPUComputeDevice, MLGPUComputeDevice, MLNeuralEngineComputeDevice)
        from coremltools.models.compute_plan import MLComputePlan
    except ImportError:
        print("  ! this coremltools has no MLComputePlan -- skipping placement check")
        return None, None
    # both of these must pin CPU_AND_NE: with the default (ALL) the planner
    # reports every op on the GPU -- the same choice that costs 4x at runtime.
    # m has to stay alive, the compiled path is tied to its lifetime.
    m = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
    plan = MLComputePlan.load_from_path(m.get_compiled_model_path(),
                                        compute_units=ct.ComputeUnit.CPU_AND_NE)
    hist, per_op = Counter(), defaultdict(Counter)
    for op in plan.model_structure.program.functions["main"].block.operations:
        usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
        if usage is None:
            hist["const_or_unreported"] += 1
            continue
        d = usage.preferred_compute_device
        name = ("ANE" if isinstance(d, MLNeuralEngineComputeDevice) else
                "GPU" if isinstance(d, MLGPUComputeDevice) else
                "CPU" if isinstance(d, MLCPUComputeDevice) else type(d).__name__)
        hist[name] += 1
        per_op[op.operator_name][name] += 1
    return dict(hist), {k: dict(v) for k, v in per_op.items()}


def bench(path, batch):
    rows = []
    x = {"x": np.random.randn(batch, 4, az.BOARD, az.BOARD).astype(np.float16)}
    for name, cu in UNITS:
        t0 = time.perf_counter()
        m = ct.models.MLModel(path, compute_units=cu)
        load_s = time.perf_counter() - t0
        for _ in range(20):
            m.predict(x)
        n = max(20, BENCH_ITERS // max(1, batch))
        t0 = time.perf_counter()
        for _ in range(n):
            m.predict(x)
        ms = (time.perf_counter() - t0) / n * 1000
        rows.append({"unit": name, "load_s": round(load_s, 3),
                     "ms_per_infer": round(ms, 3),
                     "ms_per_position": round(ms / batch, 4)})
        print(f"  {name:<4} load {load_s:5.2f}s   {ms:8.3f} ms/infer   "
              f"{ms/batch:7.3f} ms/position")
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    net = load_net()
    n_par = sum(p.numel() for p in net.parameters())
    print(f"{az.BOARD}x{az.BOARD} {az.CHANNELS}ch/{az.BLOCKS}blk  "
          f"{n_par/1e6:.2f}M params  <- {CKPT}")

    report = {"ckpt": os.path.basename(CKPT), "board": az.BOARD,
              "channels": az.CHANNELS, "blocks": az.BLOCKS,
              "n_in_row": az.N_IN_ROW, "params": int(n_par),
              "coremltools": ct.__version__, "torch": torch.__version__,
              "tol_policy": TOL_POLICY, "tol_value": TOL_VALUE, "models": []}
    all_ok = True

    for batch in BATCHES:
        path = f"{OUT}/GomokuAZ_b{batch}.mlpackage"
        print(f"\n=== batch {batch} -> {path} ===")
        if os.path.isdir(path):
            shutil.rmtree(path)
        convert(net, batch, path)
        size = sum(os.path.getsize(os.path.join(d, f))
                   for d, _, fs in os.walk(path) for f in fs)
        print(f"  saved {size/1e6:.1f} MB")

        entry = {"batch": batch, "path": os.path.basename(path),
                 "bytes": size}

        # ANE is the deployment target, so parity is checked on that unit
        model = ct.models.MLModel(path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        print("  -- testvec parity (ANE) --")
        rows, ok = check_testvec(model, batch)
        entry["testvec"] = rows
        entry["testvec_pass"] = ok
        all_ok &= ok

        print("  -- compute plan --")
        hist, per_op = compute_plan(path)
        entry["compute_plan"] = hist
        entry["compute_plan_per_op"] = per_op
        if hist is not None:
            stray = {k: v for k, v in hist.items()
                     if k not in ("ANE", "const_or_unreported")}
            print(f"  {hist}")
            if stray:
                msg = f"  ! {sum(stray.values())} compute ops off the ANE: {stray}"
                print(msg)
                all_ok &= ALLOW_NON_ANE
            else:
                print(f"  all {hist.get('ANE', 0)} compute ops prefer the ANE")

        if not SKIP_BENCH:
            print("  -- latency --")
            entry["bench"] = bench(path, batch)
        report["models"].append(entry)

    if os.path.exists(TESTVEC):
        shutil.copyfile(TESTVEC, f"{OUT}/testvec.json")
    report["pass"] = all_ok
    with open(f"{OUT}/coreml_report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(f"\nwrote {OUT}  ({'PASS' if all_ok else 'FAIL'})")
    if not all_ok:
        sys.exit("acceptance checks failed -- see coreml_report.json")


if __name__ == "__main__":
    main()
