"""Build the self-contained AlphaZero Gomoku report from results/gomoku_*.json.

Reads both training arms (pure self-play vs rule-guided cold start), their GPU
logs, the arm-vs-arm head-to-head eval if present, and inlines everything into
scripts/gomoku_report_template.html.
"""
import csv
import datetime
import json
import os
import sys

RESULTS = os.environ.get("GOMOKU_RESULTS", "results")
OUT = os.environ.get("GOMOKU_OUT", "report/gomoku.html")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gomoku_report_template.html")
# which results/gomoku_<tag>.json files make up the report, in display order;
# a single tag (e.g. GOMOKU_ARMS=p15) produces a single-arm report with the
# whole A/B comparison section removed
ARMS = [t for t in os.environ.get("GOMOKU_ARMS", "pure,rules").split(",") if t]


def read_gpu_log(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("timestamp") and r.get("util_pct")]
    if not rows:
        return []
    out = []
    for r in rows:
        try:
            out.append({"ts": int(r["timestamp"]), "util": int(r["util_pct"]),
                        "mem": int(r["mem_used_mib"]), "power": float(r["power_w"]),
                        "temp": int(r["temp_c"])})
        except (ValueError, KeyError):
            continue
    return out


def thin(seq, cap):
    """Keep at most cap evenly spaced samples, always including the last one."""
    if len(seq) <= cap:
        return seq
    step = len(seq) / cap
    idx = sorted({int(i * step) for i in range(cap)} | {len(seq) - 1})
    return [seq[i] for i in idx]


def bucket_gpu(rows, cap):
    """Average GPU samples into cap buckets.

    Plain decimation of a 1 Hz trace turns AlphaZero's self-play/train duty
    cycle into unreadable noise; bucket means keep the envelope legible while
    still showing the ~90 s cycle.
    """
    if len(rows) <= cap:
        return rows
    out = []
    for i in range(cap):
        lo = len(rows) * i // cap
        hi = max(lo + 1, len(rows) * (i + 1) // cap)
        chunk = rows[lo:hi]
        n = len(chunk)
        out.append({
            "ts": chunk[n // 2]["ts"],
            "util": round(sum(c["util"] for c in chunk) / n, 1),
            "mem": round(sum(c["mem"] for c in chunk) / n),
            "power": round(sum(c["power"] for c in chunk) / n, 1),
            "temp": round(sum(c["temp"] for c in chunk) / n),
        })
    return out


def read_cross_arm(path):
    """Head-to-head arm-vs-arm eval, or None if that pass was not run.

    Drops elo_pairs: the round-robin pair list is only there for auditing the
    Elo fit and would be the biggest single thing in the payload.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        c = json.load(f)
    c.pop("elo_pairs", None)
    return c


def main():
    arms = {}
    for tag in ARMS:
        p = f"{RESULTS}/gomoku_{tag}.json"
        if not os.path.exists(p):
            print(f"!! missing {p}, skipping arm {tag}", file=sys.stderr)
            continue
        with open(p) as f:
            m = json.load(f)
        gpu = read_gpu_log(f"{RESULTS}/gpu_log_gomoku_{tag}.csv")
        t0 = gpu[0]["ts"] if gpu else 0
        m["gpu"] = [{"t": g["ts"] - t0, "util": g["util"], "mem": g["mem"],
                     "power": g["power"], "temp": g["temp"]}
                    for g in bucket_gpu(gpu, 420)]
        m["gpu_t0"] = t0
        # normalise phase spans onto the same clock as the GPU trace
        for ph in m.get("phases", []):
            ph["a"] = round(ph["t0"] - t0, 1)
            ph["b"] = round(ph["t1"] - t0, 1)
        # keep only eval spans for chart shading; self-play/train alternate every
        # ~90 s and would just smear the whole plot
        m["eval_spans"] = [[ph["a"], ph["b"]] for ph in m.get("phases", [])
                           if ph["kind"] == "eval"]
        m.pop("phases", None)          # only eval_spans is used by the template
        # The round-robin pair list is bulky and not plotted, but the template
        # needs games_per_pair to caption the ladder honestly -- the games of a
        # pair start from the empty board with deterministic play, so only two
        # of them differ.
        ed = m.pop("elo_detail", None)
        if ed:
            m["elo_detail"] = {k: ed[k]
                               for k in ("sims", "games_per_pair", "temp",
                                         "seconds")
                               if k in ed}
        # replay games: keep a spread across training, not every one.
        # One game per kept iteration -- the trainer stores two per iteration, so
        # slicing games rather than iterations would silently keep only the first
        # four iterations and the tab strip would stop at iter 19 of 50.
        sg = m.get("sample_games", [])
        sp = [g for g in sg if g.get("kind") == "selfplay"]
        vs = [g for g in sg if g.get("kind", "").startswith("vs-")]
        by_iter = {}
        for g in sp:
            by_iter.setdefault(g["iter"], g)
        keep_iters = thin(sorted(by_iter), 8)
        # eval games: a spread of the repeated opponent plus every distinct one
        # (the final harder baseline only appears once, and it is the headline)
        rep = [g for g in vs if g["kind"] == (vs[0]["kind"] if vs else "")]
        rest = [g for g in vs if g["kind"] != (vs[0]["kind"] if vs else "")]
        m["sample_games"] = ([by_iter[i] for i in keep_iters]
                             + thin(rep, 3) + rest[-2:])
        arms[tag] = m

    if not arms:
        sys.exit("no arm results found; nothing to report")

    ref = arms.get("pure") or next(iter(arms.values()))
    cross = read_cross_arm(f"{RESULTS}/gomoku_cross_arm.json")
    if cross and not all(a in arms for a in cross.get("arms", [])):
        print(f"!! cross-arm data references {cross.get('arms')} but only "
              f"{list(arms)} loaded; dropping it", file=sys.stderr)
        cross = None
    # stamp the report with the run's own date, not the date this script was
    # written -- results/ accumulates runs and a wrong date conflates them
    run_date = (datetime.date.fromtimestamp(ref["phase_start"]).isoformat()
                if ref.get("phase_start") else "")
    data = {
        "arms": arms,
        "arm_order": [t for t in ARMS if t in arms],
        "cross_arm": cross,
        "node": {
            "host": "node09.tx.bj.stonewise.cn",
            "gpus": "8 × NVIDIA H20 (96 GB)",
            "driver": "535.161.07",
            "image": "harbor.stonewise.cn/base/nvidia/pytorch:25.11-py3-cuda13.0-torch2.10",
            "torch": ref.get("torch", ""), "cuda": ref.get("cuda", ""),
            "date": "2026-07-25",
        },
    }
    with open(TEMPLATE) as f:
        html = f.read()
    html = html.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    out_dir = os.path.dirname(OUT)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print("wrote", OUT, f"({os.path.getsize(OUT)//1024} KB)")
    for tag, m in arms.items():
        its = m.get("iterations", [])
        print(f"  arm {tag}: {len(its)} iters, {len(m.get('evals', []))} evals, "
              f"{len(m.get('sample_games', []))} replay games, "
              f"{len(m.get('gpu', []))} gpu samples")
    if cross:
        print(f"  cross-arm: {len(cross.get('matched', []))} matched pairs, "
              f"{len(cross.get('elo', []))} elo players")
    else:
        print("  cross-arm: none (head-to-head section will be omitted)")


if __name__ == "__main__":
    main()
