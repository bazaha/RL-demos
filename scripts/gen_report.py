"""Build the self-contained HTML validation report from results/*.

Reads dl_metrics.json, rl_metrics.json, gpu_log_dl.csv, gpu_log_rl.csv and
writes report/index.html with all data inlined (no external deps).
"""
import csv
import json
import os

RESULTS = "results"
OUT = "report/index.html"


def read_gpu_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    t0 = int(rows[0]["timestamp"])
    return [{"t": int(r["timestamp"]) - t0, "util": int(r["util_pct"]),
             "mem": int(r["mem_used_mib"]), "power": float(r["power_w"]),
             "temp": int(r["temp_c"])} for r in rows]


def main():
    with open(f"{RESULTS}/dl_metrics.json") as f:
        dl = json.load(f)
    with open(f"{RESULTS}/rl_metrics.json") as f:
        rl = json.load(f)
    data = {
        "dl": dl,
        "rl": rl,
        "gpu_dl": read_gpu_log(f"{RESULTS}/gpu_log_dl.csv"),
        "gpu_rl": read_gpu_log(f"{RESULTS}/gpu_log_rl.csv"),
        "node": {
            "host": "node09.tx.bj.stonewise.cn",
            "gpus": "8 × NVIDIA H20 (96 GB)",
            "driver": "535.161.07",
            "image": "node09-h20-validation:20260717 (NGC PyTorch 24.08)",
            "date": "2026-07-24",
        },
    }
    with open("scripts/report_template.html") as f:
        html = f.read()
    html = html.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(html)
    print("wrote", OUT, f"({os.path.getsize(OUT)//1024} KB)")


if __name__ == "__main__":
    main()
