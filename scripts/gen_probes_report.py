"""Build the hard-probe visualization: results/gomoku_hard_probes.json ->
report/gomoku_probes.html (self-contained, no dependencies)."""
import datetime
import json
import os

SRC = os.environ.get("PROBES_SRC", "results/gomoku_hard_probes.json")
OUT = os.environ.get("PROBES_OUT", "report/gomoku_probes.html")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "probes_report_template.html")

with open(SRC) as f:
    M = json.load(f)

data = {
    "M": M,
    # run facts the JSON does not carry itself
    "intervention_iter": 25,   # AZ_TEMP_MOVES 20->10 resume point of the p15 run
    "date": datetime.date.fromtimestamp(os.path.getmtime(SRC)).isoformat(),
    "host": "node09.tx.bj.stonewise.cn",
    "ckpt_tag": "p15（15×15 / 192ch / 12blk）",
}

html = open(TEMPLATE).read()
assert "__DATA_JSON__" in html
html = html.replace("__DATA_JSON__", json.dumps(data, ensure_ascii=False))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT)//1024} KB)")
