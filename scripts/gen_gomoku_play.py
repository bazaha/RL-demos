"""Assemble the playable Gomoku page: inline fp16 weights + manifest + test
vectors into scripts/gomoku_play_template.html -> report/gomoku_play.html.

Inputs (from scripts/export_gomoku_web.py, run in the container):
  results/web_export/weights_fp16.bin
  results/web_export/manifest.json
  results/web_export/testvec.json
"""
import base64
import json
import os

SRC = os.environ.get("PLAY_SRC", "results/web_export")
OUT = os.environ.get("PLAY_OUT", "report/gomoku_play.html")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "gomoku_play_template.html")

with open(f"{SRC}/weights_fp16.bin", "rb") as f:
    raw = f.read()
b64 = base64.b64encode(raw).decode("ascii")
manifest = open(f"{SRC}/manifest.json").read()
testvec = open(f"{SRC}/testvec.json").read()

html = open(TEMPLATE).read()
for k, v in (("__MANIFEST__", manifest), ("__TESTVEC__", testvec)):
    assert k in html, k
    html = html.replace(k, v)
assert "__WEIGHTS_B64__" in html
html = html.replace("__WEIGHTS_B64__", b64)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(html)
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB, "
      f"weights {len(raw)/1e6:.1f} MB fp16)")
