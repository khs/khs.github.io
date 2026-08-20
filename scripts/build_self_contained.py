#!/usr/bin/env python3
"""Generate self-contained/*.html from the embed-*.html sources.

The self-contained widgets are the embed widgets with the data fetched at
runtime replaced by data baked in at build time, so they work with no network
access beyond the Chart.js CDN.

Deriving them from the embed sources (rather than patching the previous build
in place) keeps the two in sync and avoids re-editing generated JavaScript,
which is how a backslash escape was once mangled into a syntax error that
silently killed the whole chart script.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Only the fields the widgets actually read, to keep the payload small.
LAG_FIELDS = [
    "alpha", "betas", "r2", "seasonal_premium",
    "total_passthrough", "band_passthrough",
]

WIDGETS = [
    # source,             output,                      init on success
    ("embed-chart.html", "self-contained/chart.html", "init();"),
    ("embed-cost.html",  "self-contained/cost.html",
     "if (oilData.data_available) renderCostCalc();\n"
     "else document.getElementById('cost-total-number').textContent = 'No data';"),
]


def build(src_name, out_name, init_stmt, oil_json, lag_json):
    src = (ROOT / src_name).read_text()

    # Drop the fetch base; nothing is fetched in the self-contained build.
    out = re.sub(r"^const DATA_BASE = [^\n]*\n", "", src, count=1, flags=re.M)
    if out == src:
        raise SystemExit(f"{src_name}: DATA_BASE declaration not found")

    boot_re = re.compile(r"^Promise\.all\(\[.*?^\}\);\s*$", re.M | re.S)
    if not boot_re.search(out):
        raise SystemExit(f"{src_name}: fetch bootstrap not found")

    # Chart.js may not have finished initialising when this inline script runs,
    # so defer anything that touches it to the next frame.
    replacement = (
        "requestAnimationFrame(function () {\n"
        f"oilData = {oil_json};\n"
        f"lagModel = {lag_json};\n"
        f"{init_stmt}\n"
        "});"
    )
    # A function replacement keeps re from interpreting backslashes in the JSON.
    out = boot_re.sub(lambda _m: replacement, out, count=1)

    (ROOT / out_name).write_text(out)
    return out_name, len(out)


def main():
    oil = json.loads((ROOT / "data/oil-futures.json").read_text())
    lag_full = json.loads((ROOT / "data/lag-model.json").read_text())
    lag = {k: lag_full[k] for k in LAG_FIELDS if k in lag_full}

    oil_json = json.dumps(oil, separators=(",", ":"))
    lag_json = json.dumps(lag, separators=(",", ":"))

    for src, out, init in WIDGETS:
        name, size = build(src, out, init, oil_json, lag_json)
        print(f"built {name} ({size:,} bytes)")


if __name__ == "__main__":
    sys.exit(main())
