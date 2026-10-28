#!/usr/bin/env python3
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for path in sorted(root.glob("L*/result.json")):
    data = json.loads(path.read_text())
    for row in data["samples"]:
        rows.append({
            "length": data["length"], "mode": data["mode"],
            "ratio": data["requested_ratio"], "repeat": row["repeat"],
            "requested_hit_tokens": row["requested_hit_tokens"],
            "actual_hit_tokens": row["actual_hit_tokens"],
            "ttft_ms": row["ttft_ms"], "verified": row["verified"]
        })
if not rows:
    raise SystemExit("No results to summarize")
with (root / "matrix.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(root / "matrix.csv")
