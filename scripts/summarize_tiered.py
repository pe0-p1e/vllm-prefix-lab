#!/usr/bin/env python3
"""Summarize controlled native vLLM long-prefix GPU/CPU cache probes."""
import csv
import json
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 scripts/summarize_tiered.py results/tiered-...")
    root = Path(sys.argv[1])
    records = {}
    rows = []
    for mode in ("gpu", "cpu"):
        path = root / mode / "probe.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        records[mode] = data
        for event in data["events"]:
            rows.append({
                "mode": mode,
                "phase": event["stage"],
                "prefix_id": event["prefix_id"],
                "prefix_len": data["prefix_len"],
                "gpu_cache_tokens": data["gpu_kv_capacity_tokens"],
                "ttft_ms": event["ttft_ms"],
                "inter_chunk_ms": event["inter_chunk_ms"],
                "e2e_ms": event["e2e_ms"],
                "stream_chunks": event["stream_chunks"],
            })
    if not rows:
        raise SystemExit("No probe.json found; experiments did not finish.")
    with (root / "requests.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Native vLLM GPU/CPU KV prefix-reuse probe",
        "",
        "Requests: cold A, immediate A reuse, enough unique B prefixes to exceed",
        "the measured GPU KV token capacity, then replay A.",
        "",
        "CPU reuse is only established by positive source=host cached-token",
        "deltas for the replay request, not by TTFT alone.",
        "",
        "| Mode | Cold A TTFT | Hot A TTFT | Revisit A TTFT | Revisit ITL* | Host cached tokens (revisit) | GPU cached tokens (revisit) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, data in records.items():
        events = data["events"]
        cold, hot, replay = events[0], events[1], events[-1]

        def fmt(value):
            return "N/A" if value is None else f"{value:.2f}"

        md.append(
            f"| {mode} | {fmt(cold['ttft_ms'])} ms | "
            f"{fmt(hot['ttft_ms'])} ms | {fmt(replay['ttft_ms'])} ms | "
            f"{fmt(replay['inter_chunk_ms'])} ms | "
            f"{data['revisit_host_cached_tokens']:.0f} | "
            f"{data['revisit_gpu_cached_tokens']:.0f} |"
        )

    valid = records.get("cpu", {}).get("cpu_hit_verified", False)
    md += [
        "",
        f"**Verified CPU-replay host KV hit: {'YES' if valid else 'NO / NOT CHECKED'}**",
        "",
        "* ITL is client-side streamed-chunk inter-arrival latency, an",
        "approximation to TPOT, not an instrumented GPU kernel time.",
        "Single-replay measurements are exploratory, not a statistically stable",
        "performance comparison. Check raw probe.json and server logs.",
        "",
        "GPU-only replay is a recompute baseline only if its device-cache token",
        "counter remains near zero. Later repeated runs should randomize",
        "experiment order and collect multiple validated host hits.",
    ]
    (root / "REPORT.md").write_text("\n".join(md) + "\n")
    print("Saved", root / "REPORT.md", "and", root / "requests.csv")
    if "cpu" in records and not valid:
        print("WARNING: no CPU cache hit. Do not present CPU replay as offload speedup.")


if __name__ == "__main__":
    main()
