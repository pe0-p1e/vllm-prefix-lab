#!/usr/bin/env python3
"""Summarize overlapping CPU-host KV reload and background Decode trace."""
import json
from pathlib import Path
import sys


def fmt(x, precision=3):
    return "N/A" if x is None else f"{x:.{precision}f}"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/summarize_overlap.py results/tiered-...")
    root = Path(sys.argv[1])
    data = json.loads((root / "cpu" / "overlap.json").read_text())
    host = data["host_cached_tokens"]
    verified = data["cpu_hit_verified"] and data["actual_overlap_verified"]
    intervals_ok = data["during"]["n"] >= 3
    lines = [
        "# Active Decode during CPU KV Restore",
        "",
        "A foreground request restores 8K-prefix host KV while a separate",
        "background request is streaming a long output.",
        "",
        f"- Verified host hit: {data['cpu_hit_verified']} ({host:.0f} cached tokens)",
        f"- Verified temporal overlap: {data['actual_overlap_verified']}",
        f"- Overlapping token intervals >= 3: {intervals_ok}",
        f"- Valid functional overlap: {verified and intervals_ok}",
        f"- Foreground restore TTFT: {data['revisit']['ttft_ms']:.2f} ms",
        f"- Foreground wall-time window: {data['load_window_ms']:.2f} ms",
        f"- Offload CPU-to-GPU bytes (delta): {data['cuda_load_bytes']:.0f} bytes",
        f"- Offload CUDA event time sum: {fmt(data['cuda_load_time_sum_seconds'], 6)} seconds",
        "",
        "| Background decode phase | Inter-chunk count | Median ms | P95 ms | Max ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for label,key in [
        ("No concurrent request (baseline)", "baseline_all"),
        ("Before restore request", "before"),
        ("During restore request", "during"),
        ("After restore request", "after"),
        ("Whole treatment stream", "treatment_all"),
    ]:
        d = data[key]
        lines.append(
            f"| {label} | {d['n']} | {fmt(d['median_ms'])} | "
            f"{fmt(d['p95_ms'])} | {fmt(d['max_ms'])} |"
        )
    lines.extend([
        "",
        "These are single-run, client-side streamed-chunk timings.",
        "The 'during' window is foreground request submit through completion,",
        "NOT the precise CUDA DMA interval. Delays may include scheduling,",
        "KV transfer, suffix Prefill, and competing Decode. To isolate DMA,",
        "compare matched control workloads and use CUDA/Prometheus metrics.",
        "Do not interpret the sum of GPU event transfer durations as",
        "request wall-clock latency or a critical-path breakdown.",
        "",
    ])
    (root / "OVERLAP_REPORT.md").write_text("\n".join(lines))
    print("Saved:", root / "OVERLAP_REPORT.md")


if __name__ == "__main__":
    main()
