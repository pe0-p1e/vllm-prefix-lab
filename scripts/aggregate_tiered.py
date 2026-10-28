#!/usr/bin/env python3
"""Summarize paired long-prefix CPU/GPU KV-recovery runs without fabricating data."""
import csv
import json
import pathlib
import statistics
import sys


def fmt(values, digits=2):
    if not values:
        return "N/A"
    return f"{statistics.median(values):.{digits}f} (range {min(values):.{digits}f}–{max(values):.{digits}f})"


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python3 scripts/aggregate_tiered.py results/tiered-repeat-...")
    root = pathlib.Path(sys.argv[1])
    rows = []
    with (root / "manifest.csv").open() as file:
        for trial in csv.DictReader(file):
            run = pathlib.Path(trial["run_root"])
            gpu = json.loads((run / "gpu" / "probe.json").read_text())
            cpu = json.loads((run / "cpu" / "probe.json").read_text())
            expected = int(cpu["prefix_len"])
            if gpu["prefix_len"] != expected:
                raise ValueError(f"Prefix lengths differ in trial {trial['repetition']}")
            gpu_revisit = gpu["events"][-1]
            cpu_revisit = cpu["events"][-1]
            gpu_hit = float(gpu["revisit_gpu_cached_tokens"])
            host_hit = float(cpu["revisit_host_cached_tokens"])
            # Mark partial/no hits separately. A full prefix hit should be
            # at least 80% of its token count, after block alignment.
            valid = gpu_hit <= 0.1 * expected and host_hit >= 0.8 * expected
            gpu_ttft = gpu_revisit["ttft_ms"]
            cpu_ttft = cpu_revisit["ttft_ms"]
            rows.append({
                "repetition": int(trial["repetition"]),
                "order": trial["order"],
                "prefix_len": expected,
                "gpu_recompute_ttft_ms": gpu_ttft,
                "cpu_restore_ttft_ms": cpu_ttft,
                "reduction_percent": (100 * (gpu_ttft - cpu_ttft) / gpu_ttft)
                if valid and gpu_ttft > 0 else None,
                "gpu_revisit_inter_chunk_ms": gpu_revisit["inter_chunk_ms"],
                "cpu_revisit_inter_chunk_ms": cpu_revisit["inter_chunk_ms"],
                "gpu_device_hits": gpu_hit,
                "cpu_host_hits": host_hit,
                "validated": valid,
            })
    if not rows:
        raise SystemExit("No completed paired runs in manifest")
    with (root / "pairs.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    valid_rows = [r for r in rows if r["validated"]]
    reductions = [r["reduction_percent"] for r in valid_rows]
    gpu_ttft = [r["gpu_recompute_ttft_ms"] for r in valid_rows]
    cpu_ttft = [r["cpu_restore_ttft_ms"] for r in valid_rows]
    gpu_itl = [r["gpu_revisit_inter_chunk_ms"] for r in valid_rows
               if r["gpu_revisit_inter_chunk_ms"] is not None]
    cpu_itl = [r["cpu_revisit_inter_chunk_ms"] for r in valid_rows
               if r["cpu_revisit_inter_chunk_ms"] is not None]
    lines = [
        "# Paired CPU KV recovery versus GPU recompute",
        "",
        f"Complete pairs: {len(rows)}; validated pairs: {len(valid_rows)}.",
        "Validation requires GPU-only replay device hits <= 10% of prefix tokens",
        "and CPU replay host hits >= 80% of prefix tokens.",
        "",
        "| Trial | Order | GPU recompute TTFT | CPU restore TTFT | Reduction | Host KV hit | Valid |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        reduction = "N/A" if r["reduction_percent"] is None else f"{r['reduction_percent']:.1f}%"
        lines.append(
            f"| {r['repetition']} | {r['order']} | "
            f"{r['gpu_recompute_ttft_ms']:.2f} ms | "
            f"{r['cpu_restore_ttft_ms']:.2f} ms | {reduction} | "
            f"{r['cpu_host_hits']:.0f} | {'YES' if r['validated'] else 'NO'} |"
        )
    lines.extend([
        "",
        "## Summary of validated pairs (median; min–max)",
        "",
        f"- GPU recompute TTFT: {fmt(gpu_ttft)} ms",
        f"- CPU restore TTFT: {fmt(cpu_ttft)} ms",
        f"- Paired TTFT reduction: {fmt(reductions, 1)}%",
        f"- GPU replay inter-chunk interval: {fmt(gpu_itl, 3)} ms",
        f"- CPU replay inter-chunk interval: {fmt(cpu_itl, 3)} ms",
        "",
        "These are sequential requests without competing decode work.",
        "Inter-chunk timing is a client-side TPOT proxy. Min–max on a few",
        "repetitions is not a confidence interval. Unrelated GPU workloads",
        "can invalidate latency comparisons even when cache hits are verified.",
        "The next experiment should explicitly overlap host KV restoration",
        "with a long-running decode and measure the foreground and background",
        "streams independently.",
        "",
    ])
    (root / "REPORT.md").write_text("\n".join(lines))
    print("Saved:", root / "REPORT.md", "and", root / "pairs.csv")
    if len(valid_rows) < len(rows):
        print(f"WARNING: {len(rows)-len(valid_rows)} pairs had unverified cache states.")
    if len(valid_rows) < 3:
        print("WARNING: fewer than 3 verified pairs. Do not infer stable speedup.")


if __name__ == "__main__":
    main()
