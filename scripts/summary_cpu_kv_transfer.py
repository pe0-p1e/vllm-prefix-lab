#!/usr/bin/env python3
"""Aggregate measured H2D microbenchmark rows without fabricating missing cells."""
import argparse
import csv
import pathlib
import statistics
from collections import defaultdict


def quantile(values, q):
    ordered = sorted(values)
    i = (len(ordered)-1)*q
    lower = int(i)
    upper = min(lower+1, len(ordered)-1)
    return ordered[lower]+(ordered[upper]-ordered[lower])*(i-lower)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=pathlib.Path)
    p.add_argument("--out", type=pathlib.Path, required=True)
    args = p.parse_args()
    groups = defaultdict(list)
    skips = []
    with args.input.open(newline="") as f:
        for r in csv.DictReader(f):
            key = (int(r["length"]), float(r["ratio"]), r["memory"])
            if r["status"] != "measured":
                skips.append((key, r["status"]))
                continue
            groups[key].append(r)
    fields = ["input_tokens","requested_ratio","cpu_memory_type","n",
              "actual_hit_tokens","bytes","h2d_wall_median_ms",
              "h2d_wall_p95_ms","h2d_wall_p99_ms",
              "h2d_cuda_median_ms","h2d_cuda_p95_ms",
              "h2d_bandwidth_median_gib_s","gpu_peak_allocated_bytes","status"]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for key, v in sorted(groups.items()):
            wall=[float(x["wall_ms"]) for x in v]
            cuda=[float(x["cuda_event_ms"]) for x in v]
            bw=[float(x["gib_per_s_event"]) for x in v]
            w.writerow(dict(input_tokens=key[0],requested_ratio=key[1],
                cpu_memory_type=key[2],n=len(v),actual_hit_tokens=v[0]["hit_tokens"],
                bytes=v[0]["bytes"],h2d_wall_median_ms=statistics.median(wall),
                h2d_wall_p95_ms=quantile(wall,.95),h2d_wall_p99_ms=quantile(wall,.99),
                h2d_cuda_median_ms=statistics.median(cuda),
                h2d_cuda_p95_ms=quantile(cuda,.95),
                h2d_bandwidth_median_gib_s=statistics.median(bw),
                gpu_peak_allocated_bytes=max(int(x["gpu_peak_allocated_bytes"]) for x in v),
                status="measured"))
        for key,status in sorted(skips):
            w.writerow(dict(input_tokens=key[0], requested_ratio=key[1],
                            cpu_memory_type=key[2],status=status))
    print("Table:",args.out)


if __name__=="__main__":
    main()
