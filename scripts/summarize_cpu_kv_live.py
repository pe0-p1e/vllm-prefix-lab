#!/usr/bin/env python3
"""Produce one row per (input tokens, forced CPU hit ratio) with no invented data."""
import argparse
import csv
import pathlib
import statistics
from collections import defaultdict


def quantile(values, q):
    values = sorted(values)
    position = (len(values)-1)*q
    low = int(position)
    high = min(low+1,len(values)-1)
    return values[low]+(values[high]-values[low])*(position-low)


def median(rows, field):
    vals = [float(x[field]) for x in rows if x.get(field) not in (None, "")]
    return statistics.median(vals) if vals else ""


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("matrix",type=pathlib.Path)
    parser.add_argument("--out",type=pathlib.Path,default=None)
    args=parser.parse_args()
    grouped=defaultdict(list)
    failed=[]
    with args.matrix.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] != "measured" or row["verified"] != "True":
                failed.append(row)
                continue
            key=(int(row["length"]),float(row["requested_ratio"]),row["mode"])
            grouped[key].append(row)
    headers=["input_tokens","requested_cpu_hit_ratio","actual_cpu_hit_tokens",
             "recomputed_prefill_tokens","mode","repeats",
             "prefill_service_median_ms","prefill_service_p95_ms",
             "ttft_median_ms","ttft_p95_ms","ttft_p99_ms",
             "h2d_cuda_median_ms","h2d_cuda_p95_ms",
             "gpu_scatter_median_ms","worker_load_wall_median_ms",
             "cpu_kv_bytes","h2d_effective_bandwidth_median_gib_s",
             "gpu_peak_allocated_bytes","ttft_speedup_vs_no_cache","status"]
    baseline={length:median(rows,"ttft_ms")
              for (length,ratio,mode),rows in grouped.items()
              if mode=="disable" and ratio==0}
    output=args.out or args.matrix.with_name("table.csv")
    with output.open("w",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=headers)
        writer.writeheader()
        for (length,ratio,mode),rows in sorted(grouped.items()):
            ttft=[float(r["ttft_ms"]) for r in rows]
            prefill=[float(r["prefill_service_ms"]) for r in rows]
            h2d=[float(r["h2d_cuda_ms"]) for r in rows]
            speed=baseline.get(length)
            writer.writerow(dict(
                input_tokens=length,requested_cpu_hit_ratio=ratio,
                actual_cpu_hit_tokens=median(rows,"injected_hit_tokens"),
                recomputed_prefill_tokens=median(rows,"actual_prefill_kv_tokens"),
                mode=mode,repeats=len(rows),
                prefill_service_median_ms=statistics.median(prefill),
                prefill_service_p95_ms=quantile(prefill,.95),
                ttft_median_ms=statistics.median(ttft),
                ttft_p95_ms=quantile(ttft,.95),
                ttft_p99_ms=quantile(ttft,.99),
                h2d_cuda_median_ms=statistics.median(h2d),
                h2d_cuda_p95_ms=quantile(h2d,.95),
                gpu_scatter_median_ms=median(rows,"scatter_cuda_ms"),
                worker_load_wall_median_ms=median(rows,"worker_load_wall_ms"),
                cpu_kv_bytes=median(rows,"kv_bytes"),
                h2d_effective_bandwidth_median_gib_s=median(rows,"h2d_gib_per_s"),
                gpu_peak_allocated_bytes=median(rows,"peak_gpu_allocated_bytes"),
                ttft_speedup_vs_no_cache=(speed/statistics.median(ttft))
                if speed else "",
                status="verified"))
        for x in failed:
            writer.writerow(dict(input_tokens=x.get("length",""),
                                 requested_cpu_hit_ratio=x.get("requested_ratio",""),
                                 mode=x.get("mode",""),status=x["status"]))
    print("Table:",output)
    if failed:
        raise SystemExit(f"{len(failed)} failed/unverified rows; table retained with status")


if __name__ == "__main__":
    main()
