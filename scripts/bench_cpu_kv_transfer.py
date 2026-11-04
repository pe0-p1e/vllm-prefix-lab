#!/usr/bin/env python3
"""Synthetic CPU KV H2D microbenchmark (NOT integrated vLLM prefill).

Reports independent wall-clock and CUDA event timing; no fake cache hits.
Each sample copies the full synthetic KV payload to GPU after allocation.
The shape models dense full-attention K and V across all decoder layers;
it is deliberately NOT claimed to match vLLM's paged per-layer layout.
"""
import argparse
import csv
import json
import math
import os
import pathlib
import statistics
import time

DTYPE_BYTES = {"fp16": 2, "bf16": 2, "fp32": 4}


def kv_bytes(tokens, layers, kv_heads, head_dim, dtype_bytes):
    if min(tokens, layers, kv_heads, head_dim, dtype_bytes) < 0:
        raise ValueError("negative dimension")
    return tokens * layers * 2 * kv_heads * head_dim * dtype_bytes


def percentile(values, q):
    v = sorted(values)
    pos = (len(v) - 1) * q
    low = int(pos)
    high = min(low + 1, len(v) - 1)
    return v[low] + (v[high] - v[low]) * (pos - low)


def profile_model(config):
    layers = int(config.num_hidden_layers)
    qheads = int(config.num_attention_heads)
    kvheads = int(getattr(config, "num_key_value_heads", qheads))
    dim = getattr(config, "head_dim", None)
    if dim is None:
        dim = int(config.hidden_size) // qheads
    return layers, kvheads, int(dim)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="/state/partition/model/Qwen3-0.6B")
    p.add_argument("--lengths", nargs="+", type=int, default=[2048, 4096, 8192, 16384, 32768])
    p.add_argument("--ratios", nargs="+", type=float, default=[0, .25, .5, .75, 1])
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--dtype", choices=tuple(DTYPE_BYTES), default="bf16")
    p.add_argument("--memory", choices=("pinned", "pageable"), default="pinned")
    p.add_argument("--max-gpu-fraction", type=float, default=.65)
    p.add_argument("--out", type=pathlib.Path, required=True)
    args = p.parse_args()
    if (args.block_size < 1 or args.repeats < 1 or args.warmups < 0
            or not 0 < args.max_gpu_fraction <= .8
            or any(x <= 0 for x in args.lengths)
            or any(not 0 <= x <= 1 for x in args.ratios)):
        p.error("Invalid benchmark configuration")
    # Import only after validating CLI, so the unit tests run without Torch.
    import torch
    from transformers import AutoConfig
    if not torch.cuda.is_available():
        p.error("CUDA unavailable")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").count(","):
        p.error("Only one physical GPU is supported")
    config = AutoConfig.from_pretrained(args.model_path, local_files_only=True,
                                       trust_remote_code=False)
    layers, heads, dim = profile_model(config)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
             "fp32": torch.float32}[args.dtype]
    free, total = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = {
        "kind": "synthetic_h2d_only_not_vllm_integrated",
        "model_path": args.model_path,
        "layers": layers, "kv_heads": heads, "head_dim": dim,
        "dtype": args.dtype, "memory": args.memory,
        "block_size": args.block_size, "repeats": args.repeats,
        "warmups": args.warmups, "gpu_free_bytes_start": free,
        "gpu_total_bytes": total, "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
    }
    print(json.dumps(metadata), flush=True)
    for length in args.lengths:
        for ratio in args.ratios:
            tokens = (int(length * ratio) // args.block_size) * args.block_size
            bytes_total = kv_bytes(tokens, layers, heads, dim, DTYPE_BYTES[args.dtype])
            if tokens == 0:
                rows.append(dict(length=length, ratio=ratio, hit_tokens=0,
                                 bytes=0, status="no_transfer", memory=args.memory))
                continue
            free_now, _ = torch.cuda.mem_get_info()
            if bytes_total > min(free_now * .80, total * args.max_gpu_fraction):
                rows.append(dict(length=length, ratio=ratio, hit_tokens=tokens,
                                 bytes=bytes_total, status="skipped_gpu_capacity",
                                 memory=args.memory))
                print("SKIP GPU capacity", length, ratio, bytes_total, flush=True)
                continue
            # Single contiguous buffer approximates aggregate KV payload size.
            # Shape is not a statement about the physical vLLM KV block layout.
            shape = (tokens, layers, 2, heads, dim)
            try:
                host = torch.empty(shape, dtype=dtype,
                                   device="cpu", pin_memory=args.memory == "pinned")
                host.fill_(.125)
                device = torch.empty(shape, dtype=dtype, device="cuda:0")
            except (RuntimeError, MemoryError) as exc:
                rows.append(dict(length=length, ratio=ratio, hit_tokens=tokens,
                                 bytes=bytes_total, status="allocation_failed",
                                 error=str(exc)[:500], memory=args.memory))
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                continue
            stream = torch.cuda.Stream()
            wall_ms, event_ms = [], []
            for index in range(args.warmups + args.repeats):
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter_ns()
                with torch.cuda.stream(stream):
                    start.record(stream)
                    device.copy_(host, non_blocking=args.memory == "pinned")
                    end.record(stream)
                end.synchronize()
                elapsed_wall = (time.perf_counter_ns() - t0) / 1e6
                elapsed_event = start.elapsed_time(end)
                if index >= args.warmups:
                    wall_ms.append(elapsed_wall)
                    event_ms.append(elapsed_event)
            peak = torch.cuda.max_memory_allocated()
            for repeat, (wall, event) in enumerate(zip(wall_ms, event_ms)):
                rows.append(dict(
                    length=length, ratio=ratio, repeat=repeat,
                    hit_tokens=tokens, bytes=bytes_total,
                    wall_ms=round(wall, 4), cuda_event_ms=round(event, 4),
                    gib_per_s_event=round(bytes_total / (event / 1000) / 2**30, 3)
                    if event > 0 else None,
                    gib_per_s_wall=round(bytes_total / (wall / 1000) / 2**30, 3)
                    if wall > 0 else None,
                    gpu_peak_allocated_bytes=peak,
                    memory=args.memory, status="measured"))
            print(json.dumps({"length": length, "ratio": ratio, "bytes": bytes_total,
                              "h2d_median_wall_ms": statistics.median(wall_ms),
                              "h2d_median_cuda_ms": statistics.median(event_ms),
                              "h2d_p95_wall_ms": percentile(wall_ms, .95)}),
                  flush=True)
            del host, device
            torch.cuda.empty_cache()
    names = sorted(set().union(*(r.keys() for r in rows)))
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=names)
        w.writeheader()
        w.writerows(rows)
    (args.out.parent / (args.out.stem + ".metadata.json")).write_text(
        json.dumps(metadata, indent=2) + "\n")
    print("Results:", args.out, flush=True)


if __name__ == "__main__":
    main()
