#!/usr/bin/env python3
"""Isolated decode vs CPU-KV-load interference probe using native vLLM.

The treatment request restores a verified host-resident prefix while a
separate 768-token decode is in flight. We measure both streams separately.
No external packages beyond the existing vLLM lab venv are required.
"""
import argparse
import json
import pathlib
import statistics
import threading
import time
import urllib.request

from transformers import AutoTokenizer

from tiered_probe import (
    differences,
    http_json,
    make_prefix,
    make_suffix,
    read_counters,
    stream_completion,
)


def stream_trace(opener, base_url, token_ids, served_model,
                 output_len, trigger_at=None, trigger=None):
    payload = {
        "model": served_model, "prompt": token_ids,
        "max_tokens": output_len, "temperature": 0,
        "ignore_eos": True, "stream": True,
    }
    start = time.perf_counter()
    arrivals = []
    with http_json(opener, base_url + "/v1/completions", payload) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            chunk = line[5:].strip()
            if chunk == b"[DONE]":
                break
            obj = json.loads(chunk)
            if "error" in obj:
                raise RuntimeError(str(obj["error"]))
            choices = obj.get("choices") or []
            if choices and choices[0].get("finish_reason") is None:
                arrivals.append(time.perf_counter())
                if trigger is not None and len(arrivals) >= trigger_at:
                    trigger.set()
    end = time.perf_counter()
    if len(arrivals) < output_len // 2:
        raise RuntimeError(
            f"Decode stream too short: {len(arrivals)} vs requested {output_len}"
        )
    return {
        "start": start, "end": end, "arrivals": arrivals,
        "stream_chunks": len(arrivals),
        "ttft_ms": round(1000 * (arrivals[0] - start), 3),
    }


def distribution(values):
    if not values:
        return {"n": 0, "median_ms": None, "p95_ms": None,
                "mean_ms": None, "max_ms": None}
    values = sorted(values)
    def p(q):
        idx = (len(values) - 1) * q
        lo = int(idx)
        hi = min(lo+1, len(values)-1)
        return values[lo] * (hi-idx) + values[hi] * (idx-lo)
    return {
        "n": len(values),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(p(.95), 3),
        "mean_ms": round(statistics.mean(values), 3),
        "max_ms": round(max(values), 3),
    }


def intervals_in_window(arrivals, lower=None, upper=None):
    intervals = []
    for previous, current in zip(arrivals, arrivals[1:]):
        if lower is not None and previous < lower:
            continue
        if upper is not None and current > upper:
            continue
        intervals.append(1000 * (current - previous))
    return intervals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18082")
    parser.add_argument("--served-model", default="qwen-tiered-lab")
    parser.add_argument("--prefix-len", type=int, default=8192)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--decode-len", type=int, default=768)
    parser.add_argument("--trigger-at", type=int, default=96)
    parser.add_argument("--fillers", type=int, required=True)
    parser.add_argument("--gpu-kv-tokens", type=int, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.trigger_at < 32 or args.decode_len < args.trigger_at + 256:
        parser.error("Need sufficient tokens before and after overlap")
    if args.fillers < 2 or args.fillers > 32:
        parser.error("fillers must be in 2..32")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    prefix_a = make_prefix(tokenizer, 0, args.prefix_len)
    suffix_cold = make_suffix(tokenizer, "overlap-cold", args.suffix_len)

    print("Preparing CPU cache: cold A + eviction prefixes", flush=True)
    cold = stream_completion(
        opener, args.base_url, prefix_a + suffix_cold,
        args.served_model, 8,
    )
    for i in range(1, args.fillers + 1):
        ids = make_prefix(tokenizer, i, args.prefix_len)
        ids += make_suffix(tokenizer, f"overlap-filler-{i}", args.suffix_len)
        stream_completion(opener, args.base_url, ids, args.served_model, 8)
    print(f"Prepared A + {args.fillers} fillers; cold TTFT={cold['ttft_ms']} ms",
          flush=True)

    # Small and DISTINCT decode prompts. Neither shares prefix A.
    def small_prompt(case):
        return tokenizer.encode(
            f"\nBACKGROUND DECODE CASE {case}: Explain computing concepts in "
            "detail, sentence by sentence. Continue until the token limit. ",
            add_special_tokens=False,
        )

    print(f"Baseline: decode {args.decode_len} tokens alone", flush=True)
    baseline = stream_trace(
        opener, args.base_url, small_prompt("baseline"),
        args.served_model, args.decode_len,
    )

    # Make sure source=host counters and transfer stats are sampled before
    # overlap. There are no other host-backed requests in this experiment.
    before = read_counters(opener, args.base_url)
    triggered = threading.Event()
    treatment_holder = {}
    errors = []

    def decode_background():
        try:
            treatment_holder["stream"] = stream_trace(
                opener, args.base_url, small_prompt("treatment"),
                args.served_model, args.decode_len,
                trigger_at=args.trigger_at, trigger=triggered,
            )
        except Exception as exc:
            errors.append(repr(exc))
            triggered.set()

    thread = threading.Thread(target=decode_background, daemon=True)
    print(f"Overlap: start decoder, inject CPU KV reload after "
          f"{args.trigger_at} chunks", flush=True)
    thread.start()
    if not triggered.wait(60):
        raise RuntimeError("Background decode did not reach injection point")
    if errors:
        raise RuntimeError(f"Background decode failed: {errors}")
    load_start = time.perf_counter()
    revisit_ids = prefix_a + make_suffix(
        tokenizer, "overlap-revisit", args.suffix_len,
    )
    reload_result = stream_completion(
        opener, args.base_url, revisit_ids, args.served_model, 8,
    )
    load_end = time.perf_counter()
    thread.join(timeout=60)
    if thread.is_alive() or errors or "stream" not in treatment_holder:
        raise RuntimeError(f"Background decode incomplete: {errors}")
    treatment = treatment_holder["stream"]
    after = read_counters(opener, args.base_url)
    changes = differences(before, after)
    host_hits = changes.get("vllm:prompt_tokens_cached_by_source:host", 0)
    gpu_hits = changes.get("vllm:prompt_tokens_cached_by_source:device", 0)
    load_bytes = changes.get("vllm:kv_offload_load_bytes", 0)
    load_cuda_time = changes.get("vllm:kv_offload_load_time", None)
    valid = host_hits >= 0.8*args.prefix_len
    overlap_window_exists = (
        treatment["arrivals"][0] < load_start
        and treatment["arrivals"][-1] > load_end
    )
    before_intervals = intervals_in_window(
        treatment["arrivals"], upper=load_start
    )
    during_intervals = intervals_in_window(
        treatment["arrivals"], lower=load_start, upper=load_end
    )
    after_intervals = intervals_in_window(
        treatment["arrivals"], lower=load_end
    )
    result = {
        "config": vars(args) | {"out": str(args.out)},
        "cold_A": cold,
        "baseline": {k:v for k,v in baseline.items() if k!="arrivals"},
        "treatment": {k:v for k,v in treatment.items() if k!="arrivals"},
        "revisit": reload_result,
        "cache_deltas": changes,
        "host_cached_tokens": host_hits,
        "gpu_cached_tokens": gpu_hits,
        "cuda_load_bytes": load_bytes,
        "cuda_load_time_sum_seconds": load_cuda_time,
        "cpu_hit_verified": valid,
        "actual_overlap_verified": overlap_window_exists,
        "before": distribution(before_intervals),
        "during": distribution(during_intervals),
        "after": distribution(after_intervals),
        "baseline_all": distribution(intervals_in_window(baseline["arrivals"])),
        "treatment_all": distribution(intervals_in_window(treatment["arrivals"])),
        "load_window_ms": round(1000*(load_end-load_start),3),
        "load_start_time": load_start,
        "load_end_time": load_end,
        "decode_treatment_intervals_ms": [
            round(1000*(b-a),3) for a,b in zip(
                treatment["arrivals"], treatment["arrivals"][1:]
            )
        ],
        "decode_baseline_intervals_ms": [
            round(1000*(b-a),3) for a,b in zip(
                baseline["arrivals"], baseline["arrivals"][1:]
            )
        ],
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Host cached tokens={host_hits}, host valid={valid}; "
        f"decode overlap valid={overlap_window_exists}; "
        f"revisit TTFT={reload_result['ttft_ms']} ms",
        flush=True,
    )
    print(
        f"Decode ITL medians (ms): baseline={result['baseline_all']['median_ms']}, "
        f"before={result['before']['median_ms']}, "
        f"during={result['during']['median_ms']}, "
        f"after={result['after']['median_ms']}",
        flush=True,
    )
    print(
        f"Offload H2D: {load_bytes} bytes, "
        f"CUDA-event accumulated time={load_cuda_time} s; "
        "not end-to-end wall time",
        flush=True,
    )
    print("Saved:", args.out, flush=True)
    if not valid or not overlap_window_exists or result["during"]["n"] < 3:
        print("WARNING: unverified host hit / insufficient overlapping decoder "
              "intervals. Do not infer decoder interference.", flush=True)


if __name__ == "__main__":
    main()
