#!/usr/bin/env python3
"""Controlled vLLM CPU-vs-GPU KV reuse probe. No KV implementation here.

A(cold) -> A(hot) -> B1..Bn(evict A from GPU) -> A(revisit).
Token IDs are submitted to vLLM's standard /v1/completions endpoint.
Only source=host counter deltas prove a CPU KV hit.
"""
import argparse
import json
import math
import pathlib
import re
import time
import urllib.request

from transformers import AutoTokenizer


def http_json(opener, url, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
        method="POST" if body is not None else "GET",
    )
    return opener.open(request, timeout=240)


def read_counters(opener, base_url):
    with http_json(opener, base_url + "/metrics") as response:
        lines = response.read().decode("utf-8", "replace").splitlines()
    names = (
        "vllm:prompt_tokens_cached_by_source",
        "vllm:prefix_cache_hits",
        "vllm:external_prefix_cache_hits",
        "vllm:prefix_cache_queries",
        "vllm:external_prefix_cache_queries",
    )
    result = {}
    for line in lines:
        if line.startswith("#"):
            continue
        for name in names:
            if not line.startswith(name + "_total"):
                continue
            match = re.match(r"^[^ {\t]+(?:\{([^}]*)\})?\s+([-+0-9.eE]+)", line)
            if not match:
                continue
            labels = match.group(1) or ""
            src = re.search(r'(?:^|,)source="([^"]+)"', labels)
            key = name + (":" + src.group(1) if src else "")
            result[key] = result.get(key, 0.0) + float(match.group(2))
    return result


def differences(before, after):
    return {
        key: round(after.get(key, 0.0) - before.get(key, 0.0), 3)
        for key in sorted(set(before) | set(after))
        if after.get(key, 0.0) - before.get(key, 0.0) != 0
    }


def make_prefix(tokenizer, index, prefix_len):
    header = tokenizer.encode(
        f"\nDOCUMENT-IDENTIFIER-{index:07d}. Unique topic number {index}. ",
        add_special_tokens=False,
    )
    base = tokenizer.encode(
        " Language models answer questions about computer systems, memory "
        "hierarchies, attention, history, networking, and software engineering. "
        "A previous context may be reused when its token prefix is identical. ",
        add_special_tokens=False,
    )
    if not base or len(header) > prefix_len:
        raise ValueError("Prefix template/tokenizer incompatible with requested length")
    needed = prefix_len - len(header)
    return header + (base * math.ceil(needed / len(base)))[:needed]


def make_suffix(tokenizer, tag, suffix_len):
    base = tokenizer.encode(
        f"\nQUESTION-ID:{tag}. Explain the main idea briefly. ",
        add_special_tokens=False,
    )
    pad = tokenizer.encode(" Please be concise and accurate. ", add_special_tokens=False)
    needed = suffix_len - len(base)
    if needed < 0 or not pad:
        raise ValueError("Suffix length too short")
    return base + (pad * math.ceil(needed / len(pad)))[:needed]


def stream_completion(opener, base_url, token_ids, model, max_tokens):
    payload = {
        "model": model,
        "prompt": token_ids,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    start = time.perf_counter()
    arrivals = []
    with http_json(opener, base_url + "/v1/completions", payload) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            choices = event.get("choices") or []
            # Each non-final completion SSE event corresponds to a generated
            # token. Some tokens render as empty text, so do not filter on text.
            if choices and choices[0].get("finish_reason") is None:
                arrivals.append(time.perf_counter())
    end = time.perf_counter()
    if not arrivals:
        raise RuntimeError("No streamed output tokens received")
    count = len(arrivals)
    return {
        "ttft_ms": round((arrivals[0] - start) * 1000, 3),
        "inter_chunk_ms": round(
            (arrivals[-1] - arrivals[0]) * 1000 / (count - 1), 3
        ) if count > 1 else None,
        "e2e_ms": round((end - start) * 1000, 3),
        "stream_chunks": count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gpu", "cpu"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", default="qwen-tiered-lab")
    parser.add_argument("--base-url", default="http://127.0.0.1:18081")
    parser.add_argument("--prefix-len", type=int, default=8192)
    parser.add_argument("--suffix-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--fillers", type=int, required=True)
    parser.add_argument("--gpu-kv-tokens", type=int, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if not (1024 <= args.prefix_len and 1 <= args.fillers <= 32):
        parser.error("Unexpected prefix length or filler count")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    prefix_a = make_prefix(tokenizer, 0, args.prefix_len)

    def call(stage, index, suffix):
        pref = prefix_a if index == 0 else make_prefix(tokenizer, index, args.prefix_len)
        ids = pref + make_suffix(tokenizer, suffix, args.suffix_len)
        timing = stream_completion(
            opener, args.base_url, ids, args.served_model, args.output_len
        )
        print(
            f"{args.mode:>3} | {stage:<13} | TTFT={timing['ttft_ms']:>8.2f} ms "
            f"| inter-chunk={timing['inter_chunk_ms']} ms "
            f"| tokens={timing['stream_chunks']}", flush=True,
        )
        return {"stage": stage, "prefix_id": index, "input_tokens": len(ids), **timing}

    initial = read_counters(opener, args.base_url)
    events = [call("cold-A", 0, "cold")]
    before_hot = read_counters(opener, args.base_url)
    events.append(call("hot-A", 0, "hot"))
    after_hot = read_counters(opener, args.base_url)
    for index in range(1, args.fillers + 1):
        events.append(call("filler", index, f"filler-{index:03d}"))
    before_revisit = read_counters(opener, args.base_url)
    events.append(call("revisit-A", 0, "revisit"))
    after_revisit = read_counters(opener, args.base_url)

    hot = differences(before_hot, after_hot)
    revisit = differences(before_revisit, after_revisit)
    host_hit = revisit.get("vllm:prompt_tokens_cached_by_source:host", 0.0)
    device_hit = revisit.get("vllm:prompt_tokens_cached_by_source:device", 0.0)
    external_hit = revisit.get("vllm:external_prefix_cache_hits", 0.0)
    result = {
        "mode": args.mode,
        "prefix_len": args.prefix_len,
        "suffix_len": args.suffix_len,
        "output_len": args.output_len,
        "fillers": args.fillers,
        "gpu_kv_capacity_tokens": args.gpu_kv_tokens,
        "events": events,
        "metrics_total_delta": differences(initial, after_revisit),
        "hot_metrics_delta": hot,
        "revisit_metrics_delta": revisit,
        "revisit_host_cached_tokens": host_hit,
        "revisit_gpu_cached_tokens": device_hit,
        "revisit_external_cached_tokens": external_hit,
        "cpu_hit_verified": bool(args.mode == "cpu" and host_hit > 0),
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved:", args.out)
    if args.mode == "cpu":
        if host_hit > 0:
            print(f"VERIFIED CPU HOST KV HIT: {host_hit:.0f} tokens")
        else:
            print("CPU HOST HIT NOT VERIFIED. Do not label this as CPU recovery.")
    else:
        print(f"GPU baseline revisit: device_cached={device_hit:.0f} tokens")


if __name__ == "__main__":
    main()
