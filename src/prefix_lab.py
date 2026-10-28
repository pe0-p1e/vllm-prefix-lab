#!/usr/bin/env python3
"""Controlled request construction and metrics for vLLM APC.

Input consists of token-ID lists, not tokenized strings. The test request has
a deliberate common prefix with a primer request. Only device-hit metrics
(not the requested overlap) may validate that the cache was actually used.
"""
import argparse
import json
import pathlib
import re
import statistics
import time
import urllib.request


def align_hit(length, ratio, block):
    if length < 2 or block < 1 or not 0 <= ratio <= 1:
        raise ValueError("invalid length, ratio or block size")
    return (int(length * ratio) // block) * block


def make_ids(tokenizer, length, tag):
    """Return precisely length valid IDs with a distinct beginning per tag."""
    header = tokenizer.encode("PREFIX-LAB DOCUMENT %09d. " % tag,
                              add_special_tokens=False)
    body = tokenizer.encode(
        " Context memory cache attention inference benchmarking. ",
        add_special_tokens=False)
    if not body or len(header) >= length:
        raise ValueError("tokenizer or length incompatible with input")
    return header + (body * ((length + len(body) - 1) // len(body)))[:length-len(header)]


def build_pair(tokenizer, length, hit, tag):
    test = make_ids(tokenizer, length, tag * 2 + 1)
    if not hit:
        return None, test
    if hit == length:
        return list(test), test
    # A different suffix, but the first divergent token MUST differ.
    # Construct from a token known unequal at the boundary.
    alternate = make_ids(tokenizer, length, tag * 2 + 2)
    if alternate[hit] == test[hit]:
        alternate[hit] = next(i for i in range(tokenizer.vocab_size)
                              if i != test[hit] and i not in
                              set(tokenizer.all_special_ids))
    primer = test[:hit] + alternate[hit:]
    assert len(primer) == length and primer[:hit] == test[:hit]
    assert primer[hit] != test[hit]
    return primer, test


def get(url, opener):
    with opener.open(url, timeout=15) as response:
        return response.read().decode("utf-8")


def cache_counters(opener, base):
    """Extract source=device cumulative tokens. None means absent metric."""
    raw = get(base + "/metrics", opener)
    values = []
    for line in raw.splitlines():
        if line.startswith("#") or not line.startswith(
                "vllm:prompt_tokens_cached_by_source_total"):
            continue
        match = re.match(r'[^\s{]+(?:\{([^}]*)\})?\s+([-+0-9.eE]+)', line)
        if match and re.search(r'(?:^|,)source="device"(?:,|$)', match[1] or ""):
            values.append(float(match[2]))
    return sum(values) if values else None


def one_token(opener, base, model, ids):
    payload = json.dumps({
        "model": model, "prompt": ids, "max_tokens": 1,
        "temperature": 0, "stream": True, "ignore_eos": True
    }).encode()
    request = urllib.request.Request(
        base + "/v1/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    first = None
    with opener.open(request, timeout=600) as response:
        for line in response:
            if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                obj = json.loads(line[5:].strip())
                if "error" in obj:
                    raise RuntimeError(str(obj["error"]))
                if obj.get("choices") and obj["choices"][0].get("finish_reason") is None:
                    first = time.perf_counter()
                    break
    if first is None:
        raise RuntimeError("No first generated token")
    return round(1000 * (first - start), 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--length", type=int, required=True)
    p.add_argument("--ratio", type=float, required=True)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--mode", choices=("on", "off"), required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    args = p.parse_args()
    if args.repeats < 1 or args.length < 128:
        p.error("repeats >=1 and length>=128 required")
    if args.mode == "off" and args.ratio != 0:
        p.error("APC-off only supports baseline ratio 0")
    hit = align_hit(args.length, args.ratio, args.block_size)
    if hit and hit < 64:
        p.error("hit prefix too short to construct reliably")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    rows = []
    for i in range(args.repeats):
        primer, query = build_pair(tokenizer, args.length, hit, i + 10000)
        if args.mode == "on" and primer is not None:
            one_token(opener, args.base_url, "prefix-lab", primer)
        before = cache_counters(opener, args.base_url)
        ttft = one_token(opener, args.base_url, "prefix-lab", query)
        after = cache_counters(opener, args.base_url)
        actual = None if before is None or after is None else after - before
        # In vLLM, not every last block can necessarily be treated as a hit
        # (especially fully cached inputs); report rather than conceal.
        valid = actual is not None and abs(actual - hit) <= args.block_size
        if args.mode == "off":
            # Some disabled-APC versions do not expose the cache counter.
            valid = actual is None or actual == 0
        rows.append(dict(repeat=i, requested_hit_tokens=hit,
                         actual_hit_tokens=actual, ttft_ms=ttft,
                         verified=valid))
        print(json.dumps(rows[-1]), flush=True)
    out = dict(mode=args.mode, length=args.length, requested_ratio=args.ratio,
               hit_tokens=hit, block_size=args.block_size,
               verified=all(x["verified"] for x in rows), samples=rows,
               median_ttft_ms=statistics.median(x["ttft_ms"] for x in rows))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    if not out["verified"]:
        raise SystemExit("Unverified hit count: results retained, cell rejected")


if __name__ == "__main__":
    main()
