#!/usr/bin/env bash
# Extract the meaningful startup failure from a vLLM server log.
# Usage: bash scripts/diagnose_startup.sh /path/to/server.log
set -euo pipefail
[[ $# -eq 1 && -f "$1" ]] || { echo 'Usage: bash scripts/diagnose_startup.sh server.log' >&2; exit 2; }
LOG="$1"
export VLLM_STARTUP_LOG="$LOG"
python3 - <<'PY'
import os
import pathlib
import re

path = pathlib.Path(os.environ["VLLM_STARTUP_LOG"])
lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
if not lines:
    print("The server log is empty.")
    raise SystemExit(0)

patterns = [
    r"EngineCore.*failed to start",
    r"Traceback \(most recent call last\)",
    r"(?i)(?:cuda|driver|nvidia|ptx|nvcc|flashinfer|triton).*?(?:error|failed|unsupported|incompatible)",
    r"(?i)(?:error|exception|out of memory|oom|assertionerror|file.*not found|not enough memory)",
]
matches = [
    i for i, line in enumerate(lines)
    if any(re.search(p, line) for p in patterns)
]
# Prefer early EngineCore errors over the API server's generic last traceback.
engine_errors = [i for i in matches if "EngineCore" in lines[i] and "failed to start" in lines[i]]
primary = engine_errors[:2] if engine_errors else matches[:4]

print(f"Log: {path} ({len(lines)} lines)")
if not matches:
    print("No obvious error marker. Showing last 100 lines.")
    for i in range(max(0, len(lines)-100), len(lines)):
        print(f"{i+1:5d}: {lines[i]}")
else:
    intervals = []
    for i in primary:
        intervals.append((max(0, i-22), min(len(lines), i+85)))
    # Print the earliest likely root-cause, not just the tail's wrapper error.
    if matches[0] not in primary:
        i = matches[0]
        intervals.insert(0, (max(0, i-10), min(len(lines), i+35)))
    intervals.sort()
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    for lo, hi in merged[:3]:
        print(f"\n--- server.log lines {lo+1}-{hi} ---")
        for i in range(lo, hi):
            print(f"{i+1:5d}: {lines[i]}")
PY
