# CPU KV benchmark: measurement definitions

The primary target is a synthetic CPU-resident KV payload injected into
vLLM's active paged GPU KV cache for exactly H prefix tokens, then running
the remaining L-H tokens through real prefill. **That integrated runner is
NOT implemented yet.** Do not label the existing GPU APC matrix as CPU KV.
Do not label the standalone copy benchmark as prefill.

## What is implemented now

`scripts/bench_cpu_kv_transfer.py` performs a *standalone* CPU→GPU
CUDA copy of a dense synthetic K/V payload. It computes bytes from the model
config using: H × layers × 2 (K/V) × num_kv_heads × head_dim × dtype_bytes.
It allocates CPU source memory and GPU destination memory, and records:
- actual aligned H, planned payload bytes, memory source type
- CUDA event H2D time: duration of the copy on the CUDA stream
- CPU wall time through stream completion, including submission/sync
- effective bandwidth (GiB/s) using both timing denominators
- per-sample GPU peak allocated memory, repeat ID, CUDA/PyTorch versions
- missing/skipped GPU allocation cells (not fabricated timings)

These timings are NOT network transport, not CPU cache lookup, not vLLM
KV block scatter/gather, not remaining prefill, and not TTFT. The contiguous
tensor copy is an upper-bound-oriented hardware throughput microbenchmark
which may be much faster than a real paged per-layer KV load. Source
contents intentionally carry no language semantics. The output is a raw CSV.

## Commands

Run only with a GPU you are authorized to use. Do not share latency claims
from an actively contended GPU. No sudo or automatic dependency installation.

```bash
cd /state/partition/fyzhou/vllm-prefix-lab
git fetch origin && git switch exp/prefix-hit-matrix && git pull --ff-only
python3 -m unittest discover -s tests -v
export CUDA_VISIBLE_DEVICES=0  # replace with an authorized GPU
VENV=/state/partition/fyzhou/venvs/vllm
mkdir -p /state/partition/fyzhou/results/cpu-kv
"$VENV/bin/python" scripts/bench_cpu_kv_transfer.py \
 --lengths 2048 4096 8192 16384 32768 \
 --ratios 0 .25 .5 .75 1 \
 --repeats 10 --warmups 3 --memory pinned \
 --out /state/partition/fyzhou/results/cpu-kv/h2d-pinned.csv
```

Pageable host comparison: change `--memory pageable` and output path.
For 64K and 128K add those lengths only when model context and memory
budgets are separately validated. Test can skip cells that exceed GPU
memory budget; a pinned allocation can fail due to OS limits.

## Required integrated metrics (future work)

Per request: L, requested H, actually inserted and consumed KV tokens,
CPU bytes, GPU KV slots occupied, per-layer block IDs, hash lookup disabled,
host cache preparation excluded from timing, H2D CUDA events, H2D wall,
load ready barrier, remaining prefill (model execution GPU timing),
scheduler/queue overhead, first-token decode, full TTFT, CPU RAM,
GPU VRAM peak, SM busy, copy engine utilization, PCIe link width/speed,
GPU clock and power, CPU memory/page faults, errors, p50/p95/p99 and n.
Measure transfer and computation on the correct CUDA streams; if overlapped,
report individual component times and makespan without adding them.

Acceptance criteria: no content hash reuse, exact block-aligned forced H,
correct block table addresses and KV layout, no uninitialized reads, CPU
residency verified, CUDA-event synchronized timing, all layers copied, and
L-H real tokens computed. The vLLM V1 KVConnector interface is a possible
integration point, but should be pinned to vLLM 0.32.0 and validated by
on-server integration tests before claiming functionality.
