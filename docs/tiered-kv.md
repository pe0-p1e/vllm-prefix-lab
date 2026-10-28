# CPU–GPU hierarchical KV cache baseline (vLLM 0.32.0)

This experiment studies long-prefix KV reuse after GPU HBM eviction, using
vLLM's built-in native CPU KV offloading. No custom cache, Mooncake, proxy,
or global package installation is involved.

## Run on 4090-02

From the repo root, with GPU 1 genuinely available:

```bash
source scripts/env.sh
source scripts/proxy_env.sh off
nvidia-smi
CUDA_VISIBLE_DEVICES=1 bash scripts/run_tiered.sh 2>&1 | tee results/tiered-console.log

RUN_DIR=$(cat results/TIERED_LATEST)
cat "$RUN_DIR/REPORT.md"
cat "$RUN_DIR/requests.csv"
```

This **does not overwrite** the earlier APC experiment's results/LATEST.
It creates results/tiered-<UTC timestamp>/ and a separate TIERED_LATEST.

### Conditions

- GPU mode: vLLM APC enabled; cached KV resides in GPU HBM only.
- CPU mode: same APC settings plus native CPU offloading:
  --kv-offloading-backend native --kv-offloading-size 12
- Default prefix length: 8192 tokens; new suffix: 64 tokens.
- Default output: 256 tokens, rather than 32, to measure streamed
  inter-chunk latency in addition to TTFT.
- max model length: 16384; GPU memory utilization: 0.28.
- Filler requests are generated using distinct exact token-ID prefixes.
- Number of fillers is computed from actual vLLM GPU KV cache capacity and
  fixed across the two conditions. This attempts to evict prefix A from HBM.
- Source=host counter delta on the final replay is **required** for a verified
  CPU KV cache hit. A low TTFT without host hit is not sufficient evidence.
- Neither source=host nor device-cache status is inferred from latency alone.
- Free host memory is checked before requesting 12 GiB of pinned-DRAM cache;
  the script aborts rather than consuming another user's GPU or RAM.

### Request trace

1. Cold A: fill cache from a long prefix.
2. Immediate A again, with a different suffix: expect an HBM cache hit.
3. B1, B2, ..., BN: distinct long prefixes, enough to exceed GPU KV capacity.
4. Revisit A with another suffix. In GPU-only mode it should be a cache miss
   (check source=device); in CPU mode it must show source=host KV reuse.

Results contain request timings and raw metric deltas in gpu/probe.json and
cpu/probe.json, server logs, plus requests.csv and REPORT.md.

### Limitations

The primary test is **sequential**, with one final replay per condition. It
tests functionality and a preliminary speed comparison, not sustained
concurrency or statistically stable tail latency. Inter-chunk timing measured
over streamed HTTP responses is only an approximation of TPOT.
Further experiments should repeat confirmed host hits under competing decode
requests, record true per-token server latencies, and vary CPU cache size,
prefix length, and host-to-device transfer pressure.

### Tunables

Environment variables, not persistent global settings:

```bash
# Only after the 8K validation, to test 16K prefixes:
TIER_PREFIX_LEN=16384 TIER_MAX_MODEL_LEN=32768 \
TIER_CPU_GIB=16 CUDA_VISIBLE_DEVICES=1 bash scripts/run_tiered.sh

# To run a single condition or change the memory budget:
TIER_MODE=cpu TIER_CPU_GIB=12 CUDA_VISIBLE_DEVICES=1 bash scripts/run_tiered.sh
```

Other knobs: TIER_GPU_UTIL, TIER_OUTPUT_LEN, TIER_SUFFIX_LEN, TIER_FILLERS,
TIER_PORT, MODEL_OVERRIDE. Using more CPU memory on a shared server requires
sufficient available host memory. When host hit is zero, do not report an
offloading speedup.

Official documentation for vLLM 0.32.0:
https://docs.vllm.ai/en/v0.32.0/features/kv_offloading_usage/

### Optional shared GPU: functional test only

The default selector requires an almost-idle GPU. On a shared lab node it
may be acceptable to use **only spare VRAM**, without touching another user's
process, when the model and cache budgets fit. Use explicit opt-in and a
generous reserve; the guard is checked before **each** new vLLM server:

~~~bash
CUDA_VISIBLE_DEVICES=1 LAB_ALLOW_SHARED_GPU=1 \
TIER_GPU_UTIL=0.22 LAB_GPU_RESERVE_MIB=4096 \
bash scripts/run_tiered.sh
~~~

The shared-mode guard requires:

    free VRAM >= (total VRAM * TIER_GPU_UTIL) + LAB_GPU_RESERVE_MIB

If this is not satisfied the experiment refuses to start. The guard is a
snapshot rather than a VRAM reservation: the other process may grow and
trigger OOM later. Leave room for it and do not terminate or modify its
process. Confirm that GPU sharing is permitted by lab policy.

**Results gathered concurrently with unrelated GPU workloads are exploratory
only**, suitable for checking host-cache functionality. Their TTFT, ITL and
throughput are confounded by contention. For a publishable GPU-vs-CPU
latency comparison, repeat on an otherwise idle GPU and log device
utilization throughout the experiment.

### Repeated paired measurements

After the single A-B-A experiment has verified host hits and GPU eviction,
collect independent pairs and alternate run order. Each pair starts fresh
vLLM services; only validated full-prefix host hits are included in summaries.

~~~bash
source scripts/env.sh
source scripts/proxy_env.sh off

# Prefer an otherwise idle GPU; defaults to 4 paired repetitions.
CUDA_VISIBLE_DEVICES=1 TIER_REPEATS=4 \
bash scripts/repeat_tiered.sh 2>&1 | tee results/repeat-console.log

ROOT=$(cat results/TIERED_REPEAT_LATEST)
cat "$ROOT/REPORT.md"
cat "$ROOT/pairs.csv"
~~~

If the GPU is legitimately shared, the explicit spare-VRAM opt-in still
works, but numbers are exploratory until repeated with no competing jobs:

~~~bash
CUDA_VISIBLE_DEVICES=1 LAB_ALLOW_SHARED_GPU=1 \
TIER_GPU_UTIL=0.22 LAB_GPU_RESERVE_MIB=4096 TIER_REPEATS=4 \
bash scripts/repeat_tiered.sh
~~~

The paired summary reports the median and observed range, **not** a
confidence interval. A confirmed host hit only establishes the cache
mechanism; it does not rule out shared-GPU interference with TTFT or TPOT.
For a meaningful host-vs-HBM study also compare 8K/16K/24K prefixes and
concurrent decode workloads.

### Active Decode interference probe

This optional experiment is **separate** from the paired A-B-A benchmark.
It starts one native-CPU-offloading vLLM server, fills CPU KV for prefix A,
evicts A from GPU HBM, then measures:

1. A long Decode stream without a competing request (baseline).
2. Another long Decode stream, injecting A's host-KV restoration after 96
   streamed chunks (the treatment). The two requests genuinely overlap.
3. The background stream's before/during/after inter-chunk intervals,
   the restoration request's TTFT, host-token-hit source, and native
   CPU-to-GPU offloading byte and CUDA-event time counters.

On an idle GPU:

~~~bash
cd /state/partition/fyzhou/vllm-prefix-lab
git pull --ff-only origin exp/vllm-apc-baseline
source scripts/env.sh
source scripts/proxy_env.sh off
nvidia-smi

CUDA_VISIBLE_DEVICES=1 TIER_MODE=cpu TIER_RUN_OVERLAP=1 \
TIER_GPU_UTIL=0.22 OVERLAP_DECODE_LEN=768 OVERLAP_TRIGGER_AT=96 \
bash scripts/run_tiered.sh 2>&1 | tee results/overlap-console.log

RUN_DIR=$(cat results/OVERLAP_LATEST)
cat "$RUN_DIR/OVERLAP_REPORT.md"
cat "$RUN_DIR/cpu/overlap.log"
~~~

This reuses run_tiered.sh's GPU selection and host RAM checks. It writes
OVERLAP_LATEST without overwriting the repeated-pair result pointer.
If LAB_ALLOW_SHARED_GPU=1 is needed, leave sufficient VRAM reserve as
described above. Do not compare interference latencies on a busy GPU.

Interpretation limits: server-side prefill and scheduling may coincide with
DMA. The foreground request's full submission-to-response window is
not the exact DMA interval. The vLLM native kv_offload_load_time counter
reflects the sum of CUDA-event timing of completed H2D transfers, not the
end-to-end user-perceived delay. The single treatment run is a functional
check; repeat with randomized order and comparable GPU-only control
requests before drawing causal performance conclusions.
