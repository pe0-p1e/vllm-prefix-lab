# CPU synthetic KV cache benchmark (vLLM 0.32.0)

**Status: integration code implemented; not yet GPU-tested on 4090-02.**
This branch has a custom vLLM V1 KV Connector, not APC or native
KV offloading. No previous request is required. It declares a forced
cache hit of H=block_aligned(min(L*r, L-1)) token positions, places
synthetic CPU-resident K/V into the **actual GPU KV block IDs** and
then executes the rest of the prompt with the real Qwen3-0.6B model.

The cached K/V values are synthetic and not semantically correct. This
is a performance-only experiment; generated output must not be treated
as a correctness or accuracy result.

## Auditable files

- src/synthetic_cpu_connector.py: vLLM V1 KV connector, forced hit and
  real cache block injection, persistent CPU KV source buffers
- src/token_ids.py: generate exactly L token IDs, no text-length variation
- scripts/run_cpu_kv_matrix.py: isolated server per cell and repeat,
  start/stop own server process group, metric deltas, enforce checks
- scripts/summarize_cpu_kv_live.py: summary table and statistical quantiles
- scripts/bench_cpu_kv_transfer.py: separate standalone H2D copy control
- tests/: CPU-only math/metrics and input construction checks

Old APC and offload scripts are retained in branch exp/vllm-apc-baseline.

## Measurement definitions

Each row records: L, requested r and H, actually injected host tokens,
actual computed prefill tokens, host-hit Prometheus metric, prefill service
time, HTTP-streamed TTFT/E2E, server queue/inference/decode/TTFT, worker
wall-clock total load, CUDA-event H2D and paged-GPU scatter, CPU
preparation, total bytes, effective H2D bandwidth, GPU peak allocation.

vLLM request_prefill_time_seconds measures its **PREFILL phase**, not
an isolated GPU kernel. It may include KV load and scheduling overhead.
CUDA events measure copies and scatter individually; their durations
are not simply subtracted from prefill time or TTFT. CPU source creation
occurs on first (excluded) warmup and is reused for timed repeats.
The first request per cell always serves as warmup even if --warmups 0.

All metrics are from explicit instrumentation or vLLM counters; missing
required prefill/host/computed counters fail validation, not fabricated.
The 100% cell leaves the last block to recompute; actual hit ratio
should be reported as injected_hit_tokens/L, not literally 100%.

## Commands

```bash
ssh 4090-02
cd /state/partition/fyzhou/vllm-prefix-lab
git fetch origin
git switch exp/prefix-hit-matrix
git pull --ff-only
python3 -m compileall -q src scripts tests
python3 -m unittest discover -s tests -v
```

No sudo; existing venv and model only. Observe GPU availability first.
If GPU is shared, use --allow-shared but mark results CONTENDED and
do not treat these measurements as isolated-performance benchmarks.

```bash
/state/partition/fyzhou/venvs/vllm/bin/python \
  scripts/run_cpu_kv_matrix.py --gpu 0 \
  --lengths 2048 --ratios 0 .5 1 --repeats 2 --warmups 2
```

Expand after verifying the smoke test:
```bash
/state/partition/fyzhou/venvs/vllm/bin/python \
 scripts/run_cpu_kv_matrix.py --gpu 0 \
 --lengths 2048 4096 8192 16384 32768 65536 131072 \
 --ratios 0 .25 .5 .75 1 --repeats 10 --warmups 3
```

The run directory is saved in
`/state/partition/fyzhou/results/cpu-kv-live/LATEST`.
`matrix.csv` contains individual measurements.
```bash
R=$(cat /state/partition/fyzhou/results/cpu-kv-live/LATEST)
python3 scripts/summarize_cpu_kv_live.py "$R/matrix.csv" --out "$R/table.csv"
```

For long context beyond max_position_embeddings, the runner skips
the cell unless explicitly opted in with --allow-extended-context.
That opt-in sets VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 and **does not guarantee**
the model/VRAM can run 128K; failed cells retain logs and stop rather than
fabricating measurements. Observe server-reported KV token capacity.
Uninterrupted GPU memory capacity must fit the entire prompt's KV.
This build targets a **single GPU**, not distributed TP, and may be
unable to run 128K on the present 24GB device.

## Acceptance criteria (not claimed passed until on-server test)

1. Startup recognizes SyntheticCPUConnector; vLLM 0.32.0 and full
   attention LBHNC tensor shape detected; no unsupported cache group.
2. 0% case has no injected KV; forced case has one worker log per
   request, all model KV layers injected with valid GPU block IDs.
3. Actual host hits match forced H; prefill KV computed count equals
   L-H within one cache block. No content hash matching or prior
   same-prefix request occurs.
4. Host K/V buffers are retained across measured requests; H2D CUDA
   event records nonzero payload for H>0.
5. All rows pass verification; no OOM, failures, missing metrics, or
   unexpected preemptions. Raw per-layer files retained.
6. Repeat with an idle GPU; capture raw server logs and setup.json.
   One success on a shared GPU does not establish reproducibility.
