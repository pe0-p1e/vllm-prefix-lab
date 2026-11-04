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

## Installed 0.32.0 FlashAttention correction (2026-11-04)

The original 50% crash was a layout assertion, not an EngineCore scheduling
failure. Installed FlashAttention consumes `[blocks, kv_heads, block_size, 2*head_dim]`
and splits the final dimension after transposing heads/tokens. Accepting an
arbitrary 4D tensor with dimension 1 equal to 2 was incorrect. This connector now
uses `register_kv_caches` and validates the actual model dimensions. Each newly
prepared CPU source is fully read back from all selected GPU blocks during warmup;
measured requests reuse the CPU allocations and perform fresh copies and scatter.
All loads are synchronous before model forward; per-layer event synchronization
makes `wait_for_layer_load` a no-op. Eager mode is an explicit experimental control,
not evidence that CUDA graphs caused the original failure.

Run GPU injection checks explicitly (normal unit discovery skips them):

```sh
CUDA_VISIBLE_DEVICES=1 CPU_KV_GPU_TEST=1 /state/partition/fyzhou/venvs/vllm/bin/python -m unittest discover -s tests -p test_synthetic_injection.py
```

`--reuse-server` uses one cache-disabled engine and one CPU engine, with a fixed
maximum supported context across all cells. CPU hit counts are conveyed through
`vllm_xargs.synthetic_cpu_hit_tokens`, validated by the scheduler connector; each
cell still gets its own warmups. CPU buffers persist across requests within each
cell. This avoids repeated server initialization while retaining an independent
baseline. Do not compare shared-GPU timing as an isolated hardware benchmark.

```sh
/state/partition/fyzhou/venvs/vllm/bin/python scripts/run_cpu_kv_matrix.py --gpu 1 --gpu-util .25 --allow-shared --reuse-server --warmups 3 --repeats 10
python3 scripts/summarize_cpu_kv_live.py /absolute/run/matrix.csv
python3 scripts/validate_cpu_kv_results.py /absolute/run
```

Native `max_position_embeddings=40960`, no RoPE scaling: 64K/128K are recorded as
unsupported. A second GPU increases memory but does not extend the model's native
positional limit. No unsupported override is used for reported measurements.
