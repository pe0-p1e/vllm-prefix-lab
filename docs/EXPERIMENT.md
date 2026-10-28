# Experiment protocol and audit guide

## Objective

For Qwen3-0.6B, measure streaming TTFT (prefill-dominated, **not** isolated
GPU prefill kernel time) versus input length and requested cached prefix ratio.
Use built-in vLLM APC and raw token-ID arrays via /v1/completions.

## Variables

Length: default 2K/4K/8K; extend to 16K/32K/64K/128K only if model context,
KV capacity and memory permit. 256K is not assumed supported.
Requested ratio: 0 / .25 / .5 / .75 / 1.
Control: fresh APC-disabled server per length. APC-on zero-hit control also
included. Separate server for each length/mode/ratio cell; process startup is
excluded from latency. Repeats each have a distinct document identifier.

## Measurement validity

Prime with a different suffix sharing precisely the requested initial token
IDs. Then issue test request with max_tokens=1. vLLM can only reuse full blocks
and may handle the last block differently. The source=device Prometheus
counter is sampled before/after each measured request; unverified/mismatched
cells fail rather than being silently presented as controlled cache hits.
The configured LAB_BLOCK_SIZE must match the server's KV block size.

TTFT includes HTTP transport, queue, scheduling, first decode, etc. It is
NOT an isolated kernel timer. A 100% requested hit does not imply zero
work. Cold/APC-off counters may be absent. Check server logs and metric
presence per version.

## Execution / resource policy

All writes under /state/partition/fyzhou. Reuse existing preinstalled
/state/partition/fyzhou/venvs/vllm and local model. No sudo, automated
downloads, proxy changes, VS Code installation, or host-global changes.
Specify one idle GPU explicitly via CUDA_VISIBLE_DEVICES. Only child
server PID started by script is stopped. Stop on errors, OOM, metric
mismatch, unsupported CLI flags, occupied port, non-idle GPU at start.

## Commands

```bash
cd /state/partition/fyzhou/vllm-prefix-lab
git switch exp/prefix-hit-matrix
python3 -m unittest discover -s tests -v
export CUDA_VISIBLE_DEVICES=0  # replace with inspected, idle physical GPU
LAB_LENGTHS="2048 4096" LAB_RATIOS="0 0.5 0.75" \
LAB_REPEATS=3 bash scripts/run.sh
cat "$(cat /state/partition/fyzhou/results/prefix-lab/LATEST)/matrix.csv"
```

Longer matrix after checking model configuration and GPU capacity:

```bash
LAB_MAX_MODEL_LEN=131072 LAB_LENGTHS="2048 4096 8192 16384 32768 65536 131072" \
LAB_RATIOS="0 0.25 0.5 0.75 1" LAB_REPEATS=5 bash scripts/run.sh
```

## Audit checklist

Check src/prefix_lab.py: exact length, divergence and metrics verification;
scripts/run.sh: environment, GPU idle guard, child PID cleanup, fresh server;
scripts/summary.py: no invented metrics. No modified vLLM internals. This
branch deliberately omits obsolete offload/overlap/proxy/install scripts.
Retained old source is at exp/vllm-apc-baseline.
