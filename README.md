# CPU KV Cache — Controlled Prefill Benchmark

This is the **CPU KV only** experiment branch for Qwen3-0.6B on vLLM
**0.32.0**. The goal is to measure forced CPU KV prefix length versus
remaining Prefill and H2D transfer time. No content-matching APC and no
automatic offloading. Synthetic K/V values are **not** semantically valid.

**Integration code is committed but not GPU-validated on 4090-02.**
Do not report results until the validation flags pass.

## Code layout

```text
src/
  synthetic_cpu_connector.py    # vLLM V1 forced host KV injection
  token_ids.py                  # exact token-count input lists
scripts/
  run_cpu_kv_matrix.py          # launch, sample, verify, persist
  summarize_cpu_kv_live.py      # report median, p95, p99 and bandwidth
  bench_cpu_kv_transfer.py      # standalone contiguous H2D control
  summary_cpu_kv_transfer.py
tests/
  test_cpu_kv_live.py
  test_transfer_math.py
docs/
  CPU_KV_EXPERIMENT.md          # detailed runbook and acceptance criteria
```

## Codex on 4090-02 (only this process uses Clash)

The launcher is part of this branch and does **not** edit your shell profile:

```bash
cd /state/partition/fyzhou/vllm-prefix-lab
bash scripts/codex.sh                   # new Codex session
bash scripts/codex.sh resume            # choose existing session
bash scripts/codex.sh resume --last     # resume last session
CLASH_PORT=7890 bash scripts/codex.sh resume  # explicit local proxy port
```

It uses `/state/partition/fyzhou/tools/bin/codex`, auto-detects a
listening Mihomo/Clash port (17890, 7890, 7897), preserves existing
`CODEX_HOME` or `~/.codex` login state and passes all arguments unchanged.
It **does not** install Codex or fix a missing
`codex-code-mode-host` executable; if that binary is missing, repair the
partition-local `install.sh` installation before starting a new session.

## Test and run

```bash
cd /state/partition/fyzhou/vllm-prefix-lab
git fetch origin
git switch exp/prefix-hit-matrix
git pull --ff-only
/state/partition/fyzhou/venvs/vllm/bin/python -m compileall -q src scripts tests
/state/partition/fyzhou/venvs/vllm/bin/python -m unittest discover -s tests -v

# Inspect nvidia-smi and select an authorized, sufficiently free GPU.
# Small integration smoke test first:
/state/partition/fyzhou/venvs/vllm/bin/python scripts/run_cpu_kv_matrix.py \
  --gpu 0 --lengths 2048 --ratios 0 .5 1 --repeats 2

# Full experiment (128K may be skipped by model/VRAM constraints):
/state/partition/fyzhou/venvs/vllm/bin/python scripts/run_cpu_kv_matrix.py \
  --gpu 0 \
  --lengths 2048 4096 8192 16384 32768 65536 131072 \
  --ratios 0 .25 .5 .75 1 --repeats 10 --warmups 3

R=$(cat /state/partition/fyzhou/results/cpu-kv-live/LATEST)
python3 scripts/summarize_cpu_kv_live.py "$R/matrix.csv" --out "$R/table.csv"
```

See [the full protocol](docs/CPU_KV_EXPERIMENT.md) before treating any data
as experimental evidence. All writes stay under
`/state/partition/fyzhou`, with no sudo, no global CUDA edits and no
unrelated GPU process termination. The original APC experiments remain
available in `exp/vllm-apc-baseline`.
