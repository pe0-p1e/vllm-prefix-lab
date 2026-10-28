# vLLM Prefix Cache Lab

Controlled single-node vLLM automatic prefix caching (APC) latency experiments.
**Active branch:** `exp/prefix-hit-matrix`. Old CPU KV/offload, mihomo, installation
and overlap experiments remain in `exp/vllm-apc-baseline`, not in this tree.

## Layout

- `src/prefix_lab.py`: exact-length token ID construction, HTTP client, metrics verification
- `scripts/run.sh`: isolated vLLM server per (length, ratio, mode) cell
- `scripts/summary.py`: aggregate validated JSON to CSV
- `tests/test_prefix_lab.py`: CPU-only deterministic unit tests
- `docs/EXPERIMENT.md`: protocol, limitations, commands, audit guide

## Quick start (4090-02)

```bash
ssh 4090-02
cd /state/partition/fyzhou/vllm-prefix-lab
git fetch origin
git switch exp/prefix-hit-matrix
git pull --ff-only
python3 -m unittest discover -s tests -v
# Set vLLM executable and model path only if different from current lab:
LAB_LENGTHS="2048 4096" LAB_RATIOS="0 0.5 0.75" LAB_REPEATS=3 bash scripts/run.sh
cat "$(cat /state/partition/fyzhou/results/prefix-lab/LATEST)/matrix.csv"
```

Read `docs/EXPERIMENT.md` before running. No script installs dependencies,
uses sudo, edits shell startup files, kills unrelated processes or downloads
models. Model weights and runtime results do not belong in Git.
