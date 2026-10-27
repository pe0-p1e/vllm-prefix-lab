# vLLM Prefix Cache Lab

Native vLLM Automatic Prefix Caching experiments on the shared lab 4090-02 GPU node. Not a Mooncake reproduction.

All writable data and tools stay under `/state/partition/fyzhou`. No sudo or global shell changes. See branch `exp/vllm-apc-baseline` for scripts.

## Clone

```bash
ssh 4090-02
mkdir -p /state/partition/fyzhou
cd /state/partition/fyzhou
git clone https://github.com/pe0-p1e/vllm-prefix-lab.git
cd vllm-prefix-lab
git switch exp/vllm-apc-baseline
source scripts/env.sh
bash scripts/doctor.sh
bash scripts/install_uv.sh
bash scripts/install_vllm.sh
bash scripts/run_matrix.sh --quick
```

The repo contains a reproducible APC on/off comparison using vLLM's built-in `prefix_repetition` dataset. No model weights or credentials should be committed.

## CPU–GPU tiered KV baseline

Once the GPU-only APC benchmark is working, the next task is to distinguish HBM prefix reuse from CPU DRAM KV recovery. This uses **vLLM native CPU KV offloading**, not a custom cache. With a free GPU and enough host RAM:

```bash
source scripts/env.sh
source scripts/proxy_env.sh off
CUDA_VISIBLE_DEVICES=1 bash scripts/run_tiered.sh
cat "$(cat results/TIERED_LATEST)/REPORT.md"
```

See [CPU–GPU experiment design and resource limits](docs/tiered-kv.md). The separate `TIERED_LATEST` pointer preserves the original APC results.
