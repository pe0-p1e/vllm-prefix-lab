# 4090-02 Runbook

Use vLLM's own serve and bench serve commands, not a Mooncake replacement or custom KV cache code. All writes stay in /state/partition/fyzhou; do not use sudo, edit global shell files or stop unrelated GPU processes.

## Clone and setup

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

Model location is detected from /state/partition/model/Qwen3-0.6B or Qwen3.5-2B. If unavailable, set MODEL_OVERRIDE to a verified existing absolute path.

## Native vLLM APC experiments

    bash scripts/run_matrix.sh --list
    bash scripts/run_matrix.sh --quick
    cat "$(cat results/LATEST)/REPORT.md"

Full sweep: APC off/on; prefix length 1024/4096; concurrency 1/4.

    bash scripts/run_matrix.sh --full

Each case launches its own server, records vLLM version, model path, GPU status, result.json, Prometheus metrics and logs. The script stops only the server that it launched. Results and credentials are gitignored.

## Codex

    bash scripts/install_codex.sh
    codex login --device-auth
    codex

Codex binary: /state/partition/fyzhou/tools/bin. CODEX_HOME: /state/partition/fyzhou/private/codex.

## Optional Mihomo (OFF by default)

    bash scripts/install_mihomo.sh
    bash scripts/mihomo.sh init

Replace the DIRECT-only example at /state/partition/fyzhou/private/mihomo/config.yaml with a trusted config that binds only to localhost and listens on port 17890.

    bash scripts/mihomo.sh start
    source scripts/proxy_env.sh on
    # network tasks
    source scripts/proxy_env.sh off
    bash scripts/mihomo.sh stop

Nothing modifies system-wide proxy settings or creates an auto-start service.

## Scope and limitations

The vLLM CLI flags change across versions. If the benchmark fails, inspect case-specific bench.log and server.log and adjust only version-incompatible arguments. Use the official model files and respect the shared GPU's available memory. No experiments have been run on your hardware from this chat.
