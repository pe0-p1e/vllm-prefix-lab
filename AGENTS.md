# Experiment constraints
- Focus on built-in vLLM Automatic Prefix Caching and official benchmark CLI, not Mooncake reproduction.
- Work only within /state/partition/fyzhou on the 4090-02 server, with optional LAB_ROOT override.
- Never use sudo or install to /usr/local or edit ~/.bashrc, ~/.config, ~/.ssh, /etc, or global Git settings.
- Never store passwords, GitHub tokens, proxy subscriptions, Codex credentials, or model checkpoints in Git.
- Never stop unrelated processes on a shared GPU server.
- Record model revision, vLLM version, GPU status, APC flags, raw benchmark outputs and metrics.
- Do not invent experimental results. Correctness and resource isolation take precedence over throughput.
