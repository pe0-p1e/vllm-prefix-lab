#!/usr/bin/env bash
# Source after scripts/env.sh. Respect an explicit CUDA_VISIBLE_DEVICES;
# otherwise select an almost-idle GPU for this single-GPU benchmark.
# Fail rather than disrupt other users' jobs on a shared server.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Source this file from an experiment script' >&2
  exit 2
fi
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi unavailable' >&2; return 2; }
GPU_STATUS="$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)" || return 2
export GPU_STATUS
GPU_CHOICE="$(python3 - <<'PY'
import os,sys
devices=[]
for line in os.environ['GPU_STATUS'].splitlines():
    try:
        idx, used, total = (int(x.strip()) for x in line.split(',')[:3])
        devices.append((idx, used, total))
    except (ValueError,IndexError):
        continue
if not devices:
    sys.exit('Cannot parse nvidia-smi GPU memory usage')
requested=os.environ.get('CUDA_VISIBLE_DEVICES','').strip()
if requested:
    if not requested.isdigit():
        sys.exit('For the single-GPU lab set CUDA_VISIBLE_DEVICES to one physical GPU index, e.g. 1')
    options=[d for d in devices if d[0]==int(requested)]
    if not options:
        sys.exit('Requested GPU index is not reported by nvidia-smi')
else:
    options=sorted(devices,key=lambda d:d[1]/d[2])
# Default: require an almost-idle GPU. Shared mode needs explicit opt-in,
# an explicit physical GPU index, and enough FREE VRAM for vLLM's fraction
# of TOTAL device memory plus a configurable reserve.
shared = os.environ.get("LAB_ALLOW_SHARED_GPU", "0") == "1"
if shared:
    if not requested:
        sys.exit("Shared GPU mode requires an explicit CUDA_VISIBLE_DEVICES=<physical index>")
    try:
        fraction = float(os.environ.get("LAB_REQUIRED_GPU_FRACTION", "0.45"))
        reserve_mib = int(os.environ.get("LAB_GPU_RESERVE_MIB", "2048"))
    except ValueError:
        sys.exit("Invalid LAB_REQUIRED_GPU_FRACTION or LAB_GPU_RESERVE_MIB")
    if not (0.05 <= fraction <= 0.95 and reserve_mib >= 1024):
        sys.exit("Invalid memory fraction or reserve (< 1024 MiB)")
    usable = [d for d in options if d[2]-d[1] >= d[2]*fraction + reserve_mib]
    if not usable:
        for idx, used, total in options:
            needed = total*fraction + reserve_mib
            print(f"GPU {idx}: free={total-used} MiB, need >= {needed:.0f} MiB "
                  f"(fraction={fraction}, reserve={reserve_mib} MiB)",file=sys.stderr)
        sys.exit("Not enough free GPU memory for shared test. Aborting without affecting other jobs.")
else:
    usable = [d for d in options if d[1] <= max(512, int(d[2]*0.10))]
    if not usable:
        sys.exit("No sufficiently idle GPU. For a limited shared functional test, "
                 "set LAB_ALLOW_SHARED_GPU=1 explicitly and keep sufficient VRAM reserve.")
chosen = usable[0]
print(chosen[0])
print(f"Selected physical GPU {chosen[0]}: {chosen[1]}/{chosen[2]} MiB used; "
      f"{'shared (latency data confounded)' if shared else 'idle'}",file=sys.stderr)
PY
)" || return 2
export CUDA_VISIBLE_DEVICES="$GPU_CHOICE"
unset GPU_STATUS GPU_CHOICE
