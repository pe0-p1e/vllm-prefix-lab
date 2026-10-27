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
# Conservative shared-server guard: another workload likely holds the GPU
usable=[d for d in options if d[1] <= max(512,int(d[2]*0.10))]
if not usable:
    sys.exit('No sufficiently idle GPU. Inspect nvidia-smi; do not stop other users\' jobs.')
chosen=usable[0]
print(chosen[0])
print(f'Selected physical GPU {chosen[0]}: {chosen[1]}/{chosen[2]} MiB used',file=sys.stderr)
PY
)" || return 2
export CUDA_VISIBLE_DEVICES="$GPU_CHOICE"
unset GPU_STATUS GPU_CHOICE
