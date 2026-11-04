#!/usr/bin/env python3
"""Run a real vLLM request with forced, synthetic CPU-resident KV hits.

Requires vLLM 0.32.0 and src.synthetic_cpu_connector. Does NOT perform
content/hash lookup, automatic prefix caching or KV offloading.
"""
import argparse
import csv
import json
import os
import pathlib
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

from src.prefix_lab import make_ids


METRICS = {
    "prefill_ms": "vllm:request_prefill_time_seconds_sum",
    "prefill_computed_tokens": "vllm:request_prefill_kv_computed_tokens_sum",
    "host_cached_tokens": "vllm:prompt_tokens_cached_by_source_total",
    "queue_ms": "vllm:request_queue_time_seconds_sum",
}


def metric_values(text):
    result = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        if " " not in line:
            continue
        name, value = line.rsplit(None, 1)
        if name.startswith(METRICS["host_cached_tokens"]):
            if 'source="host"' not in name:
                continue
            name = METRICS["host_cached_tokens"]
        else:
            name = name.split("{", 1)[0]
        if name in METRICS.values():
            result[name] = result.get(name, 0) + float(value)
    return result


def metrics(opener, base):
    with opener.open(base + "/metrics", timeout=15) as resp:
        return metric_values(resp.read().decode())


def sample_request(opener, base, token_ids):
    payload = json.dumps(dict(model="prefix-lab", prompt=token_ids,
                              max_tokens=1, temperature=0,
                              ignore_eos=True, stream=True)).encode()
    req = urllib.request.Request(
        base + "/v1/completions", data=payload,
        method="POST", headers={"Content-Type": "application/json"})
    t0 = time.perf_counter_ns()
    first = None
    finished = False
    with opener.open(req, timeout=900) as resp:
        for line in resp:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                finished = True
                break
            obj = json.loads(data)
            if "error" in obj:
                raise RuntimeError(obj["error"])
            choices = obj.get("choices") or []
            if first is None and choices and (
                    "text" in choices[0] or "token_ids" in choices[0]):
                first = time.perf_counter_ns()
    if first is None or not finished:
        raise RuntimeError("Stream did not emit token and finish normally")
    return (first - t0) / 1e6, (time.perf_counter_ns() - t0) / 1e6


def wait_ready(opener, base, process, timeout):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if process.poll() is not None:
            raise RuntimeError("vLLM process exited before readiness")
        try:
            with opener.open(base + "/v1/models", timeout=2):
                return
        except Exception:
            time.sleep(1)
    raise TimeoutError("Timed out waiting for vLLM readiness")


def shutdown(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def report(rows, output):
    fields = ["length", "requested_ratio", "mode", "repeat", "requested_hit_tokens",
              "actual_host_hit_tokens", "injected_hit_tokens", "actual_prefill_kv_tokens",
              "prefill_service_ms", "ttft_ms", "e2e_ms", "h2d_cuda_ms",
              "scatter_cuda_ms", "worker_load_wall_ms", "worker_prepare_ms",
              "kv_bytes", "h2d_gib_per_s", "peak_gpu_allocated_bytes",
              "verified", "status", "error"]
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lengths", nargs="+", type=int,
                   default=[2048,4096,8192,16384,32768,65536,131072])
    p.add_argument("--ratios", nargs="+", type=float, default=[0,.25,.5,.75,1])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--allow-extended-context", action="store_true",
                   help="Experimental: permit context beyond model config; no correctness guarantee")
    p.add_argument("--model-path", default="/state/partition/model/Qwen3-0.6B")
    p.add_argument("--venv", default="/state/partition/fyzhou/venvs/vllm")
    p.add_argument("--root", default="/state/partition/fyzhou")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--gpu-util", type=float, default=.65)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--port", type=int, default=18083)
    p.add_argument("--ready-timeout", type=int, default=300)
    p.add_argument("--allow-shared", action="store_true")
    p.add_argument("--pinned", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", type=pathlib.Path, default=None)
    a = p.parse_args()
    if a.repeats < 1 or a.warmups < 0 or any(x < 128 for x in a.lengths):
        p.error("Invalid experiment lengths/repetitions")
    if any(not 0 <= r <= 1 for r in a.ratios) or not 0.1 <= a.gpu_util <= .9:
        p.error("Invalid ratio or gpu-util")
    if a.block_size < 1 or a.gpu < 0:
        p.error("Invalid block size/GPU")
    root = pathlib.Path(a.root).resolve()
    base = root / "results" / "cpu-kv-live"
    out = a.out or base / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out.resolve()
    if not out.is_relative_to(root):
        p.error("Output directory must be under --root")
    out.mkdir(parents=True, exist_ok=False)
    (base / "LATEST").write_text(str(out)+"\n")
    repo = pathlib.Path(__file__).resolve().parent.parent
    py = pathlib.Path(a.venv) / "bin/python"
    vllm = pathlib.Path(a.venv) / "bin/vllm"
    if not vllm.exists():
        p.error(f"Missing vLLM: {vllm}")
    subprocess.run([str(vllm), "--version"], check=True)
    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    cfg = AutoConfig.from_pretrained(a.model_path, local_files_only=True)
    max_pos = int(getattr(cfg, "max_position_embeddings", 0) or 0)
    gpu = subprocess.check_output([
        "nvidia-smi", "-i", str(a.gpu),
        "--query-gpu=memory.used,memory.total",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    used, total = [int(x.strip()) for x in gpu.split(",")[:2]]
    if used > 512 and not a.allow_shared:
        p.error("GPU busy; --allow-shared required for exploratory results")
    if total-used < a.gpu_util*total+1024:
        p.error("Insufficient free VRAM for gpu-util fraction + 1GiB reserve")
    setup = dict(model=a.model_path, vllm_version=subprocess.check_output(
        [str(vllm), "--version"],text=True).strip(), gpu=a.gpu,
        gpu_mem_used_mib=used, gpu_mem_total_mib=total,
        contended=used > 512, model_max_position_embeddings=max_pos,
        args=vars(a) | {"out": str(out)})
    (out / "setup.json").write_text(json.dumps(setup, indent=2, default=str))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),
               VLLM_USE_FLASHINFER_SAMPLER="0",
               HF_HOME=str(root/"cache/huggingface"),
               XDG_CACHE_HOME=str(root/"cache/xdg"),
               VLLM_CACHE_ROOT=str(root/"cache/vllm"),
               FLASHINFER_WORKSPACE_BASE=str(root/"cache/flashinfer-workspace"),
               TRITON_CACHE_DIR=str(root/"cache/triton"),
               CUDA_CACHE_PATH=str(root/"cache/cuda"),
               TMPDIR=str(root/"tmp"),
               PYTHONPATH=str(repo)+os.pathsep+env.get("PYTHONPATH",""))
    if a.allow_extended_context:
        # For synthetic benchmark only: data semantics are deliberately invalid.
        # vLLM validates max_model_len by default; this override allows the
        # explicit experiment to attempt longer positions.
        env["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    for key in ["HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","http_proxy","https_proxy",
                "all_proxy"]:
        env.pop(key, None)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base_url = f"http://127.0.0.1:{a.port}"
    result = []
    for length in a.lengths:
        if max_pos and length > max_pos and not a.allow_extended_context:
            result.append(dict(length=length, status="unsupported_model_context",
                               error=f"max_position_embeddings={max_pos}"))
            continue
        for ratio in a.ratios:
            hit = min((int(length*ratio)//a.block_size)*a.block_size,
                      ((length-1)//a.block_size)*a.block_size)
            mode = "cpu" if hit else "disable"
            case = out / f"L{length}_R{ratio:g}"
            case.mkdir()
            try:
                with socket.create_connection(("127.0.0.1", a.port), timeout=.5):
                    raise RuntimeError("Port already occupied by another process")
            except (ConnectionRefusedError, TimeoutError, OSError):
                pass
            extra = dict(forced_hit_tokens=hit, pinned=a.pinned,
                         metrics_dir=str(case))
            transfer = dict(kv_connector="SyntheticCPUConnector",
                            kv_connector_module_path="src.synthetic_cpu_connector",
                            kv_role="kv_both", kv_connector_extra_config=extra)
            cmd = [str(vllm), "serve", a.model_path,
                   "--served-model-name","prefix-lab",
                   "--host","127.0.0.1","--port",str(a.port),
                   "--max-model-len",str(length+1),
                   "--gpu-memory-utilization",str(a.gpu_util),
                   "--block-size",str(a.block_size),
                   "--max-num-batched-tokens","8192",
                   "--disable-hybrid-kv-cache-manager",
                   "--no-enable-prefix-caching"]
            if hit:
                cmd += ["--kv-transfer-config",json.dumps(transfer)]
            (case/"command.json").write_text(json.dumps(cmd,indent=2))
            process = None
            log = (case/"server.log").open("w")
            try:
                process = subprocess.Popen(cmd, cwd=repo, env=env, stdout=log,
                                           stderr=subprocess.STDOUT,
                                           start_new_session=True)
                wait_ready(opener, base_url, process, a.ready_timeout)
                for repeat in range(max(a.warmups, 1)+a.repeats):
                    source = make_ids(tok, length, 900000+repeat)
                    # Exact list length; raw token IDs passed to OpenAI API.
                    assert len(source) == length
                    prior = metrics(opener,base_url)
                    files_before = set(case.glob("kvload-*.json"))
                    ttft,e2e = sample_request(opener,base_url,source)
                    after = metrics(opener,base_url)
                    delta = {k:after.get(k,0)-prior.get(k,0)
                             for k in METRICS.values()
                             if k in prior and k in after}
                    new_files = set(case.glob("kvload-*.json"))-files_before
                    load = None
                    if hit:
                        if len(new_files)!=1:
                            raise RuntimeError(
                                f"Expected one CPU KV injection, got {len(new_files)}")
                        load = json.loads(next(iter(new_files)).read_text())
                    host = delta.get(METRICS["host_cached_tokens"],None)
                    prefill = delta.get(METRICS["prefill_ms"],None)
                    computed = delta.get(METRICS["prefill_computed_tokens"],None)
                    injected = load["injected_hit_tokens"] if load else 0
                    valid = (injected == hit and
                             computed is not None and
                             abs(computed-(length-hit)) <= a.block_size and
                             (hit == 0 or
                              (host is not None and abs(host-hit)<=a.block_size)) and
                             prefill is not None)
                    row = dict(length=length,requested_ratio=ratio,
                               mode=mode,repeat=repeat-max(a.warmups, 1),
                               requested_hit_tokens=hit,
                               actual_host_hit_tokens=host,
                               injected_hit_tokens=injected,
                               actual_prefill_kv_tokens=computed,
                               prefill_service_ms=prefill*1000 if prefill is not None else None,
                               ttft_ms=ttft,e2e_ms=e2e,
                               h2d_cuda_ms=load["sum_layer_h2d_cuda_ms"] if load else 0,
                               scatter_cuda_ms=load["sum_layer_scatter_cuda_ms"] if load else 0,
                               worker_load_wall_ms=load["total_worker_load_wall_ms"] if load else 0,
                               worker_prepare_ms=load["sum_layer_prepare_ms"] if load else 0,
                               kv_bytes=load["bytes"] if load else 0,
                               h2d_gib_per_s=(load["bytes"]/2**30)/
                               (load["sum_layer_h2d_cuda_ms"]/1000)
                               if load and load["sum_layer_h2d_cuda_ms"] else None,
                               peak_gpu_allocated_bytes=load["gpu_peak_allocated_bytes"]
                               if load else None,
                               verified=valid,status="measured" if valid else "unverified")
                    if repeat >= max(a.warmups, 1):
                        result.append(row)
                        report(result,out/"matrix.csv")
                    print(json.dumps(row),flush=True)
                    if not valid:
                        raise RuntimeError("KV injection/host metric mismatch")
            except Exception as error:
                result.append(dict(length=length,requested_ratio=ratio,
                                   mode=mode,status="failed",error=str(error)))
                report(result,out/"matrix.csv")
                print("FAILED",case,error,flush=True)
                # Preserve logs and abort: never silently drop failed cells.
                raise
            finally:
                shutdown(process)
                log.close()
    report(result,out/"matrix.csv")
    print("Results:",out)


if __name__=="__main__":
    main()
