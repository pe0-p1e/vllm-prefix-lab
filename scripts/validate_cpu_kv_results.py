#!/usr/bin/env python3
"""Independently audit raw injection records and publish descriptive statistics."""
import csv
import json
import pathlib
import statistics
import sys
import itertools
from collections import defaultdict
from summarize_cpu_kv_live import quantile


def main():
    root = pathlib.Path(sys.argv[1])
    config = json.loads((root/"configuration.json").read_text()) if (root/"configuration.json").exists() else {}
    rows = list(csv.DictReader((root/'matrix.csv').open()))
    loads = [json.loads(p.read_text()) for p in root.rglob('kvload-*.json')]
    # Files are numbered in worker execution order; payload changes start a new buffer group.
    ordered = [json.loads(p.read_text()) for p in sorted(root.rglob('kvload-*.json'))]
    host_groups = []
    for payload, group in itertools.groupby(ordered, key=lambda r: r['bytes']):
        group = list(group)
        pointers = [{x['layer']: x['host_ptr'] for x in r['layer_metrics']} for r in group]
        stable = all(p == pointers[0] for p in pointers)
        host_groups.append(dict(bytes=payload, requests=len(group), host_buffers_stable=stable))
        assert stable
    checks = []
    for r in loads:
        h = r['injected_hit_tokens']
        expected = h * 28 * 2 * 8 * 128 * 2
        layers = r['layer_metrics']
        checks.append(dict(request_id=r['request_id'], tokens=h,
                           expected_bytes=expected, bytes=r['bytes'],
                           passed=r['bytes']==expected and len(layers)==28
                           and len({x['layer'] for x in layers})==28
                           and all(x['destination_ptr'] > 0 for x in layers)))
    grouped = defaultdict(list)
    for r in rows:
        if r['status']=='measured' and r['verified']=='True':
            l, h, m = (int(float(r[k])) for k in
                       ('length','injected_hit_tokens','actual_prefill_kv_tokens'))
            assert h+m==l
            assert float(r['preemptions'])==0
            assert int(float(r['kv_bytes']))==h*28*2*8*128*2
            grouped[(l,r['requested_ratio'])].append(r)
    fields = ['length','requested_ratio','actual_ratio','metric','n','p50','p95','p99','stdev','min','max']
    with (root/'summary.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for (l,ratio),rs in sorted(grouped.items()):
            for metric in ('h2d_cuda_ms','scatter_cuda_ms','worker_load_wall_ms','prefill_service_ms','ttft_ms','e2e_ms'):
                vs = [float(r[metric]) for r in rs]
                w.writerow(dict(length=l,requested_ratio=ratio,
                    actual_ratio=float(rs[0]['injected_hit_tokens'])/l,
                    metric=metric,n=len(vs),p50=statistics.median(vs),
                    p95=quantile(vs,.95),p99=quantile(vs,.99),
                    stdev=statistics.stdev(vs) if len(vs)>1 else '',min=min(vs),max=max(vs)))
    validation = dict(host_buffer_checks=host_groups, raw_load_checks=checks, all_loads_pass=all(c['passed'] for c in checks),
                      measured_rows=sum(map(len,grouped.values())),
                      unsupported=[r for r in rows if r['status']!='measured'],
                      readback_layers=sum(x.get('readback_verified',False) for r in loads for x in r['layer_metrics']))
    (root/'validation.json').write_text(json.dumps(validation,indent=2))
    lines = ['# Synthetic CPU KV prefill experiment', '',
        'Exploratory shared-GPU measurements; contention prevents isolated performance claims.',
        'Qwen3-0.6B, vLLM 0.32.0, BF16, 28 layers, 8 KV heads, head dimension 128; 114688 bytes/token.',
        'Explicit token lists; APC disabled; one output token, temperature zero; eager execution; chunked prefill 8192.',
        f"Warmups={config.get('args',{}).get('warmups','unknown')}; measured repetitions={config.get('args',{}).get('repeats','unknown')} per cell. Separate cache-disabled engine. CPU buffers reused within each cell.",
        'H is block aligned; the final block is recomputed at requested 100%. Synthetic KV has no semantic validity.',
        'H2D and scatter are separate CUDA-event intervals. Per-layer synchronization intentionally serializes loading.',
        'Prefill service time includes loading and scheduling overhead and is not pure GPU compute. TTFT/completion use client wall clock.',
        'No reliable GPU model-kernel or Attention-only timing was collected. CPU RSS and PCIe throughput unavailable.',
        'Hardware CSV records whole-device utilization, including other users. Small-sample P99 is unstable.',
        'Native model context is 40960 without RoPE scaling. 64K and 128K are unsupported; no override or shorter substitution was used.',
        'Measured requests had no errors or preemptions. EngineDeadError after SIGTERM in the CPU server log belongs to shutdown after the last successful request.',
        f'Verified maximum input length: {max(l for l, ratio in grouped)} tokens.',
        'Next experiments: reserve an idle GPU, randomize cell order, compare minimally synchronized loading, and use a model with native 128K context for longer inputs.',
        '', '| Tokens | Requested ratio | Actual ratio | n | H2D ms | Scatter ms | Prefill service ms | TTFT ms | Total ms |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for (l,ratio),rs in sorted(grouped.items()):
        med = lambda k: statistics.median(float(r[k]) for r in rs)
        lines.append(f'| {l} | {ratio} | {float(rs[0]["injected_hit_tokens"])/l:.6f} | {len(rs)} | {med("h2d_cuda_ms"):.3f} | {med("scatter_cuda_ms"):.3f} | {med("prefill_service_ms"):.3f} | {med("ttft_ms"):.3f} | {med("e2e_ms"):.3f} |')
    (root/'experiment_report.md').write_text('\n'.join(lines)+'\n')
    assert validation['all_loads_pass']
    print(f"Validated {validation['measured_rows']} measurements, {len(checks)} loads, {validation['readback_layers']} readback layers")


if __name__=='__main__':
    main()
