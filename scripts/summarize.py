#!/usr/bin/env python3
"""Summarize real native vLLM result JSON; do not fabricate measurements."""
import csv
import json
import pathlib
import re
import sys

def parse_counter(text, name):
    vals = []
    # Prometheus exports Python Counter metrics with a _total suffix. The\n    # vLLM metric declaration omits _total, so accept both spellings but\n    # never count _created or unrelated metrics.\n    pattern = re.compile(r'^' + re.escape(name) + r'(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)(?:\s|$)')
    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            vals.append(float(m.group(1)))
    return sum(vals) if vals else None

def main():
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python3 scripts/summarize.py results/LATEST')
    root = pathlib.Path(sys.argv[1])
    if root.is_file():
        root = pathlib.Path(root.read_text().strip())
    root = root.expanduser().resolve()
    found = sorted(root.glob('*/result.json'))
    if not found:
        raise SystemExit('No completed benchmark result JSON files')
    fields = ['completed','failed','request_throughput','output_throughput','total_token_throughput','mean_ttft_ms','median_ttft_ms','p95_ttft_ms','p99_ttft_ms','mean_tpot_ms','median_tpot_ms','p95_tpot_ms','p99_tpot_ms']
    rows = []
    for f in found:
        cfg = {}
        for line in (f.parent / 'config.txt').read_text().splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                cfg[k] = v
        raw = json.loads(f.read_text())
        row = {'case': f.parent.name, **cfg, **{k:raw.get(k) for k in fields}}
        before = (f.parent / 'metrics_before.txt').read_text() if (f.parent / 'metrics_before.txt').exists() else ''
        after = (f.parent / 'metrics_after.txt').read_text() if (f.parent / 'metrics_after.txt').exists() else ''
        for name in ('vllm:prefix_cache_hits','vllm:prefix_cache_queries'):
            x, y = parse_counter(before, name), parse_counter(after, name)
            row[name.split(':')[-1]+'_delta'] = (y-x) if x is not None and y is not None else None
        hit = row['prefix_cache_hits_delta']
        query = row['prefix_cache_queries_delta']
        row['prefix_cache_hit_ratio'] = hit/query if hit is not None and query is not None and query > 0 else None
        rows.append(row)
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with (root / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    def fmt(v):
        if v is None: return 'N/A'
        if isinstance(v, float): return f'{v:.4g}'
        return str(v)
    md = ['# Native vLLM APC benchmark summary', '', 'Data comes from completed vLLM benchmark JSON. Missing metrics are N/A.', '',
          '| Case | Completed | TTFT median ms | TTFT p99 ms | Throughput req/s | Prefix hit ratio |',
          '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        md.append('| ' + ' | '.join(fmt(x) for x in [row['case'],row.get('completed'),row.get('median_ttft_ms'),row.get('p99_ttft_ms'),row.get('request_throughput'),('N/A' if row.get('prefix_cache_hit_ratio') is None else f"{100*row['prefix_cache_hit_ratio']:.1f}%")]) + ' |')
    md += ['', 'Prefix hit ratio is token-level delta(prefix_cache_hits_total) / delta(prefix_cache_queries_total); N/A indicates zero or missing queries.', '', 'Compare APC on/off only with the same model, prefix length, concurrency, arrival rate and GPU memory budget.']
    (root / 'REPORT.md').write_text('\n'.join(md)+'\n')
    print('Saved:',root/'summary.csv','and',root/'REPORT.md')

if __name__ == '__main__':
    main()
