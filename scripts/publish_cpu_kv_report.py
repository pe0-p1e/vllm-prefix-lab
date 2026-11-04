#!/usr/bin/env python3
"""Publish a compact human-readable Chinese Markdown snapshot of a validated run."""
import argparse
import csv
import hashlib
import json
import pathlib
import shutil
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=pathlib.Path)
    p.add_argument('--out',required=True,type=pathlib.Path)
    a=p.parse_args()
    src,dst=a.run,a.out
    cfg=json.loads((src/'configuration.json').read_text())
    validation=json.loads((src/'validation.json').read_text())
    assert validation['all_loads_pass']
    table=list(csv.DictReader((src/'table.csv').open()))
    valid=[r for r in table if r['status']=='verified']
    raw=list(csv.DictReader((src/'matrix.csv').open()))
    for r in valid:
        samples=[x for x in raw if x['status']=='measured'
                 and int(x['length'])==int(r['input_tokens'])
                 and float(x['requested_ratio'])==float(r['requested_cpu_hit_ratio'])]
        assert len(samples)==int(r['repeats'])==cfg['args']['repeats']
        assert abs(statistics.median(float(x['ttft_ms']) for x in samples)-float(r['ttft_median_ms']))<1e-8
        for x in samples:
            assert float(x['injected_hit_tokens'])+float(x['actual_prefill_kv_tokens'])==int(x['length'])
            assert float(x['kv_bytes'])==float(x['injected_hit_tokens'])*114688
    dst.mkdir(parents=True,exist_ok=True)
    for name in ('table.csv','summary.csv','configuration.json','cell_order.json'):
        if (src/name).exists():shutil.copyfile(src/name,dst/name)
    audit={k:validation[k] for k in ('all_loads_pass','measured_rows','readback_layers','host_buffer_checks')}
    audit['raw_load_checks_count']=len(validation['raw_load_checks'])
    audit['unsupported_configurations']=[{'length':r['length'],'ratio':r['requested_ratio'],'reason':r['error']} for r in validation['unsupported']]
    (dst/'validation-summary.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    max_length=max(int(r['input_tokens']) for r in valid)
    largest=[r for r in valid if int(r['input_tokens'])==max_length]
    full=next(r for r in largest if float(r['requested_cpu_hit_ratio'])==1)
    base=next(r for r in largest if float(r['requested_cpu_hit_ratio'])==0)
    lines=['# CPU KV Cache 细粒度 Prefill 实验结果','',f'运行编号：`{src.name}`；模型 Qwen3-0.6B；实验日期以运行配置和硬件采样为准。','',
        '## 结果摘要','',
        f"本轮完成 {len(valid)} 个配置，每格 {cfg['args']['warmups']} 次预热、{cfg['args']['repeats']} 次测量，共 {validation['measured_rows']} 次有效测量。最大验证输入为 {max_length:,} token。[1][2]",'',
        f"在 {max_length:,} token 输入下，近全命中 TTFT 中位数为 {float(full['ttft_median_ms']):.2f} ms，独立禁用缓存基线为 {float(base['ttft_median_ms']):.2f} ms，观测加速比 {float(full['ttft_speedup_vs_no_cache']):.2f}×。实际注入 {int(float(full['actual_cpu_hit_tokens'])):,} token，重算 {int(float(full['recomputed_prefill_tokens']))} token。[1]",'',
        '**共享 GPU 负载、同步开销和基线与 CPU 模式分开执行会影响比较；这些是描述性测量，不代表独占 GPU 的因果性能收益。** 本轮随机化了各模式内部配置顺序，但没有消除跨模式时间漂移。','',
        '## 曲线与拟合','',
        '见 [完整图表、拟合系数和交叉验证](figures/README.md)。图中阴影/误差棒为观测 P10–P90，不是置信区间。','',
        '![命中率与延迟](figures/latency_by_hit.png)','',
        '![观测加速比](figures/ttft_speedup.png)','',
        '![传输拟合](figures/transport_fit.png)','','![H2D 带宽](figures/h2d_bandwidth.png)','',
        '## 全部配置结果','',
        '来源：[1]。时间为 ms，中位数；每格 n 相同。0% 为独立 Cache Disabled 引擎。近100% 实际仍重算最后一块。各指标分别取中位数，不应强求列间加法相等。','',
        '| 输入 token | 请求比例 | 实际比例 | 注入 token | 重算 token | H2D ms | 页块写入 ms | Prefill 服务 ms | TTFT ms | 总延迟 ms | 加速比 |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    def label(r):
        q=float(r['requested_cpu_hit_ratio'])
        return '近100%' if q==1 else f'{100*q:g}%'
    for r in valid:
        f=lambda k:float(r[k])
        lines.append(f"| {r['input_tokens']} | {label(r)} | {100*f('actual_cpu_hit_ratio'):.5f}% | {f('actual_cpu_hit_tokens'):.0f} | {f('recomputed_prefill_tokens'):.0f} | {f('h2d_cuda_median_ms'):.2f} | {f('gpu_scatter_median_ms'):.2f} | {f('prefill_service_median_ms'):.2f} | {f('ttft_median_ms'):.2f} | {f('e2e_median_ms'):.2f} | {f('ttft_speedup_vs_no_cache'):.2f}× |")
    lines+=['','## CPU KV 载荷与加载','','来源：[1]。MiB=2²⁰ 字节；GiB/s=2³⁰ 字节/秒。BF16 每 token KV 为 114688 字节。','','| 输入 token | 请求比例 | KV MiB | 总加载 ms | H2D GiB/s |','|---:|---:|---:|---:|---:|']
    for r in valid:
        if r['mode']=='disable':continue
        f=lambda k:float(r[k])
        lines.append(f"| {r['input_tokens']} | {label(r)} | {f('cpu_kv_bytes')/2**20:.2f} | {f('worker_load_wall_median_ms'):.2f} | {f('h2d_effective_bandwidth_median_gib_s'):.2f} |")
    lines+=['','## TTFT 尾延迟','','来源：[1]。小样本 P99 不稳定，不能作为服务等级承诺。','','| 输入 token | 请求比例 | n | P50 ms | P95 ms | P99 ms |','|---:|---:|---:|---:|---:|---:|']
    for r in valid:
        f=lambda k:float(r[k])
        lines.append(f"| {r['input_tokens']} | {label(r)} | {r['repeats']} | {f('ttft_median_ms'):.2f} | {f('ttft_p95_ms'):.2f} | {f('ttft_p99_ms'):.2f} |")
    lines+=['','## 验证与实验口径','',
        f"来源：[2][3]。{len(validation['raw_load_checks'])} 次加载字节审计通过，{validation['readback_layers']} 层次完整回读通过。每格 CPU 缓冲保持复用，各请求重新执行传输和写入；实际注入 token 加重算 token 等于输入长度。",'',
        '- 输入是精确长度的 token ID 列表；synthetic KV 已初始化，但不要求语义匹配，不验证输出正确性。',
        '- BF16、28 层、8 KV heads、head dimension 128；单 GPU、单请求；温度 0，输出 1 token；APC 关闭。',
        '- eager、8192-token chunked Prefill；pinned CPU buffer；每层同步；两模式使用相同最大上下文与 GPU memory utilization。具体值见配置文件。',
        '- H2D、GPU 页块写入分别使用 CUDA Events；总加载是 worker 墙钟时间；Prefill 是包含加载开销的 vLLM 服务阶段时间；TTFT/完成延迟是客户端墙钟时间。',
        '- 没有可靠的纯模型 GPU compute、Attention-only 时间；不能把 Prefill 服务时间减去传输就称作纯计算。单输出 token 也不能评估持续 decode 吞吐。',
        '- `cell_order.json` 记录真实顺序；固定 seed 打乱各模式内部配置，基线模式整体先执行。共享负载及跨模式漂移仍存在。',
        '- 模型 max_position_embeddings=40960，rope_scaling=null。本轮按用户确认仅在原生范围内执行；64K/128K 未执行推理并继续记录为不支持。原生上限不是权限限制，也不等于安全覆盖后的模型有效性保证。', '',
        '## 下一步','',
        '独占 GPU 后随机交替基线与注入模式，检验本轮曲线是否复现；再比较逐层同步和低同步加载，量化 instrumentation 对延迟的影响。需要 64K/128K 时，应使用原生支持相应上下文的模型，另建实验口径。','',
        '## 数据来源与复核','',
        '1. [table.csv](table.csv) 和 [summary.csv](summary.csv)：逐配置中位数、分位数、标准差及范围。',
        '2. [configuration.json](configuration.json) 与 [cell_order.json](cell_order.json)：运行设置与执行顺序。',
        '3. [validation-summary.json](validation-summary.json)：完整审计的轻量摘要。','',
        f'服务器完整原始记录目录：`{src}`。CSV 汇总与图片公开于本页；大体积逐层 JSON、逐请求矩阵、硬件采样与日志保留在服务器。', '',
        '本页配置数量均从实际源数据计算，不以预定计划代替成功结果。全部有效配置的样本数和 TTFT 中位数已从 matrix.csv 独立重新分组复核。','',
        '原始源文件 SHA256：','','```text']
    for name in ('matrix.csv','table.csv','summary.csv','configuration.json','validation.json'):
        lines.append(hashlib.sha256((src/name).read_bytes()).hexdigest()+'  '+name)
    lines+=['```','']
    (dst/'README.md').write_text('\n'.join(lines))
    print(dst/'README.md')


if __name__=='__main__':main()
