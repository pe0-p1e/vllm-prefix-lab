#!/usr/bin/env python3
"""Static scientific figures and descriptive affine fits from audited raw runs.

Requires matplotlib (a plotting environment separate from vLLM is sufficient).
No extrapolation or pure compute inference is performed.
"""
import argparse
import csv
import hashlib
import json
import math
import pathlib
import statistics
from collections import defaultdict


def quantile(values, q):
    values = sorted(values)
    p = (len(values)-1)*q
    i = int(p)
    return values[i] + (values[min(i+1,len(values)-1)]-values[i])*(p-i)


def affine_fit(x, y):
    if len(x) != len(y) or len(x) < 3 or len(set(x)) < 2:
        raise ValueError('Need at least three observations and two distinct x values')
    xm, ym = statistics.mean(x), statistics.mean(y)
    slope = sum((a-xm)*(b-ym) for a,b in zip(x,y))/sum((a-xm)**2 for a in x)
    intercept = ym-slope*xm
    residuals = [b-(intercept+slope*a) for a,b in zip(x,y)]
    ss = sum(v*v for v in residuals)
    total = sum((b-ym)**2 for b in y)
    return dict(intercept_ms=intercept, slope_ms_per_gib=slope,
                r_squared=1-ss/total if total else None,
                rmse_ms=math.sqrt(ss/len(x)), n_cells=len(x), n_payload_sizes=len(set(x)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('matrix',type=pathlib.Path)
    p.add_argument('--out',required=True,type=pathlib.Path)
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
        'grid.alpha':.2,'svg.fonttype':'none','pdf.fonttype':42})
    a.out.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    with a.matrix.open() as f:
        for r in csv.DictReader(f):
            if r['status']=='measured' and r['verified']=='True':
                assert int(float(r['injected_hit_tokens']))+int(float(r['actual_prefill_kv_tokens']))==int(r['length'])
                grouped[(int(r['length']),float(r['requested_ratio']))].append(r)
    cells=[]
    for (length,ratio),rs in sorted(grouped.items()):
        cell={'length':length,'ratio':ratio,'n':len(rs)}
        for k in ('ttft_ms','e2e_ms','prefill_service_ms','h2d_cuda_ms','scatter_cuda_ms','worker_load_wall_ms','kv_bytes','injected_hit_tokens'):
            vs=[float(r[k]) for r in rs]
            cell[k]=statistics.median(vs)
            cell[k+'_p10']=quantile(vs,.1)
            cell[k+'_p90']=quantile(vs,.9)
        cell['actual_percent']=100*cell['injected_hit_tokens']/length
        cells.append(cell)
    with (a.out/'plotted_cells.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    lengths=sorted({c['length'] for c in cells})
    colors=plt.get_cmap('tab10')
    footer='Synthetic KV; shared RTX 4090; serialized loads; bands/bars: observed P10–P90, not confidence intervals.'
    def save(fig,name):
        fig.savefig(a.out/(name+'.png'),dpi=220,bbox_inches='tight')
        fig.savefig(a.out/(name+'.svg'),bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(math.ceil(len(lengths)/3),3,figsize=(14,3.5*math.ceil(len(lengths)/3)),squeeze=False)
    for ax,length in zip(axes.flat,lengths):
        cs=[c for c in cells if c['length']==length]
        x=[c['actual_percent'] for c in cs]
        y=[c['ttft_ms'] for c in cs]
        ax.fill_between(x,[c['ttft_ms_p10'] for c in cs],[c['ttft_ms_p90'] for c in cs],color=colors(0),alpha=.15)
        ax.plot(x,y,'o-',label='Client TTFT median',color=colors(0))
        ax.plot(x,[c['prefill_service_ms'] for c in cs],'s--',label='Prefill service median',color=colors(1))
        ax.set(title=f'{length:,} input tokens',xlabel='Actual CPU hit (%)',ylabel='Latency (ms)',xlim=(-2,102),ylim=(0,None))
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(lengths):]:ax.set_visible(False)
    fig.suptitle('Observed latency versus CPU KV hit ratio (each configuration measured separately)')
    fig.text(.5,0,footer,ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,.95));save(fig,'latency_by_hit')
    fig,ax=plt.subplots(figsize=(8,4.8))
    for i,length in enumerate(lengths):
        cs=[c for c in cells if c['length']==length]
        base=next(c['ttft_ms'] for c in cs if c['ratio']==0)
        ax.plot([c['actual_percent'] for c in cs],[base/c['ttft_ms'] for c in cs],'o-',color=colors(i),label=f'{length:,} tokens')
    ax.axhline(1,color='black',linestyle='--',linewidth=1)
    ax.set(xlabel='Actual CPU hit (%)',ylabel='Baseline median TTFT / CPU median TTFT',title='Observed speedup under shared-GPU contention',ylim=(0,None))
    ax.legend(ncol=2,fontsize=8);fig.text(.5,0,footer,ha='center',fontsize=7);fig.tight_layout(rect=(0,.03,1,1));save(fig,'ttft_speedup')
    cpu=[c for c in cells if c['ratio']>0]
    x=[c['kv_bytes']/2**30 for c in cpu]
    fits={}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,metric,title in zip(axes,('h2d_cuda_ms','scatter_cuda_ms'),('H2D movement','GPU paged-cache scatter')):
        y=[c[metric] for c in cpu]
        fit=affine_fit(x,y)
        # Hold out ALL cells sharing one payload, rather than leaking duplicate sizes.
        errs=[]
        for payload in sorted(set(x)):
            train=[i for i,v in enumerate(x) if v!=payload]
            model=affine_fit([x[i] for i in train],[y[i] for i in train])
            errs.extend(y[i]-(model['intercept_ms']+model['slope_ms_per_gib']*x[i]) for i,v in enumerate(x) if v==payload)
        fit['leave_one_payload_out_rmse_ms']=math.sqrt(statistics.mean(e*e for e in errs))
        if metric=='h2d_cuda_ms' and fit['slope_ms_per_gib']>0:
            fit['slope_equivalent_bandwidth_gib_s']=1000/fit['slope_ms_per_gib']
        fits[metric]=fit
        for i,length in enumerate(lengths):
            cs=[c for c in cpu if c['length']==length]
            med=[c[metric] for c in cs]
            ax.errorbar([c['kv_bytes']/2**30 for c in cs],med,
                yerr=[[v-c[metric+'_p10'] for c,v in zip(cs,med)],[c[metric+'_p90']-v for c,v in zip(cs,med)]],
                fmt='o',markersize=4,color=colors(i),alpha=.8,label=f'{length:,}')
        domain=[min(x),max(x)]
        ax.plot(domain,[fit['intercept_ms']+fit['slope_ms_per_gib']*v for v in domain],color='black',linestyle='--',label='Affine fit (cell medians)')
        ax.set(xlabel='CPU KV payload (GiB)',ylabel='CUDA event time (ms)',title=f"{title}: R²={fit['r_squared']:.3f}")
        ax.legend(fontsize=7)
    fig.text(.5,0,footer,ha='center',fontsize=8);fig.tight_layout(rect=(0,.03,1,1));save(fig,'transport_fit')
    fig,ax=plt.subplots(figsize=(8,4.8))
    for i,length in enumerate(lengths):
        cs=[c for c in cpu if c['length']==length]
        ax.plot([c['kv_bytes']/2**30 for c in cs],
                [(c['kv_bytes']/2**30)/(c['h2d_cuda_ms']/1000) for c in cs],
                'o-',color=colors(i),label=f'{length:,} tokens')
    ax.set(xlabel='CPU KV payload (GiB)',ylabel='Payload / median H2D time (GiB/s)',
           title='Effective H2D bandwidth at measured payload sizes',ylim=(0,None))
    ax.legend(fontsize=8,ncol=2)
    fig.text(.5,0,'Ratio of payload to median CUDA time; not median per-request bandwidth. Shared GPU.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1));save(fig,'h2d_bandwidth')
    # Service-phase model, not a measurement of GPU model/Attention time.
    import numpy as np
    feature_names=['intercept_ms','remaining_ktokens','causal_pairs_100m',
                   'payload_gib','cpu_load_indicator']
    features=[]
    for c in cells:
        length,hit=c['length'],c['injected_hit_tokens']
        missing=length-hit
        pairs=missing*(2*hit+missing+1)/2
        features.append([1,missing/1024,pairs/1e8,c['kv_bytes']/2**30,int(hit>0)])
    design=np.asarray(features,dtype=float)
    service=np.asarray([c['prefill_service_ms'] for c in cells])
    coef,_,rank,_=np.linalg.lstsq(design,service,rcond=None)
    assert rank==len(feature_names), 'Service model is not identifiable'
    predicted=design@coef
    residual=service-predicted
    heldout=[]
    for length in lengths:
        mask=np.asarray([c['length']!=length for c in cells])
        beta,_,_,_=np.linalg.lstsq(design[mask],service[mask],rcond=None)
        heldout.extend((service[~mask]-design[~mask]@beta).tolist())
    service_fit=dict(coefficients=dict(zip(feature_names,coef.tolist())),
        r_squared=float(1-np.sum(residual**2)/np.sum((service-service.mean())**2)),
        rmse_ms=float(np.sqrt(np.mean(residual**2))),
        leave_one_length_out_rmse_ms=math.sqrt(statistics.mean(v*v for v in heldout)),
        features='M=(L-H); causal_pairs=M*(2H+M+1)/2. Intercept plus M/1024, pairs/1e8, payload GiB, CPU-load indicator.',
        interpretation='Empirical service-time regression including CPU loading. Coefficients are not measured GPU costs. Leave-one-length-out tests generalization, including boundary-length extrapolation within this native-context experiment. Never extrapolate to unsupported contexts.')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7))
    for i,length in enumerate(lengths):
        mask=np.asarray([c['length']==length for c in cells])
        axes[0].scatter(service[mask],predicted[mask],color=colors(i),label=f'{length:,}')
        axes[1].scatter(predicted[mask],residual[mask],color=colors(i),label=f'{length:,}')
    limit=max(float(service.max()),float(predicted.max()))*1.05
    axes[0].plot([0,limit],[0,limit],'k--',linewidth=1)
    axes[0].set(xlabel='Observed median prefill service (ms)',ylabel='Fitted service time (ms)',
                title=f"Descriptive service model: R²={service_fit['r_squared']:.3f}")
    axes[1].axhline(0,color='black',linestyle='--',linewidth=1)
    axes[1].set(xlabel='Fitted service time (ms)',ylabel='Observed minus fitted (ms)',title='Residuals: no monotonicity constraints')
    axes[0].legend(fontsize=8)
    fig.text(.5,0,'Includes loading and host overhead; coefficients are not pure GPU compute costs. Shared GPU.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.03,1,1));save(fig,'prefill_service_fit')
    result=dict(plot_environment={"matplotlib":matplotlib.__version__,"numpy":np.__version__},matrix_sha256=hashlib.sha256(a.matrix.read_bytes()).hexdigest(),
        n_valid_cells=len(cells),n_measurements=sum(c['n'] for c in cells),
        method='Unweighted OLS on cell medians: time_ms = intercept + slope * payload_GiB. No zero-hit points. Prediction only within observed payload range.',
        caveats='Descriptive shared-GPU fit; samples/cells may be time-correlated. No causal inference, no pure compute inference, no 64K/128K extrapolation. Negative intercepts indicate limited-range empirical behavior, not physical negative overhead.',
        payload_range_gib=[min(x),max(x)],fits=fits,prefill_service_fit=service_fit)
    (a.out/'fits.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 作图与传输拟合','','本页使用同一运行的有效测量数据。连线仅连接实际配置，不是 Prefill 计算模型。阴影和误差棒为各配置观测 P10–P90，不是置信区间。','',
        '![命中率与延迟](latency_by_hit.png)','','![观测加速比](ttft_speedup.png)','','![传输与写入拟合](transport_fit.png)','','![H2D 带宽](h2d_bandwidth.png)','','## 拟合口径与结果','','传输阶段拟合 `时间(ms) = 截距 + 斜率 × KV载荷(GiB)`。对每个配置先取中位数，再做等权最小二乘。留一载荷交叉验证会同时移除相同字节数的全部配置，避免相同载荷泄漏。','','| 阶段 | 截距 ms | 斜率 ms/GiB | R² | 残差 RMSE ms | 留一载荷 RMSE ms |','|---|---:|---:|---:|---:|---:|']
    for metric,fit in fits.items():
        lines.append(f"| {metric} | {fit['intercept_ms']:.3f} | {fit['slope_ms_per_gib']:.3f} | {fit['r_squared']:.4f} | {fit['rmse_ms']:.3f} | {fit['leave_one_payload_out_rmse_ms']:.3f} |")
    lines+=['','## Prefill 服务时间的描述性拟合','',
        '![服务时间拟合与残差](prefill_service_fit.png)','',
        '响应变量是各配置的 Prefill 服务时间中位数。特征为剩余 token 数 M/1024、causal Attention pairs/1e8、KV载荷 GiB、是否加载 CPU KV，以及截距。pairs=M×(2H+M+1)/2 只是全因果 Attention 工作量代理，不能当作实测 kernel FLOPs。','',
        f"R²={service_fit['r_squared']:.4f}；训练残差 RMSE={service_fit['rmse_ms']:.3f} ms；留一输入长度 RMSE={service_fit['leave_one_length_out_rmse_ms']:.3f} ms。留一长度包含边界长度的泛化测试；训练拟合好不代表跨长度预测可靠。",'',
        '**系数不可解释为纯 GPU 每 token 或每 Attention pair 的耗时。** 模型包含加载和主机开销，受共享负载、分块与调度影响；不约束单调性，不外推 64K/128K。精确系数保存在 fits.json。']
    lines+=['',f"拟合使用 {len(cpu)} 个 CPU 配置中位数，{len(set(x))} 个不同载荷；有效测量共 {sum(c['n'] for c in cells)} 次。",'',
        '**限制：**共享 GPU、时间相关性和同步开销会影响拟合。R² 高不代表因果关系；拟合截距不一定具有物理意义。不将服务时间减去 H2D 当作纯计算时间，不外推到 64K/128K。','',
        '精确系数、交叉验证误差与源数据 SHA256 见 [fits.json](fits.json)；每个绘图点及 P10/P90 见 [plotted_cells.csv](plotted_cells.csv)。图片同时提供 SVG，可用于进一步排版。','']
    (a.out/'README.md').write_text('\n'.join(lines))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
