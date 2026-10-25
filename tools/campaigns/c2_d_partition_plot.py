"""Ensemble plots for the C2 i^-1/i^-2 QR partition replay."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from lqcdweave.storage.dataset import DatasetReader
from tools.campaigns.single_current_vdp_plan import CFGS
from tools.campaigns.single_source_dh_analysis import error_statistics


def run(root,baseline,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True)
    values=[];reference=[];identity=[];definition=None
    for cfg in CFGS:
        with DatasetReader(root/'correlators',cfg=cfg) as r, DatasetReader(baseline/'leg-densities'/'V',cfg=cfg) as v, DatasetReader(baseline/'leg-densities'/'Q100',cfg=cfg) as h:
            d=r.spec.document['definition'];times=r.spec.coordinates('dt')
            key=tuple(d[k] for k in ('cases','partition_groups','dirac','estimator','normalization','source_time'))
            if definition is None: definition=key;cases=d['cases'];groups=d['partition_groups']
            elif key!=definition:raise ValueError('different ensemble definitions')
            values.append(np.stack([r.read(dt=t) for t in times]))
            reference.append(np.asarray([(v.read(dt=t)[0].sum()+24**3/64*h.read(dt=t)[0].sum())/24**3 for t in times]))
            identity.append(r.data_id)
    data=np.stack(values);ref=np.stack(reference)
    complete=np.concatenate((data,ref[...,None]),axis=-1)[:,None]
    mean,se,ratios,g,ge,re=error_statistics(complete,24)
    mean,se,ratios,g,ge,re=[x[0] for x in (mean,se,ratios,g,ge,re)]
    # Paired alpha2/alpha1 G ratio: reference cancels, delete the same cfg in both.
    idx={(c['alpha'],c['m'],c['H']):i for i,c in enumerate(cases)}
    n=len(CFGS);pairs=[(idx[1,m,h],idx[2,m,h]) for m in (2,3,4) for h in (8,16,32,64)]
    rel=np.asarray([g[b]/g[a] for a,b in pairs]);deletions=[]
    for k in range(n):
        err=np.delete(complete.real,k,axis=0).std(0,ddof=1)[0]/np.sqrt(n)
        score=np.sqrt(np.mean((err/se[:,24,None])**2,axis=0))
        deletions.append([score[b]/score[a] for a,b in pairs])
    deleted=np.asarray(deletions);relerr=np.sqrt((n-1)/n*np.sum((deleted-deleted.mean(0))**2,axis=0))
    np.savez_compressed(output/'statistics.npz',data=data,reference=ref,mean=mean,se=se,G=g,G_error=ge,
                        dt_ratio=ratios,dt_ratio_error=re,alpha_ratio=rel,alpha_ratio_error=relerr)
    with (output/'scan.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['alpha','m','D','H','R','G','G_jackknife_error'])
        for i,c in enumerate(cases):w.writerow([c['alpha'],c['m'],10*c['m'],c['H'],c['R'],g[i],ge[i]])
    with (output/'dt.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['alpha','m','H','dt','mean_real','mean_imag','SE','SE_ratio_VH64','ratio_error'])
        for i,c in enumerate(cases):
            for t,dt in enumerate(times):w.writerow([c['alpha'],c['m'],c['H'],dt,mean[t,i].real,mean[t,i].imag,se[t,i],ratios[t,i],re[t,i]])
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,m in zip(axs,(2,3,4)):
        for a,color,marker in ((1,'#2474b7','o'),(2,'#e58a00','s')):
            indices=[idx[a,m,h] for h in (8,16,32,64)]
            ax.errorbar((8,16,32,64),g[indices],yerr=ge[indices],color=color,marker=marker,capsize=4,label=f'i^(-{a})')
        ax.set(title=f'D={10*m} ({m} columns / interval)',xlabel='H points (all 3 colors)',ylabel='C2 error G / VH64',xticks=(8,16,32,64));ax.grid(alpha=.2);ax.axhline(1,color='gray',ls='--');ax.legend()
    fig.suptitle('10 QR intervals in modes 101..428 | Q428 H | 18 cfg | tsrc=0 | dt=1..18')
    fig.savefig(output/'comparison.png',dpi=180);fig.savefig(output/'comparison.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for j,m in enumerate((2,3,4)):
        ax.errorbar((8,16,32,64),rel[4*j:4*j+4],yerr=relerr[4*j:4*j+4],marker='o',capsize=4,label=f'D{10*m}')
    ax.axhline(1,color='gray',ls='--');ax.set(xlabel='H points',ylabel='G(i^-2) / G(i^-1)',title='Paired comparison: below 1 favors i^-2',xticks=(8,16,32,64));ax.grid(alpha=.2);ax.legend()
    fig.savefig(output/'paired_ratio.png',dpi=180);plt.close(fig)
    for quantity in ('error','correlator'):
        fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained')
        for ax,h in zip(axs.flat,(8,16,32,64)):
            for m,color in zip((2,3,4),('#2474b7','#129b79','#b051a1')):
                for a,style,marker in ((1,'-','o'),(2,'--','s')):
                    i=idx[a,m,h]
                    y,ye=(ratios[:,i],re[:,i]) if quantity=='error' else (mean[:,i].real,se[:,i])
                    ax.errorbar(times,y,yerr=ye,color=color,ls=style,marker=marker,markersize=3,capsize=2,label=f'D{10*m}, i^-{a}')
            ax.set(title=f'H{h}',xlabel='dt',ylabel='SE / VH64 SE' if quantity=='error' else 'Re C2(dt)')
            if quantity=='error':ax.set_yscale('log')
            else:ax.set_yscale('symlog',linthresh=max(float(np.median(se)),1e-30))
            ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
        fig.savefig(output/f'{quantity}_dt.png',dpi=180);plt.close(fig)
    lines=['# C2：i^-1 与 i^-2 区间划分比较','','18个cfg，流端tsrc=0，dt=1…18；V100固定；medium为101…428模，10个连续区间；H用原Q428全颜色点。',
           '', '每个cfg每种划分只生成一份QR实现，每组4列，m=2/3取同组前缀。两种划分采用相同cfg种子序列，不代表相同随机子空间。',
           '', 'G为逐dt实部标准误差比的均方根，参考同通道V+H64(Q100)。G误差棒固定全样本参考分母；alpha比值误差棒配对删除同一cfg。',
           '', '|D|H|R|G i^-1|G i^-2|i^-2 / i^-1|','|---:|---:|---:|---:|---:|---:|']
    for j,(ia,ib) in enumerate(pairs):
        c=cases[ia];lines.append(f'|{10*c["m"]}|{c["H"]}|{c["R"]}|{g[ia]:.4f} ± {ge[ia]:.4f}|{g[ib]:.4f} ± {ge[ib]:.4f}|{rel[j]:.4f} ± {relerr[j]:.4f}|')
    lines+=['','## 实际区间边界','']
    for a in ('1','2'):lines.append(f'- i^-{a}: '+', '.join(f'{x[0]+1}–{x[-1]+1}' for x in groups[a]))
    checks=[json.loads((root/f'complete-{cfg}.json').read_text()) for cfg in CFGS]
    lines+=['','## 验证与限制','',f'- 旧D40传播子重建最大相对差：{max(x["old_D_reconstruction_max_relative"] for x in checks):.5g}；V与旧D40的C2逐腿贡献通过原生核复现核验。',
            '- 新D从完整父传播子线性组合，保留交叉项；未把旧D逐腿标量直接重分组。',
            '- 每组使用n/m一阶权重一次，H使用空间体积/H一次；无新Dirac求解。',
            '- 18个cfg、一个流源时间、一份随机实现；局部误差变化不保证单调，不据单份结果判定普适最优划分。',
            '- 原生产求解容差1e-7；本次浮点重组不消除原求解误差。','', '![误差比较](comparison.png)','', '![配对比值](paired_ratio.png)','', '![逐dt误差](error_dt.png)','', '![关联函数](correlator_dt.png)']
    (output/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    (output/'provenance.json').write_text(json.dumps(dict(cfgs=list(CFGS),cases=cases,inputs=identity,checks=checks),indent=2),encoding='utf-8')
    print('\n'.join(lines[8:23]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('input','baseline','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.input,a.baseline,a.output)
