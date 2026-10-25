"""Reassemble a saved single-current ensemble and plot dense D/H count scans."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path

import numpy as np

from lqcdweave.contracts.dilution import DilutionSource
from lqcdweave.storage.dataset import DatasetReader
from lqcdweave.contraction.external_current_measurement import assemble_cases
from tools.campaigns.single_current_vdp_plan import CFGS, scan_cases


def cases():
    result = [dict(case_id=f'VD-D{d}', route='VD', d_columns=d, point_count=0,
                   sampled_components=d, p_projection=None) for d in range(1, 97)]
    result += [dict(case_id=f'VP-P{h}', route='VP', d_columns=0, point_count=h,
                    sampled_components=3*h, p_projection='Q100') for h in range(1, 65)]
    for m in (2, 3, 4):
        result += [dict(case_id=f'VDP-m{m}-P{h}', route='VDP', d_columns=10*m,
                        columns_per_group=m, point_count=h, sampled_components=10*m+3*h,
                        p_projection='Q428') for h in range(1, 65)]
    return result


def error_statistics(data, reference):
    n = len(data)
    mean = data.mean(0)
    se = data.real.std(0, ddof=1)/np.sqrt(n)
    if np.any(se[..., reference] <= 0):
        raise ValueError('nonpositive reference error')
    ratios = se/se[..., reference, None]
    g = np.sqrt(np.mean(ratios**2, axis=1))
    deleted, paired = [], []
    for k in range(n):
        err = np.delete(data.real, k, axis=0).std(0, ddof=1)/np.sqrt(n)
        deleted.append(np.sqrt(np.mean((err/se[..., reference, None])**2, axis=1)))
        paired.append(err/err[..., reference, None])
    def jk(values):
        a = np.asarray(values)
        return np.sqrt((n-1)/n*np.sum((a-a.mean(0))**2, axis=0))
    return mean, se, ratios, g, jk(deleted), jk(paired)


def run(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    selected = cases(); arrays = []; identities = []; signature = None
    maximum_difference = 0.
    for cfg in CFGS:
        with ExitStack() as stack:
            readers = {name: stack.enter_context(DatasetReader(root/'leg-densities'/name, cfg=cfg))
                       for name in ('V', 'Dfull', 'Dmedium', 'Q100', 'Q428')}
            first = readers['V']; definition = first.spec.document['definition']
            times = first.spec.coordinates('dt')
            sig = (times, definition['normalization'], definition['estimator'],
                   definition['dirac'], definition['elemental_gauge'], definition.get('operators'))
            if signature is None: signature = sig
            elif signature != sig: raise ValueError('incompatible ensemble definitions')
            if any(r.spec.document['definition'] != definition for r in readers.values()):
                raise ValueError('different leg definitions')
            source = DilutionSource.from_document(definition['medium_source'])
            original = stack.enter_context(DatasetReader(root/'correlators', cfg=cfg))
            cfg_data = []
            for dt in times:
                blocks = {name: r.read(dt=dt) for name, r in readers.items()}
                def assemble(items):
                    return np.stack([assemble_cases({name: a[i] for name, a in blocks.items()}, items,
                        medium_source=source, spatial_sites=24**3, low_modes=100)/24**3
                        for i in range(5)])
                recovered = assemble(scan_cases()); saved = original.read(dt=dt)
                np.testing.assert_allclose(recovered, saved, rtol=1e-11, atol=1e-13)
                maximum_difference = max(maximum_difference, float(np.max(np.abs(recovered-saved))))
                a = assemble(selected)
                cfg_data.append(np.stack((a[0], a[1], a[2:5].mean(0))))
            arrays.append(np.stack(cfg_data, axis=1))
            identities.append({name: r.data_id for name, r in readers.items()})
    data = np.stack(arrays)
    ref = next(i for i, c in enumerate(selected) if c['case_id']=='VP-P64')
    mean, se, ratio, g, ge, re = error_statistics(data, ref)
    np.savez_compressed(output/'ensemble.npz', data=data, mean=mean, standard_error=se,
                        error_ratio=ratio, G=g, G_error=ge, ratio_error=re, dt=times)
    report = dict(configurations=list(CFGS), cases=selected, reference='VH-Q100-H64',
                  source_time=0, dt=list(times), identities=identities,
                  original_38_case_maximum_absolute_difference=maximum_difference,
                  metric='RMS of pointwise real standard-error ratio; G jackknife fixed denominator',
                  selection='nested original point and per-group column prefixes; no subset averaging',
                  H='full-color spatial points; Q100 for VH, Q428 for VDH',
                  limitations='18 cfg, one source time; no independent sampling replicates or unbiased exact reference')
    (output/'analysis.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (output/'scan.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f); writer.writerow(['channel','case','D','H','R','G','G_jackknife_error'])
        for ch, name in enumerate(('C2','C2p','C3')):
            for i, c in enumerate(selected):
                writer.writerow([name,c['case_id'],c['d_columns'],c['point_count'],c['sampled_components'],g[ch,i],ge[ch,i]])
    with (output/'dt.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['channel','case','dt','mean_real','mean_imag','SE','error_ratio','ratio_jackknife_error'])
        for ch,name in enumerate(('C2','C2p','C3')):
            for i,c in enumerate(selected):
                for t,dt in enumerate(times):
                    w.writerow([name,c['case_id'],dt,mean[ch,t,i].real,mean[ch,t,i].imag,se[ch,t,i],ratio[ch,t,i],re[ch,t,i]])
    colors = ['#2474b7','#e58a00','#129b79','#b051a1','#ca4e49']
    groups = [[i for i,c in enumerate(selected) if c['route']=='VD'],
              [i for i,c in enumerate(selected) if c['route']=='VP']]
    groups += [[i for i,c in enumerate(selected) if c['route']=='VDP' and c['columns_per_group']==m] for m in (2,3,4)]
    labels = ['V+D full complement','V+H (Q100)','V+D20+H (Q428)','V+D30+H (Q428)','V+D40+H (Q428)']
    fig, axes = plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for ch, ax in enumerate(axes):
        for ix,label,color in zip(groups,labels,colors):
            x = [selected[i]['sampled_components'] for i in ix]
            ax.plot(x,g[ch,ix],label=label,color=color)
            ax.fill_between(x,np.maximum(g[ch,ix]-ge[ch,ix],1e-12),g[ch,ix]+ge[ch,ix],color=color,alpha=.12)
        ax.set(title=('C2','C2p','C3')[ch],xlabel='Sampled components R = ND + 3 NH',ylabel='Error ratio G vs VH64',yscale='log')
        ax.axhline(1,color='gray',ls='--'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.suptitle('Single current tsrc=0 | 18 cfg | dt=1..18 | fixed V100')
    fig.savefig(output/'budget.png',dpi=180); fig.savefig(output/'budget.pdf'); plt.close(fig)
    fig, axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    for ch, ax in enumerate(axes):
        for ix,label,color in zip(groups[1:],labels[1:],colors[1:]):
            ax.plot(range(1,65),g[ch,ix],label=label,color=color)
        ax.set(title=('C2','C2p','C3')[ch],xlabel='H points (3 colors each)',ylabel='G vs VH64',yscale='log');ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(output/'H_scan.png',dpi=180);plt.close(fig)
    picks=['VD-D48','VD-D96','VP-P16','VP-P32','VP-P64','VDP-m2-P16','VDP-m3-P16','VDP-m4-P16','VDP-m4-P32','VDP-m4-P64']
    for ch,name in enumerate(('C2','C2p','C3')):
        fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
        for case in picks:
            i=next(i for i,c in enumerate(selected) if c['case_id']==case)
            axes[0].errorbar(times,mean[ch,:,i].real,yerr=se[ch,:,i],label=case,marker='.',lw=.8,capsize=2)
            axes[1].errorbar(times,ratio[ch,:,i],yerr=re[ch,:,i],label=case,marker='.',lw=.8,capsize=2)
        axes[0].set(title=name+' real correlator',ylabel='Re C(dt)')
        axes[0].set_yscale('symlog',linthresh=max(float(np.median(se[ch])),1e-30))
        axes[1].set(xlabel='dt',ylabel='Pointwise SE / VH64 SE',yscale='log')
        for ax in axes: ax.grid(alpha=.2)
        axes[0].legend(ncol=2,fontsize=8);fig.savefig(output/(name+'_dt.png'),dpi=180);plt.close(fig)
    print(json.dumps(dict(cases=len(selected),cfgs=len(CFGS),max_reassembly_difference=maximum_difference,output=str(output))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.input,args.output)
