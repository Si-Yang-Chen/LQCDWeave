"""Statistics for independent V+H draws, averaging inside each cfg first."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from lqcdweave.storage.dataset import DatasetReader
from tools.campaigns.single_source_dh_analysis import error_statistics

ESTIMATOR='ne20-finite-VH-independent-point-repeats-v1'


def assemble_averages(raw,baseline,prefixes):
    """Input axes cfg,dt,measurement,replica,H; cfg remains the sample axis."""
    channel=np.stack((raw[:,:,0],raw[:,:,1],raw[:,:,2:5].mean(2)),axis=1)
    base=np.stack((baseline[:,:,0],baseline[:,:,1],baseline[:,:,2:5].mean(2)),axis=1)
    averaged=np.stack([channel[...,:k,:].mean(-2) for k in prefixes],axis=-2)
    data=np.concatenate((base,averaged.reshape(*base.shape[:-1],-1)),axis=-1)
    return data,channel


def run(root,output,cfgs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    draws=[];baselines=[];costs=[];links=[];ids=[];definition=None;cases=None
    for cfg in cfgs:
        with DatasetReader(root/'correlators',cfg=cfg) as r,DatasetReader(root/'baseline',cfg=cfg) as b,\
             DatasetReader(root/'unique-points',cfg=cfg) as c,DatasetReader(root/'links',cfg=cfg) as l:
            actual=r.spec.document['definition'];bd=b.spec.document['definition']
            if actual['estimator']!=ESTIMATOR or bd['copy_origin_id']!=actual['comparison_id']:
                raise ValueError('wrong independent redraw estimator/reference')
            signature={k:actual[k] for k in ('estimator','source_time','meson_source_time','current_time',
                'meson_modes','current_low_modes','dt','normalization','dirac','elemental_gauge',
                'operators','measurements','point_repetitions','point_counts','point_capacity','repeat_prefixes',
                'point_seed_base','point_lattice','point_design','H_projection','averaging','preparation','budget')}
            if definition is None:definition=signature;cases=list(bd['cases'])
            elif signature!=definition or bd['cases']!=cases:raise ValueError('incompatible redraw ensemble')
            times=actual['dt']
            draws.append(np.stack([r.read(dt=t) for t in times]))
            baselines.append(np.stack([b.read(dt=t) for t in times]))
            costs.append(np.stack([c.read(dt=t) for t in times]));links.append(np.stack([l.read(dt=t) for t in times]))
            ids.append(r.data_id)
    counts=definition['point_counts'];prefixes=definition['repeat_prefixes'];nbase=len(cases)
    for k in prefixes:
        for h in counts:cases.append(dict(case_id=f'VH-redraw-H{h}-K{k}',family='VH-repeat',D=0,H=h,
            K=k,R=3*h,total_draw_components=3*h*k))
    data,channel=assemble_averages(np.stack(draws),np.stack(baselines),prefixes)
    lookup={c['case_id']:i for i,c in enumerate(cases)};reference=lookup['FULL']
    mean,se,ratio,g,ge,re=error_statistics(data,reference)
    residual=data.real-data.real[...,reference,None]
    paired_mean=residual.mean(0);paired_se=residual.std(0,ddof=1)/np.sqrt(len(cfgs))
    paired_z=np.divide(paired_mean,paired_se,out=np.zeros_like(paired_mean),where=paired_se>0)
    scale=np.sqrt(np.mean(data.real[...,reference]**2,axis=(0,2)))
    relative_residual=np.sqrt(np.mean(residual**2,axis=(0,2)))/scale[:,None]
    within_var=channel.real.var(axis=-2,ddof=1).mean(0)
    sampling_se=np.stack([np.sqrt(within_var/(len(cfgs)*k)) for k in prefixes],axis=-2)
    cost=np.stack(costs);geometry=np.stack(links);unique_mean=3*cost.mean(axis=(0,1))
    np.savez_compressed(output/'ensemble.npz',data=data,per_draw_data=channel,mean=mean,standard_error=se,
        G=g,G_error=ge,error_ratio=ratio,ratio_error=re,relative_residual=relative_residual,
        paired_mean_difference=paired_mean,paired_difference_SE=paired_se,paired_difference_z=paired_z,
        sampling_SE_estimate=sampling_se,unique_high_components=3*cost,dt=times)
    channels=('C2','C2p','C3');colors=plt.get_cmap('tab10')

    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=180);fig.savefig(output/(name+'.pdf'));plt.close(fig)

    def draw(ax,ch,ii,label,color,style='-'):
        if not ii:return
        ax.errorbar([cases[i]['R'] for i in ii],g[ch,ii],yerr=ge[ch,ii],label=label,color=color,
            ls=style,marker='o',ms=3,capsize=2,lw=1.2)

    fig,axs=plt.subplots(1,3,figsize=(21,7.5),layout='constrained')
    for ch,ax in enumerate(axs):
        draw(ax,ch,[i for i,c in enumerate(cases[:nbase]) if c['family']=='VD' and c['D'] in (8,16,24,32,48,63,96)],'V+D, one draw','#2474b7')
        draw(ax,ch,[lookup[f'VH-H{h}'] for h in counts],'V+H, original points','#999999','--')
        first=True
        for alpha in (1,2):
            for m in (2,3,4):
                ii=[i for i,c in enumerate(cases[:nbase]) if c['family']=='weighted' and c['alpha']==alpha and c['m']==m]
                if ii:
                    ax.plot([cases[i]['R'] for i in ii],g[ch,ii],color='#bbbbbb',ls=':',lw=.8,
                        label='Previous V+D+H, q=3' if first else None);first=False
        for j,k in enumerate(sorted({1,4,16,prefixes[-1]} & set(prefixes))):
            draw(ax,ch,[lookup[f'VH-redraw-H{h}-K{k}'] for h in counts],f'V+H, {k} independent draws',colors(j+1))
        ax.axhline(1,color='gray',ls='--');ax.grid(alpha=.2)
        ax.set(title=channels[ch],xlabel='Components per draw R (D: ND; H: 3 NH)',ylabel='G = RMS_dt [SE / SE(FULL)]',yscale='log')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
    fig.suptitle(f'V+H fresh full-volume point draws | {len(cfgs)} cfg, dt=1..18 | meson V20, current V100\n'
        'Average draws within each cfg. K draws use K times the draw budget; unique-input cost shown separately.')
    save(fig,'all_schemes')

    for mode in ('repeat_convergence','cost_comparison','reference_residual'):
        fig,axs=plt.subplots(1,3,figsize=(20,6.5),layout='constrained')
        for ch,ax in enumerate(axs):
            for hi,h in enumerate(counts):
                ii=[lookup[f'VH-redraw-H{h}-K{k}'] for k in prefixes]
                x=unique_mean[:,hi] if mode=='cost_comparison' else prefixes
                y=relative_residual[ch,ii] if mode=='reference_residual' else g[ch,ii]
                if mode=='reference_residual':ax.plot(x,y,color=colors(hi),marker='o',label=f'H{h}')
                else:ax.errorbar(x,y,yerr=ge[ch,ii],color=colors(hi),marker='o',capsize=2,label=f'H{h}')
            if mode=='cost_comparison':
                ii=[i for i,c in enumerate(cases[:nbase]) if c['family']=='VD' and c['D'] in (8,16,24,32,48,63,96)]
                ax.plot([cases[i]['R'] for i in ii],g[ch,ii],color='black',ls='--',label='V+D one draw')
            else:ax.set_xticks(prefixes,[str(k) for k in prefixes])
            ax.set_xscale('log',base=10 if mode=='cost_comparison' else 2);ax.set_yscale('log');ax.grid(alpha=.2)
            ax.set(title=channels[ch],xlabel='Unique high components, mean 3 N_unique' if mode=='cost_comparison' else 'Independent point draws K',
                ylabel='RMS(C - FULL) / RMS(FULL)' if mode=='reference_residual' else 'G / FULL')
            if mode!='reference_residual':ax.axhline(1,color='gray',ls='--')
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
        fig.suptitle('Independent point redraws; cfg count unchanged | '+
            ('Shared point propagators counted once per unique site; draw budget 3 NH K also supplied in CSV' if mode=='cost_comparison' else
             'Paired complete-space residual diagnostic' if mode=='reference_residual' else 'Contract and weight each draw before averaging within cfg'))
        save(fig,mode)

    h=21 if 21 in counts else counts[len(counts)//2]
    names=['FULL']+([f'VD-D{3*h}'] if f'VD-D{3*h}' in lookup else [])+[f'VH-H{h}']+[
        f'VH-redraw-H{h}-K{k}' for k in sorted({1,4,16,prefixes[-1]} & set(prefixes))]
    chosen=[lookup[name] for name in names]
    for mode in ('correlators','per_dt_error_ratio','reference_difference'):
        fig,axs=plt.subplots(1,3,figsize=(20,6.5),layout='constrained')
        for ch,ax in enumerate(axs):
            for j,i in enumerate(chosen):
                y=mean[ch,:,i].real if mode=='correlators' else paired_mean[ch,:,i] if mode=='reference_difference' else ratio[ch,:,i]
                err=se[ch,:,i] if mode=='correlators' else paired_se[ch,:,i] if mode=='reference_difference' else re[ch,:,i]
                ax.errorbar(np.array(times)+(j-(len(chosen)-1)/2)*.045,y,yerr=err,label=cases[i]['case_id'],
                    color='black' if i==reference else None,marker='.',capsize=2,lw=1)
            ax.set(title=channels[ch],xlabel='dt',ylabel='Re C(dt)' if mode=='correlators' else
                'Mean Re(C - FULL)' if mode=='reference_difference' else 'SE / SE(FULL)')
            if mode=='per_dt_error_ratio':ax.set_yscale('log');ax.axhline(1,color='gray',ls='--')
            else:ax.set_yscale('symlog',linthresh=max(float(np.median(se[ch,:,reference])),1e-30))
            ax.grid(alpha=.2)
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
        fig.suptitle(f'Per-draw H{h}, R={3*h}; different K have different total budgets | real-part cfg error bars')
        save(fig,mode)

    with (output/'scan.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['channel','case','H','K','R_per_draw','total_draw_components','mean_unique_components','G','G_error','relative_residual'])
        for i,c in enumerate(cases):
            unique=unique_mean[prefixes.index(c['K']),counts.index(c['H'])] if c['family']=='VH-repeat' else c['R']
            for ch,name in enumerate(channels):w.writerow([name,c['case_id'],c['H'],c.get('K',1),c['R'],c.get('total_draw_components',c['R']),unique,g[ch,i],ge[ch,i],relative_residual[ch,i]])
    with (output/'per_dt.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['channel','case','dt','mean_real','SE','error_ratio','ratio_error','paired_difference','paired_difference_SE','paired_z'])
        for i,c in enumerate(cases):
            for ch,name in enumerate(channels):
                for ti,t in enumerate(times):w.writerow([name,c['case_id'],t,mean[ch,ti,i].real,se[ch,ti,i],ratio[ch,ti,i],re[ch,ti,i],paired_mean[ch,ti,i],paired_se[ch,ti,i],paired_z[ch,ti,i]])
    link_summary=[];nvolume=np.prod(definition['point_lattice'][:3])
    with (output/'C3_links.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['H','K','actual_directed_link_occurrences','expected','mean_unique_high_components'])
        for ki,k in enumerate(prefixes):
            for hi,h in enumerate(counts):
                entry=dict(H=h,K=k,actual=int(geometry[:,:,:k,hi].sum()),
                    expected=float(len(cfgs)*len(times)*k*6*h*(h-1)/(nvolume-1)),mean_unique_components=float(unique_mean[ki,hi]))
                link_summary.append(entry);w.writerow([h,k,entry['actual'],entry['expected'],entry['mean_unique_components']])
    results=[]
    for i,c in enumerate(cases[nbase:],nbase):
        oldi=lookup[f"VH-H{c['H']}"]
        results.append(dict(**c,G=g[:,i].tolist(),G_error=ge[:,i].tolist(),
            G_over_original_VH=(g[:,i]/g[:,oldi]).tolist(),relative_residual=relative_residual[:,i].tolist(),
            max_abs_paired_z=np.max(abs(paired_z[:,:,i]),axis=1).tolist()))
    report=dict(**definition,cases=cases,configurations=list(cfgs),cfg_count=len(cfgs),baseline_case_count=nbase,
        cases_count=len(cases),input_ids=ids,results=results,C3_links=link_summary,
        statistics='Average independent draws within cfg; C3 directions within cfg/draw; real SE across cfg only',
        limitation='Extra draws increase sampling budget; empirical SE need not be monotone; compare FULL residuals and actual links',
        postprocessing_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'analysis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(cfgs=len(cfgs),draws=definition['point_repetitions'],cases=len(cases),output=str(output))))
