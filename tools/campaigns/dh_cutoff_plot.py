"""Compare budget-derived D/H boundaries with the unchanged finite baseline."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from lqcdweave.storage.dataset import DatasetReader
from tools.campaigns.single_current_vdp_plan import CFGS
from tools.campaigns.single_source_dh_analysis import error_statistics

ESTIMATOR='ne20-finite-two-legs-budget-weighted-DH-v1'


def paired_gain_uncertainty(data,reference,new_indices,old_indices):
    """Correlated delete-cfg uncertainty of G(new)/G(old), same frozen FULL SE."""
    n=len(data);reference_se=data.real.std(0,ddof=1)[...,reference]/np.sqrt(n)
    values=[]
    for k in range(n):
        se=np.delete(data.real,k,axis=0).std(0,ddof=1)/np.sqrt(n)
        g=np.sqrt(np.mean((se/reference_se[...,None])**2,axis=1))
        values.append(g[:,new_indices]/g[:,old_indices])
    a=np.asarray(values)
    return np.sqrt((n-1)/n*np.sum((a-a.mean(0))**2,axis=0))


def run(root,output,cfgs=CFGS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True); arrays=[]; ids=[]; definition=None
    for cfg in cfgs:
        with DatasetReader(root/'correlators',cfg=cfg) as reader:
            actual=reader.spec.document['definition']
            if actual['estimator']!=ESTIMATOR:raise ValueError('wrong finite estimator')
            signature={k:actual[k] for k in ('estimator','source_time','dt','cases','planned_designs',
                'normalization','elemental_gauge','operators','measurements','dirac','meson_modes',
                'current_low_modes','meson_source_time','current_time','partition_rule')}
            if definition is None:definition=signature
            elif signature!=definition:raise ValueError('incompatible ensemble science/case ordering')
            a=np.stack([reader.read(dt=t) for t in actual['dt']])
            arrays.append(np.stack((a[:,0],a[:,1],a[:,2:5].mean(1))));ids.append(reader.data_id)
    cases=definition['cases']; times=definition['dt'];lookup={c['case_id']:i for i,c in enumerate(cases)}
    data=np.stack(arrays);reference=lookup['FULL'];mean,se,ratio,g,ge,re=error_statistics(data,reference)
    residual=data.real-data.real[...,reference,None]
    residual_rms=np.sqrt(np.mean(residual**2,axis=(0,2)))
    scale=np.sqrt(np.mean(data.real[...,reference]**2,axis=(0,2)))
    relative_residual=residual_rms/scale[:,None]
    paired_mean=residual.mean(0);paired_se=residual.std(0,ddof=1)/np.sqrt(len(cfgs))
    paired_z=np.divide(paired_mean,paired_se,out=np.zeros_like(paired_mean),where=paired_se>0)
    np.savez_compressed(output/'ensemble.npz',data=data,mean=mean,standard_error=se,G=g,G_error=ge,
        error_ratio=ratio,ratio_error=re,relative_residual=relative_residual,paired_mean_difference=paired_mean,
        paired_difference_SE=paired_se,paired_difference_z=paired_z,dt=times)
    colors={2:'#119c79',3:'#ae4d9d',4:'#ca4a43'}; channels=('C2','C2p','C3')
    qs=sorted({c['tail_multiplier'] for c in definition['planned_designs']})
    primary_q=3 if 3 in qs else 1
    rules=[(alpha,q) for alpha in (1,2) for q in qs]
    point_counts=sorted({c['H'] for c in definition['planned_designs']})
    rule_colors=plt.get_cmap('tab10')
    profile_label=lambda alpha: r'$\zeta_i$' if alpha==1 else r'$\zeta_i^2$'
    rule_label=lambda alpha,q: f'{profile_label(alpha)}, q={q:g}'

    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=180);fig.savefig(output/(name+'.pdf'));plt.close(fig)

    def draw(ax,ch,indices,label,color,style='-',annotate=False):
        if not indices:return
        indices=sorted(indices,key=lambda i:cases[i]['R'])
        x=[cases[i]['R'] for i in indices];y=g[ch,indices]
        ax.errorbar(x,y,yerr=ge[ch,indices],marker='o',ms=4,lw=1.2,capsize=2,color=color,ls=style,label=label)
        if annotate:
            for i,xx,yy in zip(indices,x,y):ax.annotate(f"B{cases[i]['cutoff']}",(xx,yy),xytext=(3,4),textcoords='offset points',fontsize=7,color=color)

    fig,axs=plt.subplots(1,3,figsize=(21,7.5),layout='constrained')
    for ch,ax in enumerate(axs):
        draw(ax,ch,[i for i,c in enumerate(cases) if c['family']=='VD' and c['D'] in (8,16,24,32,48,63,96)],'V+D full complement','#2375b7')
        draw(ax,ch,[lookup[f'VH-H{h}'] for h in point_counts],'V+H, Q100','#e68a00')
        for m in (2,3,4):
            draw(ax,ch,[lookup[f'VDH-original-m{m}-H{h}'] for h in point_counts],f'Fixed B428, D{10*m}+H',colors[m],':')
            for alpha,style in ((1,'--'),(2,'-')):
                draw(ax,ch,[i for i,c in enumerate(cases) if c['family']=='weighted' and c['alpha']==alpha and c['m']==m and c['tail_multiplier']==primary_q],
                    f'{profile_label(alpha)}, D{10*m}+H',colors[m],style)
        ax.set(title=channels[ch],xlabel='Finite components R = ND + 3 NH',ylabel='G = RMS_dt [SE / SE(FULL)]',yscale='log')
        ax.axhline(1,color='gray',ls='--');ax.grid(alpha=.2)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=10)
    fig.suptitle(f'Global zeta / zeta^2 partition | D block : highest H tail mass = m : {primary_q:g} NH | {len(cfgs)} cfg, dt=1..18\n'
        'Meson V20 at t=0; current V100; both finite legs sampled. Uncomputed designs omitted: missing eigenmodes or n < m.')
    save(fig,'weighted_comparison')

    fig,axs=plt.subplots(2,3,figsize=(21,12),layout='constrained')
    for row,alpha in enumerate((1,2)):
        for ch in range(3):
            ax=axs[row,ch]
            draw(ax,ch,[i for i,c in enumerate(cases) if c['family']=='VD' and c['D'] in (8,16,24,32,48,63,96)],'V+D full complement','#2375b7')
            draw(ax,ch,[lookup[f'VH-H{h}'] for h in point_counts],'V+H Q100','#e68a00')
            for m in (2,3,4):
                for qi,q in enumerate(qs):
                    style=('-', '--', ':', '-.')[qi%4]
                    ii=[i for i,c in enumerate(cases) if c['family']=='weighted' and c['alpha']==alpha and c['m']==m and c['tail_multiplier']==q]
                    draw(ax,ch,ii,f'D{10*m}, q={q:g}',colors[m],style)
            ax.set(title=f'{channels[ch]} | i^-{alpha}',xlabel='Finite components R = ND + 3 NH',ylabel='G / FULL',yscale='log')
            ax.axhline(1,color='gray',ls='--');ax.grid(alpha=.2)
    handles,labels=axs[1,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=10)
    fig.suptitle('All executable zeta / zeta^2 rules | q='+', '.join(f'{q:g}' for q in qs)+' | unavailable designs excluded')
    save(fig,'all_rules')

    fig,axs=plt.subplots(3,3,figsize=(17,13),layout='constrained')
    for ch in range(3):
        for col,m in enumerate((2,3,4)):
            ax=axs[ch,col]
            old=[lookup[f'VDH-original-m{m}-H{h}'] for h in point_counts]
            ax.errorbar(point_counts,g[ch,old],yerr=ge[ch,old],marker='s',capsize=2,color='gray',ls=':',label='Fixed B428')
            for ri,(alpha,q) in enumerate(rules):
                color=rule_colors(ri%10)
                ii=[i for i,c in enumerate(cases) if c['family']=='weighted' and c['alpha']==alpha and c['m']==m and c['tail_multiplier']==q]
                if ii:ax.errorbar([cases[i]['H'] for i in ii],g[ch,ii],yerr=ge[ch,ii],marker='o',capsize=2,color=color,label=rule_label(alpha,q))
            ax.set(title=f'{channels[ch]} | D{10*m}',xlabel='Full-color H points',ylabel='G / FULL',yscale='log',xticks=point_counts);ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
    fig.suptitle('Same D count and same H point count | tail / single D block importance = q NH / m\n'
        'q=3 counts full-color point components; point inclusion probabilities still use NH spatial labels.')
    save(fig,'tail_ratio_scan')

    fig,axs=plt.subplots(3,3,figsize=(17,13),layout='constrained')
    for ch in range(3):
        for col,m in enumerate((2,3,4)):
            ax=axs[ch,col];old=[lookup[f'VDH-original-m{m}-H{h}'] for h in point_counts]
            ax.plot(point_counts,relative_residual[ch,old],marker='s',color='gray',ls=':',label='Fixed B428')
            for ri,(alpha,q) in enumerate(rules):
                color=rule_colors(ri%10)
                ii=[i for i,c in enumerate(cases) if c['family']=='weighted' and c['alpha']==alpha and c['m']==m and c['tail_multiplier']==q]
                if ii:ax.plot([cases[i]['H'] for i in ii],relative_residual[ch,ii],marker='o',color=color,label=rule_label(alpha,q))
            ax.set(title=f'{channels[ch]} | D{10*m}',xlabel='Full-color H points',ylabel='RMS(C - FULL) / RMS(FULL)',yscale='log',xticks=point_counts);ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
    fig.suptitle('Paired deviation from complete-space correlators | same D/H counts\n'
        'Complementary diagnostic for rare-link coverage; this is not the statistical-error metric.')
    save(fig,'tail_ratio_residual')

    names=['FULL','VH-H42','VDH-original-m3-H32']+[f'a{alpha}-m3-H32-q{q:g}' for alpha,q in rules]
    selection_title='Matched R=126: D30+H32 vs VH42'
    if all(f'a{alpha}-m2-H24-q3' in lookup for alpha in (1,2)):
        names=['FULL','VD-D92','VH-H30','VDH-original-m2-H24','a1-m2-H24-q3','a2-m2-H24-q3']
        selection_title='Same D20+H24 (R=92): zeta vs zeta^2 | VD92 (R=92), VH30 (R=90)'
    chosen=[lookup[n] for n in names if n in lookup]
    for mode in ('correlators','reference_difference','per_dt_error_ratio'):
        fig,axs=plt.subplots(1,3,figsize=(19,6.5),layout='constrained')
        for ch,ax in enumerate(axs):
            for j,i in enumerate(chosen):
                c=cases[i];label=c['case_id']+(f" (B{c['cutoff']})" if 'cutoff' in c else '')
                y=mean[ch,:,i].real if mode=='correlators' else paired_mean[ch,:,i] if mode=='reference_difference' else ratio[ch,:,i]
                error=se[ch,:,i] if mode=='correlators' else paired_se[ch,:,i] if mode=='reference_difference' else re[ch,:,i]
                ax.errorbar(np.array(times)+(j-2.5)*.05,y,yerr=error,label=label,marker='.',capsize=2,lw=1,color='black' if i==reference else None)
            ax.set(title=channels[ch],xlabel='dt',ylabel='Re C(dt)' if mode=='correlators' else 'Mean Re(C - FULL)' if mode=='reference_difference' else 'SE / SE(FULL)')
            if mode=='per_dt_error_ratio':ax.set_yscale('log');ax.axhline(1,color='gray',ls='--')
            else:ax.set_yscale('symlog',linthresh=max(float(np.median(se[ch,:,reference])),1e-30))
            ax.grid(alpha=.2)
        handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
        fig.suptitle(selection_title+' | real-part cfg error bars | FULL reference')
        save(fig,mode)

    new_indices=[i for i,c in enumerate(cases) if c['family']=='weighted']
    old_indices=[lookup[f"VDH-original-m{cases[i]['m']}-H{cases[i]['H']}"] for i in new_indices]
    gain_error=paired_gain_uncertainty(data,reference,new_indices,old_indices)
    fig,axs=plt.subplots(1,3,figsize=(19,6.5),layout='constrained')
    for ch,ax in enumerate(axs):
        for m in (2,3,4):
            for alpha,style in ((1,'--'),(2,'-')):
                jj=[j for j,i in enumerate(new_indices) if cases[i]['alpha']==alpha and cases[i]['m']==m and cases[i]['tail_multiplier']==primary_q]
                if not jj:continue
                ii=np.asarray(new_indices)[jj];oi=np.asarray(old_indices)[jj]
                ax.errorbar([cases[i]['R'] for i in ii],g[ch,ii]/g[ch,oi],yerr=gain_error[ch,jj],marker='o',capsize=3,color=colors[m],ls=style,label=f'{profile_label(alpha)}, D{10*m}')
        ax.axhline(1,color='gray',ls='--');ax.grid(alpha=.2)
        ax.set(title=channels[ch],xlabel='Finite components R = ND + 3 NH',ylabel='G(new boundary) / G(fixed B428)')
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle(f'Same D/H counts | q={primary_q:g}, zeta / zeta^2 | paired cfg jackknife for error-ratio changes\n'
        'Below 1: lower observed statistical error; check FULL residuals separately, especially C3.')
    save(fig,'gain_over_fixed428')
    matches=[]
    for i,c in enumerate(cases):
        if c['family']!='weighted':continue
        oldi=lookup[f"VDH-original-m{c['m']}-H{c['H']}"]
        matches.append(dict(case=c['case_id'],cutoff=c['cutoff'],D=c['D'],H=c['H'],R=c['R'],
            G=g[:,i].tolist(),G_error=ge[:,i].tolist(),G_over_fixed428=(g[:,i]/g[:,oldi]).tolist(),
            G_over_fixed428_error=gain_error[:,new_indices.index(i)].tolist(),
            relative_residual=relative_residual[:,i].tolist(),residual_over_fixed428=(relative_residual[:,i]/relative_residual[:,oldi]).tolist(),
            max_abs_paired_z=np.max(abs(paired_z[:,:,i]),axis=1).tolist()))
    with (output/'scan.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['channel','case','D','H','R','cutoff','alpha','tail_multiplier','G','G_error','relative_residual'])
        for i,c in enumerate(cases):
            for ch,name in enumerate(channels):w.writerow([name,c['case_id'],c['D'],c['H'],c['R'],c.get('cutoff'),c.get('alpha'),c.get('tail_multiplier'),g[ch,i],ge[ch,i],relative_residual[ch,i]])
    with (output/'per_dt.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['channel','case','dt','mean_real','SE_real','error_ratio','ratio_error','paired_difference','paired_difference_SE','paired_difference_z'])
        for i,c in enumerate(cases):
            for ch,name in enumerate(channels):
                for ti,t in enumerate(times):w.writerow([name,c['case_id'],t,mean[ch,ti,i].real,se[ch,ti,i],ratio[ch,ti,i],re[ch,ti,i],paired_mean[ch,ti,i],paired_se[ch,ti,i],paired_z[ch,ti,i]])
    report=dict(**definition,configurations=list(cfgs),input_ids=ids,cases_count=len(cases),matched_comparisons=matches,
        postprocessing_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='Empirical 18-cfg point-pool study; C3 rare-link coverage limits optimization from SE alone. q=3 allocates three components per point; it does not change spatial-label inclusion probabilities.')
    (output/'analysis.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(configurations=len(cfgs),cases=len(cases),new_cases=len(matches),output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cfgs',nargs='+',default=CFGS);a=p.parse_args();run(a.input,a.output,a.cfgs)
