"""Plot only finite-production two-leg results, never complete-other-leg scans."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from lqcdweave.storage.dataset import DatasetReader
from tools.campaigns.single_current_vdp_plan import CFGS
from tools.campaigns.single_source_dh_analysis import error_statistics


def run(root,output,cfgs=CFGS):
    with DatasetReader(root/'correlators',cfg=cfgs[0]) as first:
        if first.spec.document['definition']['estimator']=='ne20-finite-VH-independent-point-repeats-v1':
            from tools.campaigns._ne20_point_repeat_plot import run as repeat_plot
            return repeat_plot(root,output,cfgs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True);arrays=[];ids=[];definition=None
    for cfg in cfgs:
        with DatasetReader(root/'correlators',cfg=cfg) as reader:
            actual=reader.spec.document['definition']
            if actual['estimator']!='ne20-fullfield-finite-two-legs-v1':
                raise ValueError('refusing complete-other-leg data in production acceptance')
            signature={k:actual[k] for k in ('estimator','source_time','dt','cases','parent_counts','normalization','elemental_gauge','operators','measurements','dirac','H_definitions','meson_modes','current_low_modes','meson_source_time','current_time')}
            if definition is None:definition=signature;cases=actual['cases'];times=actual['dt']
            elif signature!=definition:raise ValueError('incompatible ensemble science or case ordering')
            a=np.stack([reader.read(dt=t) for t in times])
            arrays.append(np.stack((a[:,0],a[:,1],a[:,2:5].mean(1))))
            ids.append(reader.data_id)
    data=np.stack(arrays);reference=next(i for i,c in enumerate(cases) if c['case_id']=='FULL')
    mean,se,ratio,g,ge,re=error_statistics(data,reference)
    residual=data.real-data.real[...,reference,None]
    residual_rms=np.sqrt(np.mean(residual**2,axis=(0,2)))
    scale=np.sqrt(np.mean(data.real[...,reference]**2,axis=(0,2)))
    relative_residual=residual_rms/scale[:,None]
    paired_mean=residual.mean(0);paired_se=residual.std(0,ddof=1)/np.sqrt(len(cfgs))
    paired_z=np.divide(paired_mean,paired_se,out=np.zeros_like(paired_mean),where=paired_se>0)
    np.savez_compressed(output/'ensemble.npz',data=data,mean=mean,standard_error=se,error_ratio=ratio,
                        G=g,G_error=ge,ratio_error=re,dt=times,residual_rms=residual_rms,relative_residual=relative_residual,
                        paired_mean_difference=paired_mean,paired_difference_SE=paired_se,paired_difference_z=paired_z)
    curves=[([i for i,c in enumerate(cases) if c['family']=='VD'],'V+D full complement','#2474b7','-'),
            ([i for i,c in enumerate(cases) if c['family']=='VH'],'V+H (Q100)','#e58a00','-')]
    colors={2:'#129b79',3:'#b051a1',4:'#ca4e49'}
    for family,style in [('original',':'),('i1','-'),('i2','--')]:
        for m in (2,3,4):
            label={'original':'original partition','i1':'i^-1','i2':'i^-2'}[family]
            curves.append(([i for i,c in enumerate(cases) if c['family']==family and c['m']==m],
                           f'V+D{10*m}+H, {label}',colors[m],style))
    fig,axs=plt.subplots(1,3,figsize=(21,7.5),layout='constrained');handles=[]
    for ch,ax in enumerate(axs):
        for indices,label,color,style in curves:
            x=[cases[i]['R'] for i in indices];y=g[ch,indices];err=ge[ch,indices]
            line,=ax.plot(x,y,label=label,color=color,ls=style,lw=1.5)
            ax.fill_between(x,np.maximum(y-err,1e-12),y+err,color=color,alpha=.045)
            if ch==0:handles.append(line)
        ax.set(title=('C2','C2p','C3')[ch],xlabel='Finite sampled components R = ND + 3 NH',
               ylabel='Error G / full-space reference',yscale='log');ax.grid(alpha=.2);ax.axhline(1,color='gray',ls='--')
    fig.suptitle(f'Finite production inputs: both current legs restricted | meson V20, current V100 | {len(cfgs)} cfg | meson t=0 | current dt=1..18\n'
                 'FULL reference separate; finite schemes restrict both legs; component-inclusion weights')
    fig.legend(handles,[c[1] for c in curves],loc='outside lower center',ncol=4,fontsize=10)
    fig.savefig(output/'all_schemes_dense.png',dpi=180);fig.savefig(output/'all_schemes_dense.pdf');plt.close(fig)
    # Main panel shows the requested H budgets; preserve the complete dense scan above.
    for mode in ('all_schemes','reference_residual'):
        fig,axs=plt.subplots(1,3,figsize=(20,7.6),layout='constrained');handles=[]
        for ch,ax in enumerate(axs):
            for indices,label,color,style in curves:
                indices=[i for i in indices if (cases[i]['family']=='VD' and cases[i]['D'] in (8,16,24,32,48,63,96)) or cases[i]['H'] in (8,16,32,64)]
                x=[cases[i]['R'] for i in indices]
                if mode=='all_schemes':
                    y=g[ch,indices];err=ge[ch,indices]
                    line=ax.errorbar(x,y,yerr=err,label=label,color=color,ls=style,marker='o',ms=3,lw=1.2,capsize=2)
                else:
                    line,=ax.plot(x,relative_residual[ch,indices],label=label,color=color,ls=style,marker='o',ms=3,lw=1.2)
                if ch==0:handles.append(line)
            ax.set(title=('C2','C2p','C3')[ch],xlabel='Finite components R = ND + 3 NH',yscale='log')
            ax.grid(alpha=.18)
            if mode=='all_schemes':
                ax.axhline(1,color='#444444',ls='--',lw=1.3)
                ax.set_ylabel('G = RMS_dt [SE / SE(FULL)]')
            else:ax.set_ylabel('RMS(C - FULL) / RMS(FULL), real part')
        fig.legend(handles,[c[1] for c in curves],loc='outside lower center',ncol=4,fontsize=9)
        fig.suptitle(f'Meson V20 at t=0; current V100 | {len(cfgs)} cfg | dt=1..18 | both sampled legs finite\n'
            +('Statistical error relative to full space (dashed G=1); H=8/16/32/64; dense scan supplied separately' if mode=='all_schemes' else
              'Paired deviation from full-space correlators; complementary diagnostic, not the statistical-error metric'))
        fig.savefig(output/(mode+'.png'),dpi=180);fig.savefig(output/(mode+'.pdf'));plt.close(fig)
    names=['FULL','VD-D63','VH-H21','VH-H64',*[f'VDH-{family}-m3-H11' for family in ('original','i1','i2')]]
    chosen=[next(i for i,c in enumerate(cases) if c['case_id']==name) for name in names]
    for mode in ('correlators','per_dt_error_ratio','mean_difference'):
        fig,axs=plt.subplots(1,3,figsize=(18,5.8),layout='constrained')
        for ch,ax in enumerate(axs):
            for j,i in enumerate(chosen):
                y=mean[ch,:,i].real if mode=='correlators' else paired_mean[ch,:,i] if mode=='mean_difference' else ratio[ch,:,i]
                err=se[ch,:,i] if mode=='correlators' else paired_se[ch,:,i] if mode=='mean_difference' else re[ch,:,i]
                ax.errorbar(np.asarray(times)+(j-3.5)*.06,y,yerr=err,label=names[j],lw=1.8 if names[j]=='FULL' else 1,
                            color='black' if names[j]=='FULL' else None,marker='.',capsize=2)
            ax.set(title=('C2','C2p','C3')[ch],xlabel='dt',ylabel='Re C(dt)' if mode=='correlators' else 'Mean Re(C - FULL)' if mode=='mean_difference' else 'SE / SE(FULL)')
            if mode=='per_dt_error_ratio':ax.set_yscale('log');ax.axhline(1,color='gray',ls='--')
            else:
                ax.set_yscale('symlog',linthresh=max(float(np.median(se[ch,:,reference])),1e-30))
                if mode=='mean_difference':ax.axhline(0,color='gray',ls='--')
            ax.grid(alpha=.2)
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
        fig.suptitle('Finite-production estimator | real part with cfg standard errors' if mode=='correlators' else
                     'Paired finite minus full-space mean, with cross-cfg standard errors | finite cases R=63 except VH64' if mode=='mean_difference' else
                     'Finite-production estimator | paired delete-one-cfg error-ratio uncertainty')
        fig.savefig(output/(mode+'.png'),dpi=180);plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(16,12),layout='constrained')
    for ch in range(3):
        for fi,family in enumerate(('original','i1','i2')):
            ax=axs[ch,fi]
            for m in (2,3,4):
                indices=[i for i,c in enumerate(cases) if c['family']==family and c['m']==m and c['H'] in (8,16,32,64)]
                ax.errorbar([cases[i]['H'] for i in indices],g[ch,indices],yerr=ge[ch,indices],marker='o',capsize=3,label=f'{m} columns/group')
            ax.set(title=f'{("C2","C2p","C3")[ch]} | {family}',xlabel='Full-color H points',ylabel='G / FULL',yscale='log',xticks=[8,16,32,64]);ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.savefig(output/'D_H_count_grid.png',dpi=180);plt.close(fig)
    with (output/'scan.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['channel','case','D','H','R','G','G_jackknife_error','residual_RMS','relative_residual_RMS'])
        for ch,name in enumerate(('C2','C2p','C3')):
            for i,c in enumerate(cases):w.writerow([name,c['case_id'],c['D'],c['H'],c['R'],g[ch,i],ge[ch,i],residual_rms[ch,i],relative_residual[ch,i]])
    with (output/'per_dt.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['channel','case','dt','mean_real','mean_imag','SE_real','error_ratio','ratio_jackknife_error','paired_mean_difference','paired_difference_SE','paired_difference_z'])
        for ch,name in enumerate(('C2','C2p','C3')):
            for i,c in enumerate(cases):
                for ti,dt in enumerate(times):w.writerow([name,c['case_id'],dt,mean[ch,ti,i].real,mean[ch,ti,i].imag,se[ch,ti,i],ratio[ch,ti,i],re[ch,ti,i],paired_mean[ch,ti,i],paired_se[ch,ti,i],paired_z[ch,ti,i]])
    report=dict(**definition,configurations=list(cfgs),input_ids=ids,reference='FULL',cases_count=len(cases),
        postprocessing_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        statistics='cfg real sample std(ddof=1)/sqrt(Ncfg); C3 directions averaged within cfg',
        metric='RMS dt pointwise standard error ratio; fixed-denominator G jackknife',
        limitations='One meson source t=0, current times 1..18, nested prefixes, 18 cfg; rare links can underestimate conditional C3 SE; full-reference residual RMS supplied separately')
    with (output/'C3_links.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['cfg','dt','H','actual_directed_links','expected_directed_links'])
        for cfg in cfgs:
            path=root/f'complete-{cfg}.json'
            if path.exists():
                for row in json.loads(path.read_text()).get('C3_point_point_links',[]):
                    w.writerow([cfg,row.get('dt'),row['H'],row['actual_directed_links'],row['expected_directed_links']])
    (output/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(cfgs=len(cfgs),cases=len(cases),output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cfgs',nargs='+',default=CFGS);a=p.parse_args()
    run(a.input,a.output,a.cfgs)
