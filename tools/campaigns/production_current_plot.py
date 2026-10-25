"""Plot only finite-production two-leg results, never complete-other-leg scans."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from lqcdweave.storage.dataset import DatasetReader
from tools.campaigns.single_current_vdp_plan import CFGS
from tools.campaigns.single_source_dh_analysis import error_statistics


def run(root,output,cfgs=CFGS):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True,exist_ok=True);arrays=[];ids=[];definition=None
    for cfg in cfgs:
        with DatasetReader(root/'correlators',cfg=cfg) as reader:
            actual=reader.spec.document['definition']
            if actual['estimator']!='two-finite-current-legs-component-inclusion-v1':
                raise ValueError('refusing complete-other-leg data in production acceptance')
            signature={k:actual[k] for k in ('estimator','source_time','dt','cases','parent_counts','normalization','elemental_gauge','operators','measurements','dirac','H_definitions')}
            if definition is None:definition=signature;cases=actual['cases'];times=actual['dt']
            elif signature!=definition:raise ValueError('incompatible ensemble science or case ordering')
            a=np.stack([reader.read(dt=t) for t in times])
            arrays.append(np.stack((a[:,0],a[:,1],a[:,2:5].mean(1))))
            ids.append(reader.data_id)
    data=np.stack(arrays);reference=next(i for i,c in enumerate(cases) if c['case_id']=='VH-H64')
    mean,se,ratio,g,ge,re=error_statistics(data,reference)
    np.savez_compressed(output/'ensemble.npz',data=data,mean=mean,standard_error=se,error_ratio=ratio,
                        G=g,G_error=ge,ratio_error=re,dt=times)
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
               ylabel='Error G / same-channel VH64',yscale='log');ax.grid(alpha=.2);ax.axhline(1,color='gray',ls='--')
    fig.suptitle(f'Finite production inputs: both current legs restricted | V100 | {len(cfgs)} cfg | tsrc=0 | dt=1..18\n'
                 'No operated-source solves or complete other leg; all schemes use component-inclusion weights')
    fig.legend(handles,[c[1] for c in curves],loc='outside lower center',ncol=4,fontsize=10)
    fig.savefig(output/'all_schemes.png',dpi=180);fig.savefig(output/'all_schemes.pdf');plt.close(fig)
    names=['VD-D63','VH-H21','VH-H64',*[f'VDH-{family}-m3-H16' for family in ('original','i1','i2')]]
    chosen=[next(i for i,c in enumerate(cases) if c['case_id']==name) for name in names]
    for mode in ('correlators','per_dt_error_ratio'):
        fig,axs=plt.subplots(1,3,figsize=(18,5.8),layout='constrained')
        for ch,ax in enumerate(axs):
            for j,i in enumerate(chosen):
                y=mean[ch,:,i].real if mode=='correlators' else ratio[ch,:,i]
                err=se[ch,:,i] if mode=='correlators' else re[ch,:,i]
                ax.errorbar(np.asarray(times)+(j-2.5)*.06,y,yerr=err,label=names[j],lw=1,marker='.',capsize=2)
            ax.set(title=('C2','C2p','C3')[ch],xlabel='dt',ylabel='Re C(dt)' if mode=='correlators' else 'SE / SE(VH64)')
            if mode!='correlators':ax.set_yscale('log');ax.axhline(1,color='gray',ls='--')
            ax.grid(alpha=.2)
        handles,labels=axs[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9)
        fig.suptitle('Finite-production estimator | real part with cfg standard errors' if mode=='correlators' else
                     'Finite-production estimator | paired delete-one-cfg error-ratio uncertainty')
        fig.savefig(output/(mode+'.png'),dpi=180);plt.close(fig)
    fig,axs=plt.subplots(3,3,figsize=(16,12),layout='constrained')
    for ch in range(3):
        for fi,family in enumerate(('original','i1','i2')):
            ax=axs[ch,fi]
            for m in (2,3,4):
                indices=[i for i,c in enumerate(cases) if c['family']==family and c['m']==m and c['H'] in (8,16,32,64)]
                ax.errorbar([cases[i]['H'] for i in indices],g[ch,indices],yerr=ge[ch,indices],marker='o',capsize=3,label=f'{m} columns/group')
            ax.set(title=f'{("C2","C2p","C3")[ch]} | {family}',xlabel='Full-color H points',ylabel='G / VH64',yscale='log',xticks=[8,16,32,64]);ax.grid(alpha=.2)
    handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.savefig(output/'D_H_count_grid.png',dpi=180);plt.close(fig)
    with (output/'scan.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['channel','case','D','H','R','G','G_jackknife_error'])
        for ch,name in enumerate(('C2','C2p','C3')):
            for i,c in enumerate(cases):w.writerow([name,c['case_id'],c['D'],c['H'],c['R'],g[ch,i],ge[ch,i]])
    with (output/'per_dt.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['channel','case','dt','mean_real','mean_imag','SE_real','error_ratio','ratio_jackknife_error'])
        for ch,name in enumerate(('C2','C2p','C3')):
            for i,c in enumerate(cases):
                for ti,dt in enumerate(times):w.writerow([name,c['case_id'],dt,mean[ch,ti,i].real,mean[ch,ti,i].imag,se[ch,ti,i],ratio[ch,ti,i],re[ch,ti,i]])
    report=dict(**definition,configurations=list(cfgs),input_ids=ids,reference='VH-H64',cases_count=len(cases),
        statistics='cfg real sample std(ddof=1)/sqrt(Ncfg); C3 directions averaged within cfg',
        metric='RMS dt pointwise standard error ratio; fixed-denominator G jackknife',
        limitations='One source, nested prefixes, 18 cfg; rare point-link events can make C3 conditional error estimates unstable')
    with (output/'C3_links.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['cfg','H','actual_directed_links','expected_directed_links'])
        for cfg in cfgs:
            path=root/f'complete-{cfg}.json'
            if path.exists():
                for row in json.loads(path.read_text()).get('C3_point_point_links',[]):
                    w.writerow([cfg,row['H'],row['actual_directed_links'],row['expected_directed_links']])
    (output/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(cfgs=len(cfgs),cases=len(cases),output=str(output))))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cfgs',nargs='+',default=CFGS);a=p.parse_args()
    run(a.input,a.output,a.cfgs)
