"""Label-free within-record windows and exploratory paired signal diagnostics."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path,write_json,file_hash
from ..data.features import SENSORS,STATISTICS
from ..data.splits import POOLS
from .inspect_psd_features import features as psd_features,BANDS
from .diagnose_relative_features import gap_between

ROOT=project_path('wear_monitoring_dynamics/outputs/signal_windows_20261012')
FS=250;ACTIVE=sorted(POOLS['fold_a']+POOLS['fold_b'])


def plateau_window(dc,fraction=.8):
    """Longest high-load DC interval, 1s guard at both ends, central fixed8s.

    Candidate only: no claim that high current alone establishes steady cutting.
    Window uses the completed current record, never VB or another record.
    """
    x=np.asarray(dc,float).ravel();n=len(x)//FS;b=x[:n*FS].reshape(n,FS).mean(1)
    lo,hi=np.quantile(b,[.1,.9]);mask=b>=lo+fraction*(hi-lo)
    starts=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))==1)
    ends=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(int))==-1)
    if len(starts)==0 or hi-lo<1e-8:return None
    j=np.argmax(ends-starts);left=(starts[j]+1)*FS;right=(ends[j]-1)*FS
    if right-left<8*FS:return None
    mid=(left+right)//2
    return dict(start=int(mid-4*FS),end=int(mid+4*FS),plateau_start=int(left),plateau_end=int(right),
                low=float(lo),high=float(hi),threshold=float(lo+fraction*(hi-lo)),high_fraction=float(mask.mean()))


def extract(x):
    x=np.asarray(x,float).ravel();mu=x.mean();sd=x.std();z=(x-mu)/sd if sd>1e-12 else np.zeros_like(x)
    values=dict(zip(STATISTICS,[mu,sd,np.sqrt(np.mean(x*x)),x.min(),x.max(),np.ptp(x),np.mean(z**3),np.mean(z**4),np.mean(x*x)]))
    if sd<1e-12:
        p=dict(ac_power=0.,centroid_hz=np.nan,bandwidth_hz=np.nan,entropy=np.nan,flatness=np.nan,peak_hz=np.nan)
        for lo,hi in BANDS:p.update({f'band_{lo}_{hi}_fraction':np.nan,f'band_{lo}_{hi}_power':0.})
    else:_,_,p=psd_features(x)
    values.update({'psd_'+k:v for k,v in p.items()});return values


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    records=sorted(np.load(project_path('mill_numpy/mill.npy'),allow_pickle=True).tolist(),key=lambda r:(r['case'],r['run']))
    qc=pd.read_csv(project_path('wear_monitoring_dynamics/outputs/psd_inspection_20261010/quality_records.csv')).set_index(['case','run'])
    write_json(ROOT/'protocol.json',dict(source_sha256=file_hash(project_path('mill_numpy/mill.npy')),script_sha256=file_hash(__file__),fs=250,
        primary_channels=['smcAC','smcDC'],all_channels=list(SENSORS),windows=['full','center50','center8','plateau8'],
        selection='DC 1s means; q10 + .8*(q90-q10); longest contiguous high interval; remove1s at each edge; center8s; no VB or across-record fitting',
        sensitivity_thresholds=[.7,.8,.9],minimum_window_samples=2000,missing='no fallback if plateau too short; paired comparisons drop same records from every window',
        interpretation='Within-record complete-signal diagnostic, not sample-online segmentation; high-load candidate is not verified steady cutting. No model training or feature selection.',
        gap='At measured-overlap VB; linear interpolation only for diagnostic integration; full-window within-case SD from each training pool used for ALL window variants; no predictive accuracy claim'))
    rows=[];blocks=[];windows=[];sensitivity=[];lookup={};reproduce=[];ceiling=[]
    mat=loadmat(project_path('mill/mill.mat'))['mill'].ravel()
    raw={(int(m['case'].item()),int(m['run'].item())):m for m in mat}
    old=pd.read_csv(project_path('wear_monitoring_dynamics/outputs/psd_inspection_20261010/psd_features.csv')).set_index(['case','run','channel'])
    for r in records:
        case,run=int(r['case']),int(r['run']);valid=bool(qc.loc[(case,run),'sensor_valid'])
        if not valid:continue
        n=len(np.asarray(r['smcDC']).ravel());mid=n//2;p=plateau_window(r['smcDC'])
        spans=dict(full=(0,n),center50=(n//4,3*n//4),center8=(mid-1000,mid+1000))
        if p:spans['plateau8']=(p['start'],p['end'])
        lookup[case,run]=(r,spans,p)
        for threshold in [.7,.8,.9]:
            pp=plateau_window(r['smcDC'],threshold)
            sensitivity.append(dict(case=case,run=run,threshold_fraction=threshold,available=pp is not None,start=np.nan if pp is None else pp['start']/FS,end=np.nan if pp is None else pp['end']/FS))
        windows.append(dict(case=case,run=run,VB=r['VB'],feed=r['feed'],DOC=r['DOC'],material=r['material'],n=n,
            available=p is not None,**({k:v for k,v in p.items()} if p else {})))
        for ch in SENSORS:
            x=np.asarray(r[ch],float).ravel();bb=x[:len(x)//FS*FS].reshape(-1,FS)
            np.testing.assert_array_equal(x,raw[case,run][ch].ravel())
            if ch=='smcDC':
                at=x==9.9951171875;edges=np.diff(np.r_[False,at,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
                selected=x[spans['plateau8'][0]:spans['plateau8'][1]] if p else np.array([np.nan])
                ceiling.append(dict(case=case,run=run,VB=r['VB'],ceiling_fraction=at.mean(),longest_ceiling_seconds=float(max(ends-starts,default=0)/FS),
                    plateau_ceiling_fraction=float((selected==9.9951171875).mean()) if p else np.nan,plateau_std=float(selected.std()),
                    first2s_mean=float(x[:500].mean()),last2s_mean=float(x[-500:].mean()),plateau_mean=float(selected.mean())))
            for j,b in enumerate(bb):blocks.append(dict(case=case,run=run,channel=ch,second=j+.5,mean=b.mean(),std=b.std(),rms=np.sqrt(np.mean(b*b))))
            for mode,(a,b) in spans.items():
                vals=extract(x[a:b]);rows.append(dict(case=case,run=run,VB=r['VB'],DOC=r['DOC'],feed=r['feed'],material=r['material'],channel=ch,window=mode,n=b-a,**vals))
                if mode=='full':reproduce.append(max(abs(vals['psd_'+f]-float(old.loc[(case,run,ch),f])) for f in ['ac_power','band_0_5_fraction','band_5_15_fraction']))
        if run==2:print(f'Case {case} processed',flush=True)
    df=pd.DataFrame(rows);block=pd.DataFrame(blocks);win=pd.DataFrame(windows);sens=pd.DataFrame(sensitivity)
    df.to_csv(ROOT/'window_features.csv',index=False);block.to_csv(ROOT/'one_second_profiles.csv',index=False);win.to_csv(ROOT/'windows.csv',index=False);sens.to_csv(ROOT/'threshold_sensitivity.csv',index=False)
    pd.DataFrame(ceiling).to_csv(ROOT/'dc_ceiling_audit.csv',index=False)
    # Same cases/records in every window comparison.
    paired=win[win.available&win.case.isin(ACTIVE)&win.VB.notna()][['case','run']]
    sub=df.merge(paired,on=['case','run']);featurecols=list(extract(np.arange(2000,dtype=float)))
    meta=win[win.case.isin(ACTIVE)][['case','DOC','feed','material']].drop_duplicates()
    pairs=[tuple(sorted(g.case.tolist())) for _,g in meta.groupby(['DOC','feed','material'])];assert all(len(p)==2 for p in pairs)
    gaps=[];rhos=[]
    for ch in ['smcAC','smcDC']:
        channel=sub[sub.channel==ch]
        for f in featurecols:
            support=channel.pivot(index=['case','run'],columns='window',values=f).dropna().reset_index()[['case','run']]
            comparable=channel.merge(support,on=['case','run'])
            for (case,mode),g in channel.groupby(['case','window']):
                rhos.append(dict(channel=ch,feature=f,case=case,window=mode,rho=spearmanr(g.VB,g[f]).statistic if len(g)>2 and g[f].nunique()>1 else np.nan,sd=g[f].std(ddof=0)))
            for fold,train in POOLS.items():
                base=comparable[(comparable.window=='full')&comparable.case.isin(train)]
                scale=np.sqrt(base.groupby('case')[f].var(ddof=0).mean())
                if scale<1e-12:continue
                for ca,cb in pairs:
                    for mode in ['full','center50','center8','plateau8']:
                        aa=comparable[(comparable.case==ca)&(comparable.window==mode)];bb=comparable[(comparable.case==cb)&(comparable.window==mode)]
                        gap=gap_between(aa.VB,aa[f],bb.VB,bb[f])
                        if gap:gaps.append(dict(fold=fold,channel=ch,feature=f,pair=f'{ca}-{cb}',window=mode,scaled_gap=gap['gap']/scale,**gap))
    gaps=pd.DataFrame(gaps);gaps.to_csv(ROOT/'matched_vb_gaps.csv',index=False);rhos=pd.DataFrame(rhos);rhos.to_csv(ROOT/'within_case_relations.csv',index=False)
    keys=['fold','channel','feature','pair'];wide=gaps.pivot(index=keys,columns='window',values='scaled_gap').reset_index();wide.to_csv(ROOT/'paired_gaps.csv',index=False)
    summary=[]
    for ch in ['smcAC','smcDC']:
        q=wide[wide.channel==ch]
        for mode in ['center50','center8','plateau8']:
            summary.append(dict(channel=ch,window=mode,mean_full_gap=q.full.mean(),mean_new_gap=q[mode].mean(),gap_ratio=q[mode].mean()/q.full.mean(),improved_fraction=float((q[mode]<q.full).mean())))
    pd.DataFrame(summary).to_csv(ROOT/'gap_summary.csv',index=False)
    # Exact log decomposition over adjacent available runs, no VB required.
    decomposition=[]
    for (ch,mode,case),g in df[df.case.isin(ACTIVE)&df.channel.isin(['smcAC','smcDC'])].groupby(['channel','window','case']):
        g=g.sort_values('run')
        for lo,hi in BANDS:
            band=g[f'psd_band_{lo}_{hi}_power'].to_numpy();total=g.psd_ac_power.to_numpy();fraction=g[f'psd_band_{lo}_{hi}_fraction'].to_numpy()
            ok=(band>0)&(total>0)&np.isfinite(fraction)&(fraction>0)
            with np.errstate(divide='ignore',invalid='ignore'):
                db=np.diff(np.log(band));dt=np.diff(np.log(total));dr=np.diff(np.log(fraction))
            good=ok[:-1]&ok[1:];np.testing.assert_allclose(dr[good],db[good]-dt[good],atol=2e-12)
            for i in range(len(dr)):
                if not good[i]:continue
                decomposition.append(dict(channel=ch,window=mode,case=case,run_from=int(g.run.iloc[i]),run_to=int(g.run.iloc[i+1]),band=f'{lo}-{hi}',
                    log_band_change=db[i],log_total_change=dt[i],log_fraction_change=dr[i],opposite_direction=bool(db[i]*dr[i]<-1e-12),denominator_larger=bool(abs(dt[i])>abs(db[i]))))
    dec=pd.DataFrame(decomposition);dec.to_csv(ROOT/'power_decomposition.csv',index=False)
    dec.groupby(['channel','window','band']).agg(opposite_fraction=('opposite_direction','mean'),denominator_larger_fraction=('denominator_larger','mean'),n=('case','size')).to_csv(ROOT/'power_decomposition_summary.csv')
    # Window effects, quantities paired to full window, not target-selected.
    effect=df.pivot(index=['case','run','channel'],columns='window',values=['psd_ac_power','psd_band_0_5_power','rms','mean'])
    effect.columns=['_'.join(c) for c in effect.columns];effect=effect.reset_index()
    for f in ['psd_ac_power','psd_band_0_5_power','rms']:
        effect[f+'_plateau_ratio']=effect[f+'_plateau8']/effect[f+'_full']
    effect.to_csv(ROOT/'window_effects.csv',index=False)
    ceil=pd.DataFrame(ceiling)
    write_json(ROOT/'readout.json',dict(dc_common_ceiling=9.9951171875,records_any_ceiling=int((ceil.ceiling_fraction>0).sum()),
        records_atleast1pct_ceiling=int((ceil.ceiling_fraction>=.01).sum()),constant_dc_plateau_windows=int((ceil.plateau_std<1e-12).sum()),
        active_records_atleast1pct_ceiling=int(((ceil.ceiling_fraction>=.01)&ceil.case.isin(ACTIVE)).sum()),
        active_constant_dc_plateau_windows=int(((ceil.plateau_std<1e-12)&ceil.case.isin(ACTIVE)).sum()),
        plateau_to_full_power_median=effect.groupby('channel').psd_ac_power_plateau_ratio.median().to_dict(),
        plateau_to_full_lowband_median=effect.groupby('channel').psd_band_0_5_power_plateau_ratio.median().to_dict(),
        gap_median_ratios={ch:float((wide.loc[wide.channel==ch,'plateau8']/wide.loc[wide.channel==ch,'full']).median()) for ch in ['smcAC','smcDC']},
        note='Descriptive pooled-record/window statistics; not confidence bounds, train-only feature choices, or prediction metrics.'))
    # Profile overview: first/middle/last valid runs, no wear-based selection.
    fig,axs=plt.subplots(4,4,figsize=(16,12),layout='constrained')
    for case,ax in enumerate(axs.ravel(),1):
        runs=sorted(r for c,r in lookup if c==case);chosen=sorted(set([runs[0],runs[len(runs)//2],runs[-1]]))
        for run in chosen:
            g=block[(block.case==case)&(block.run==run)&(block.channel=='smcDC')];line=ax.plot(g.second,g['mean'],label=f'run {run}')[0]
            p=lookup[case,run][2]
            if p:ax.axvspan(p['start']/FS,p['end']/FS,color=line.get_color(),alpha=.08)
        ax.set(title=f'Case {case}',xlabel='Seconds within record',ylabel='Stored DC current');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(ROOT/'dc_profiles.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,(case,run) in zip(axs.ravel(),[(3,2),(11,2),(4,2),(10,2),(1,2),(9,2)]):
        g=block[(block.case==case)&(block.run==run)];p=lookup[case,run][2]
        for ch in SENSORS:
            b=g[g.channel==ch];v=b['mean'].to_numpy() if ch!='smcAC' else b.rms.to_numpy();lo,hi=np.quantile(v,[.1,.9]);ax.plot(b.second,(v-lo)/max(hi-lo,1e-9),label=ch)
        if p:ax.axvspan(p['start']/FS,p['end']/FS,color='grey',alpha=.2)
        ax.set(title=f'Case {case}, run {run}',xlabel='Seconds within record',ylabel='Within-record scaled profile');ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7);fig.savefig(ROOT/'six_channel_profiles.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,(pair,ch,f) in zip(axs.ravel(),[((3,11),'smcAC','psd_band_0_5_fraction'),((3,11),'smcAC','psd_band_0_5_power'),((3,11),'smcAC','psd_ac_power'),((4,10),'smcDC','mean'),((4,10),'smcDC','psd_ac_power'),((1,9),'smcAC','rms')]):
        for case,color in zip(pair,['tab:blue','tab:orange']):
            for mode,ls in [('full','--'),('plateau8','-')]:
                g=sub[(sub.channel==ch)&(sub.case==case)&(sub.window==mode)].sort_values('VB');ax.plot(g.VB,g[f],ls,marker='.',color=color,label=f'{case} {mode}')
        ax.set(title=ch+' '+f,xlabel='Measured VB (mm)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(ROOT/'feature_comparisons.png',dpi=150);plt.close(fig)
    write_json(ROOT/'audit.json',dict(passed=True,total_records=len(records),valid_records=len(win),plateau_available=int(win.available.sum()),
        active_paired_labeled_records=len(paired),full_psd_reproduction_max_error=float(max(reproduce)),
        full_psd_reproduced=bool(max(reproduce)<1e-10),mat_npy_all_valid_signals_identical=True,log_fraction_identity_verified=True,no_model_training=True))
    print(pd.DataFrame(summary).to_string(index=False));print(json.loads((ROOT/'audit.json').read_text()))
    print(effect.groupby('channel')[[c for c in effect if c.endswith('plateau_ratio')]].median().to_string())


if __name__=='__main__':main()
