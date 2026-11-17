"""Exploratory current morphology consistency, preserving DC rail information."""
import json
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter,sosfiltfilt,welch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path,write_json,file_hash
from ..data.current_processing import FS,DC_RAIL,QUARANTINE,spans
from ..data.splits import POOLS
from .diagnose_relative_features import gap_between

ROOT=project_path('wear_monitoring_dynamics/outputs/current_features_20261013')
ACTIVE=sorted(POOLS['fold_a']+POOLS['fold_b'])


def envelopes(ac,dc):
    rms=np.sqrt(np.maximum(uniform_filter1d(np.asarray(ac,float)**2,51,mode='reflect'),0))
    # For segmentation only, do not introduce Butterworth overshoot into thresholds.
    dcmean=uniform_filter1d(np.asarray(dc,float),51,mode='reflect')
    return rms,dcmean


def normalize_profile(y):
    low,high=np.quantile(y,[.05,.9]);return (y-low)/max(high-low,1e-9),float(low),float(high)


def interruptions(ac,dc):
    """Common interior drop below25% with high-load shoulders above80%.

    Each core must last >=80ms; surrounding high shoulders within8s. Expand to
    the containing joint-low80% span. This is a morphology exclusion, not a
    diagnosis of electrical failure. No labels or case/run-specific overrides.
    """
    a,d=envelopes(ac,dc);za,_,_=normalize_profile(a);zd,_,_=normalize_profile(d)
    high=(za>.8)&(zd>.8);low=(za<.25)&(zd<.25);broad=(za<.8)&(zd<.8)
    out=[]
    for s,e in spans(low):
        if e-s<20:continue
        pre=np.flatnonzero(high[max(0,s-2000):s]);post=np.flatnonzero(high[e:min(len(a),e+2000)])
        if len(pre)<25 or len(post)<25:continue
        bounds=next(((x,y) for x,y in spans(broad) if x<=s and y>=e),None)
        if bounds and bounds not in out:out.append(bounds)
    return out


def describe_profile(y,prefix):
    z,baseline,high=normalize_profile(y);amplitude=high-baseline;n=len(y)
    active=z>=.5;plateau=z>=.8;v=y[plateau];duration=n/FS
    f={prefix+'_baseline_q05':baseline,prefix+'_p90':high,prefix+'_p95':float(np.quantile(y,.95)),
       prefix+'_median':float(np.median(y)),prefix+'_excess_p90':amplitude,
       prefix+'_plateau_median':float(np.median(v)) if len(v) else np.nan,
       prefix+'_plateau_iqr':float(np.quantile(v,.75)-np.quantile(v,.25)) if len(v) else np.nan,
       prefix+'_plateau_cv':float(np.std(v)/max(abs(np.mean(v)),1e-9)) if len(v) else np.nan,
       prefix+'_area_excess':float(np.maximum(y-baseline,0).sum()/FS),
       prefix+'_area_normalized_seconds':float(np.maximum(z,0).sum()/FS)}
    for th,tag in [(.2,'20'),(.5,'50'),(.8,'80')]:
        mask=z>=th;ss=spans(mask)
        f[prefix+'_width'+tag+'_seconds']=float(mask.sum()/FS)
        f[prefix+'_longest'+tag+'_seconds']=float(max((b-a for a,b in ss),default=0)/FS)
    ids=np.flatnonzero(active);first=int(ids[0]) if len(ids) else 0;last=int(ids[-1]) if len(ids) else n-1
    f[prefix+'_onset50_seconds']=first/FS;f[prefix+'_end50_seconds']=last/FS
    # Crossings must be observed; no fabricated exit when recording ends loaded.
    highs=np.flatnonzero(z>=.9);start90=int(highs[0]) if len(highs) else None
    low_before=np.flatnonzero(z[:start90]<.1) if start90 is not None else np.array([])
    f[prefix+'_rise10_90_seconds']=(start90-int(low_before[-1]))/FS if len(low_before) else np.nan
    end90=int(highs[-1]) if len(highs) else None
    low_after=np.flatnonzero(z[end90+1:]<.1) if end90 is not None else np.array([])
    f[prefix+'_fall90_10_seconds']=(int(low_after[0])+1)/FS if len(low_after) else np.nan
    if len(v)>2:
        t=np.flatnonzero(plateau)/FS;f[prefix+'_plateau_slope']=float(np.polyfit(t,v,1)[0])
    else:f[prefix+'_plateau_slope']=np.nan
    return f


def extract(ac,dc):
    a,d=envelopes(ac,dc);f={**describe_profile(a,'ac_env'),**describe_profile(d,'dc_level')}
    rail=np.asarray(dc)>=DC_RAIL-1e-9;ss=spans(rail)
    f.update(dc_saturation_fraction=float(rail.mean()),dc_saturation_seconds=float(rail.sum()/FS),
        dc_saturation_longest_seconds=float(max((b-a for a,b in ss),default=0)/FS),
        dc_saturation_first_seconds=float(np.flatnonzero(rail)[0]/FS) if rail.any() else np.nan,
        ac_wave_rms=float(np.sqrt(np.mean(np.asarray(ac)**2))),dc_wave_mean=float(np.mean(dc)))
    # AC spectrum remains useful even when DC is capped. Absolute and relative values coexist.
    freq,p=welch(ac,fs=FS,nperseg=1024,noverlap=512);df=freq[1]-freq[0];total=p.sum()*df
    f['ac_total_spectral_power']=float(total)
    for lo,hi in [(0,5),(5,15),(15,30),(30,60),(60,125)]:
        mask=(freq>=lo)&((freq<hi) if hi<125 else (freq<=hi));power=p[mask].sum()*df
        f[f'ac_band_{lo}_{hi}_power']=float(power);f[f'ac_band_{lo}_{hi}_fraction']=float(power/total)
    f['ac_dc_plateau_ratio']=f['ac_env_plateau_median']/max(f['dc_level_plateau_median'],1e-9)
    f['ac_dc_rise_difference']=f['ac_env_rise10_90_seconds']-f['dc_level_rise10_90_seconds']
    return f


def relations(df,features,cases):
    rows=[]
    for feature in features:
        for case in cases:
            q=df[df.case==case].dropna(subset=['VB',feature]);rho=np.nan
            if len(q)>=4 and q[feature].nunique()>2 and q.VB.nunique()>2:rho=float(spearmanr(q.VB,q[feature]).statistic)
            rows.append(dict(feature=feature,case=case,n=len(q),rho=rho,sd=float(q[feature].std(ddof=0)) if len(q) else np.nan))
    return pd.DataFrame(rows)


def main():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'arrays').mkdir(exist_ok=True)
    source=project_path('mill_numpy/mill.npy');original_hash=file_hash(source)
    records=sorted(np.load(source,allow_pickle=True).tolist(),key=lambda r:(int(r['case']),int(r['run'])))
    write_json(ROOT/'protocol.json',dict(source_sha256=original_hash,script_sha256=file_hash(__file__),fs=FS,
        retain_saturation=True,interruption='AC RMS51 and DC mean51 normalized by own q05/q90; both below25% >=80ms, >=25 high samples (>80%) before and after within8s; containing joint-low80% interval reported',
        exclusion='Existing case2/run1 and case12/run1; any interior interruption excludes complete run from feature study, retains metadata and plots. No VB/case-specific condition for new rule.',
        features='Envelope levels, excess/area, relative-threshold widths20/50/80, observed rise/fall, plateau variation/slope; DC rail amount/timing; AC spectral absolute/relative power',
        ranking='Within-case rho, paired same-condition sign agreement, matched-VB normalized gap, missingness; train-pool-only top10 shortlist transfer in each direction; exploratory same previously viewed dataset',
        widths='Observed window duration; right/left censoring is retained and fall10% is missing if unobserved; relative widths depend on envelope range and may reflect saturation',
        constraints='No initial VB, label interpolation for training, model training, or performance claim; all-run feature plots exploratory; accepted raw currents unchanged'))
    rows=[];quality=[];lookup={};fcols=None
    for r in records:
        case,run=int(r['case']),int(r['run']);ac=np.asarray(r['smcAC'],float).ravel();dc=np.asarray(r['smcDC'],float).ravel()
        reason=QUARANTINE.get((case,run),'');dips=[] if reason else interruptions(ac,dc)
        if dips:reason='interior_joint_current_drop'
        accepted=not reason
        a,d=envelopes(ac,dc) if not (case,run) in QUARANTINE else (np.full(len(ac),np.nan),np.full(len(ac),np.nan))
        quality.append(dict(case=case,run=run,time=r['time'],VB=r['VB'],accepted=accepted,reason=reason,
            intervals_seconds=json.dumps([[s/FS,e/FS] for s,e in dips]),saturation_fraction=float((dc>=DC_RAIL-1e-9).mean()) if (case,run) not in QUARANTINE else np.nan,
            end_still_loaded=bool(normalize_profile(d)[0][-1]>.5) if (case,run) not in QUARANTINE else False))
        lookup[case,run]=(r,a,d,dips,accepted)
        if accepted:
            vals=extract(ac,dc);fcols=list(vals);rows.append(dict(case=case,run=run,VB=r['VB'],time=r['time'],DOC=r['DOC'],feed=r['feed'],material=r['material'],**vals))
    qc=pd.DataFrame(quality);qc.to_csv(ROOT/'quality_decisions.csv',index=False)
    data=pd.DataFrame(rows);data.to_csv(ROOT/'candidate_features.csv',index=False)
    rel=relations(data,fcols,ACTIVE);rel.to_csv(ROOT/'case_correlations.csv',index=False)
    meta=data[data.case.isin(ACTIVE)][['case','DOC','feed','material']].drop_duplicates()
    pairs=[tuple(sorted(g.case.tolist())) for _,g in meta.groupby(['DOC','feed','material'])];assert len(pairs)==7 and all(len(p)==2 for p in pairs)
    pairrows=[];stats=[]
    for feature in fcols:
        rr=rel[rel.feature==feature];rhos=rr.set_index('case').rho
        for a,b in pairs:
            aa=data[data.case==a].dropna(subset=['VB',feature]);bb=data[data.case==b].dropna(subset=['VB',feature])
            gap=gap_between(aa.VB,aa[feature],bb.VB,bb[feature]);scale=np.sqrt(np.mean([aa[feature].var(ddof=0),bb[feature].var(ddof=0)]))
            defined=np.isfinite(rhos[a]) and np.isfinite(rhos[b]);agree=bool(rhos[a]*rhos[b]>0) if defined else np.nan
            pairrows.append(dict(feature=feature,pair=f'{a}-{b}',rho_a=rhos[a],rho_b=rhos[b],both_defined=defined,same_direction=agree,
                paired_strength=min(abs(rhos[a]),abs(rhos[b])) if defined else np.nan,
                scaled_gap=gap['gap']/scale if gap and scale>1e-9 else np.nan))
        pr=pd.DataFrame(pairrows);pr=pr[pr.feature==feature];valid=rr.rho.dropna();direction=1 if (valid>0).sum()>=(valid<0).sum() else -1
        active=data[data.case.isin(ACTIVE)&data.VB.notna()]
        stats.append(dict(feature=feature,coverage=float(active[feature].notna().mean()),defined_cases=len(valid),
            median_abs_rho=float(valid.abs().median()) if len(valid) else np.nan,
            dominant_direction=direction,direction_cases=int((valid*direction>0).sum()),
            paired_defined=int(pr.both_defined.sum()),paired_same=int((pr.same_direction==True).sum()),
            median_scaled_gap=float(pr.scaled_gap.median()),median_pair_strength=float(pr.paired_strength.median())))
    pairdf=pd.DataFrame(pairrows);pairdf.to_csv(ROOT/'same_condition_pairs.csv',index=False)
    stats=pd.DataFrame(stats).sort_values(['paired_same','median_abs_rho'],ascending=False);stats.to_csv(ROOT/'consistency_summary.csv',index=False)
    # Freeze directional shortlists from training-side rho only, before evaluating other pool.
    choices=[]
    for fold,train in POOLS.items():
        for feature in fcols:
            q=rel[(rel.feature==feature)&rel.case.isin(train)];v=q.rho.dropna();direction=1 if (v>0).sum()>=(v<0).sum() else -1
            strength=float(np.median(v*direction)) if len(v) else np.nan
            eligible=len(v)>=5 and len(data[data.case.isin(train)&data.VB.notna()].dropna(subset=[feature]))/len(data[data.case.isin(train)&data.VB.notna()])>=.8
            choices.append(dict(fold=fold,feature=feature,train_defined=len(v),direction=direction,train_score=strength,eligible=eligible))
    ranks=pd.DataFrame(choices);selected=pd.concat([g[g.eligible].sort_values(['train_score','feature'],ascending=[False,True]).head(10) for _,g in ranks.groupby('fold')])
    write_json(ROOT/'train_only_shortlists.json',selected.to_dict('records'))
    transfer=[]
    for row in selected.itertuples():
        other=[c for c in ACTIVE if c not in POOLS[row.fold]];v=rel[(rel.feature==row.feature)&rel.case.isin(other)].rho.dropna()
        transfer.append(dict(fold=row.fold,feature=row.feature,train_score=row.train_score,test_defined=len(v),test_direction_cases=int((v*row.direction>0).sum()),test_median_signed_rho=float(np.median(v*row.direction))))
    pd.DataFrame(transfer).to_csv(ROOT/'shortlist_transfer.csv',index=False)
    panels=[]
    plt.rcParams.update({'path.simplify':False,'agg.path.chunksize':20000,'font.size':9})
    for case in range(1,17):
        group=[v for (c,_),v in lookup.items() if c==case];edges=np.r_[0,np.cumsum([len(a)/FS for _,a,_,_,_ in group])]
        fig,axs=plt.subplots(2,1,figsize=(max(18,.85*len(group)),7),sharex=True);fig.subplots_adjust(left=.06,right=.99,bottom=.09,top=.84,hspace=.12)
        archive={}
        for j,(r,a,d,dips,accepted) in enumerate(group):
            run=int(r['run']);x=edges[j]+np.arange(len(a))/FS;ac=np.asarray(r['smcAC']).ravel();dc=np.asarray(r['smcDC']).ravel()
            archive[f'run{run}_ac']=ac.copy() if accepted else np.full(len(a),np.nan)
            archive[f'run{run}_dc']=dc.copy() if accepted else np.full(len(a),np.nan)
            archive[f'run{run}_rail_mask']=(dc>=DC_RAIL-1e-9)&accepted
            if accepted:
                np.testing.assert_array_equal(archive[f'run{run}_ac'],ac);np.testing.assert_array_equal(archive[f'run{run}_dc'],dc)
                axs[0].plot(x,a,color='#2779ab',lw=.6);axs[1].plot(x,dc,color='#d47c20',lw=.35)
                axs[1].plot(x,np.where(dc>=DC_RAIL-1e-9,dc,np.nan),color='#c12c31',lw=1.2)
            else:
                for ax in axs:ax.axvspan(edges[j],edges[j+1],color='grey',alpha=.18)
                if dips:
                    axs[0].plot(x,a,color='#aaa',lw=.5);axs[1].plot(x,dc,color='#aaa',lw=.5)
                    for s,e in dips:
                        for ax in axs:ax.axvspan(edges[j]+s/FS,edges[j]+e/FS,color='red',alpha=.15)
            for ax in axs:ax.axvline(edges[j],color='#888',ls='--',lw=.4)
        for ax in axs:ax.grid(axis='y',alpha=.2);ax.set_xlim(0,edges[-1])
        axs[0].set_ylabel('AC local RMS (stored units)');axs[1].set_ylabel('DC recorded level (rail retained)');axs[1].set_xlabel('Concatenated sample time (seconds); run gaps not represented')
        top=axs[0].secondary_xaxis('top');top.set_xticks((edges[:-1]+edges[1:])/2);top.set_xticklabels([str(int(r['run']))+('x' if not ok else '') for r,_,_,_,ok in group]);top.set_xlabel('Run (x = excluded from feature study)')
        fig.suptitle(f'Case {case} | V2: DC saturation retained as observation + features\nGrey: excluded record; pink: detected internal interruption. No interpolation/reconstruction.',fontsize=13)
        fig.savefig(ROOT/f'case{case:02d}.png',dpi=140);plt.close(fig)
        np.savez_compressed(ROOT/'arrays'/f'case{case:02d}.npz',**archive)
        with np.load(ROOT/'arrays'/f'case{case:02d}.npz') as saved:
            for k,v in archive.items():np.testing.assert_array_equal(saved[k],v)
        panels.append(f'<section id="case{case}"><h2>Case {case}</h2><a href="case{case:02d}.png" target="_blank">打开高清图</a><img loading="lazy" src="case{case:02d}.png"></section>')
    # Display predetermined amplitude, width, saturation and frequency candidates, not only winners.
    shown=['ac_env_plateau_median','ac_env_excess_p90','ac_env_area_excess','ac_env_width50_seconds','ac_env_width80_seconds','ac_env_rise10_90_seconds',
        'dc_level_plateau_median','dc_level_width50_seconds','dc_level_rise10_90_seconds','dc_saturation_fraction','dc_saturation_longest_seconds','ac_band_0_5_fraction']
    for a,b in pairs:
        fig,axs=plt.subplots(4,3,figsize=(15,13),layout='constrained')
        for ax,f in zip(axs.ravel(),shown):
            for case,color in [(a,'#2679ac'),(b,'#de8128')]:
                q=data[(data.case==case)&data.VB.notna()].sort_values('VB');ax.plot(q.VB,q[f],'.-',color=color,label=f'Case {case}')
            ax.set(title=f,xlabel='Measured VB (mm)');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle(f'Same-condition cases {a} / {b} | Exploratory relations, no target interpolation for training')
        fig.savefig(ROOT/f'pair{a}_{b}.png',dpi=135);plt.close(fig)
    # Explicit disputed event, valid neighbors; no historical indexing ambiguity.
    fig,axs=plt.subplots(2,1,figsize=(13,7),sharex=True,layout='constrained')
    for run in [11,12,13]:
        r,a,d,dips,_=lookup[12,run];t=np.arange(len(a))/FS
        axs[0].plot(t,a,label=f'Run {run}');axs[1].plot(t,d,label=f'Run {run}')
        for s,e in dips:
            for ax in axs:ax.axvspan(s/FS,e/FS,color='red',alpha=.15)
    for ax in axs:ax.grid(alpha=.2);ax.legend()
    axs[0].set_ylabel('AC RMS');axs[1].set(ylabel='DC 204ms mean',xlabel='Within-run seconds');fig.savefig(ROOT/'case12_interruption.png',dpi=150);plt.close(fig)
    nav=' '.join(f'<a href="#case{i}">{i}</a>' for i in range(1,17));pairlinks=' '.join(f'<a href="pair{a}_{b}.png" target="_blank">{a}/{b}</a>' for a,b in pairs)
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>电流特征跨case一致性</title><style>body{font:16px system-ui;max-width:1800px;margin:24px auto;padding:0 20px;background:#f5f7fa;color:#203248}p{line-height:1.65}img{width:100%}section{padding:16px;background:white;margin:20px 0;scroll-margin-top:70px}nav{position:sticky;top:0;background:white;padding:12px;z-index:2}a{margin:8px;color:#126b9b}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ddd}</style><h1>电流第二版：保留DC饱和，检查包络幅值与时间宽度</h1><p>饱和值完整保留，红线标记顶端。灰色run不用于本轮特征分析，内部同步下陷以粉色标出。横轴是采样秒数拼接，不补run之间的真实时间间隔。仅探索性特征诊断，没有训练模型。</p><p><a href="RESULTS.md">完整结论与限制</a><a href="candidate_features.csv">全部候选特征</a><a href="consistency_summary.csv">一致性指标</a></p><details open><summary>case12实际下陷是run12（去掉坏run1后的第11条）</summary><img src="case12_interruption.png"></details><p>同工况特征对比：PAIRLINKS</p><nav>Case: NAV</nav>PANELS</html>'''.replace('PAIRLINKS',pairlinks).replace('NAV',nav).replace('PANELS',''.join(panels))
    (ROOT/'index.html').write_text(doc,encoding='utf-8')
    audit=dict(passed=True,records=len(qc),accepted=int(qc.accepted.sum()),excluded=qc[~qc.accepted][['case','run','reason','intervals_seconds']].to_dict('records'),
        features=len(fcols),active_labeled_records=int((data.case.isin(ACTIVE)&data.VB.notna()).sum()),original_unchanged=file_hash(source)==original_hash,
        retained_raw_current_and_saturation_verified=True,archive_reload_verified=True,labels_not_used_in_processing=True,no_model_training=True)
    write_json(ROOT/'audit.json',audit);print(json.dumps(audit,indent=2));print(stats.head(20).to_string(index=False));print(pd.DataFrame(transfer).to_string(index=False))


if __name__=='__main__':main()
