"""Predeclared current component features, descriptive calibration and pool transfer."""
from pathlib import Path
from itertools import combinations
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.signal import welch, detrend
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize_scalar
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .current_feature_consistency import envelopes, normalize_profile, relations
from .compare_current_calibration import compare, PAIRS, ACTIVE
from ..data.current_processing import FS, DC_RAIL, spans
from ..data.splits import POOLS

BASE=Path(__file__).resolve().parents[1]
OLD=BASE/'outputs/current_features_20261013'
OUT=BASE/'outputs/current_components_20261013'
DEFS={
 'net_ac_rms':'稳定段AC RMS减去同run切入前低负载段AC RMS',
 'net_ac_power':'稳定段AC均方值减去同run低负载段AC均方值；允许负值',
 'net_dc_mean':'稳定段DC均值减去同run低负载段DC均值；保留饱和',
 'net_dc_median':'稳定段DC中位数减去同run低负载段DC中位数',
 'net_carrier_power':'稳定段主峰±0.75Hz功率减去低负载段相同频带功率',
 'net_harmonic_power':'稳定段二三谐波±0.75Hz合计功率减去低负载段对应功率',
 'net_spectral_floor':'稳定段5–100Hz谱密度中位数减去低负载段对应中位数',
 'period_h1_amplitude':'主频正余弦回归幅值；含常数、线性趋势及三次谐波',
 'period_h2_amplitude':'二次谐波正余弦回归幅值',
 'period_h3_amplitude':'三次谐波正余弦回归幅值',
 'period_residual_rms':'扣除趋势和三次谐波回归后的残差RMS',
 'period_residual_mad':'周期回归残差的1.4826倍MAD',
 'period_residual_q90':'周期回归残差绝对值P90',
 'period_residual_fraction':'残差RMS除以去趋势AC RMS',
 'cycle_amplitude_cv':'完整主周期内AC RMS的标准差/均值',
 'cycle_amplitude_iqr':'完整主周期内AC RMS的P75-P25',
 'cycle_adjacent_difference':'相邻完整周期各插值32相位点后的差值RMS/去趋势AC RMS',
 'cycle_template_residual':'各完整周期减去跨周期平均模板的RMS/去趋势AC RMS',
 'cycle_amplitude_median':'完整主周期内AC RMS的中位数',
 'env_detrended_mad':'204ms局部RMS包络去线性趋势后的1.4826倍MAD；丢弃边缘0.12s',
 'env_detrended_iqr':'上述包络残差P75-P25',
 'env_low_power':'上述包络残差0.25–1Hz功率',
 'env_mid_power':'上述包络残差1–4Hz功率',
 'env_lag025_correlation':'上述包络残差滞后0.25s的相关系数',
 'env_relative_mad':'上述包络MAD除以包络中位数',
 'peak_power':'Welch主峰±0.75Hz积分功率',
 'peak_background_corrected_power':'主峰±0.75Hz积分功率减去两侧2–4Hz处背景谱密度中位数乘带宽；允许负值',
 'peak_width_hz':'主峰±2Hz内去背景正功率加权的频率标准差',
 'peak_concentration':'主峰±0.75Hz功率/5–100Hz功率',
 'peak_h2_power':'二次谐波±0.75Hz积分功率',
 'peak_h3_power':'三次谐波±0.75Hz积分功率',
 'peak_h23_ratio':'二三谐波功率和/基波功率',
 'peak_sideband_power':'距离主峰1–3Hz两侧功率之和',
 'peak_sideband_ratio':'旁带功率/主峰功率',
 'peak_spectral_floor':'5–100Hz谱密度中位数',
 'control_ac_rms':'同一稳定8秒窗口的AC RMS（对照）',
 'control_dc_mean':'同一稳定8秒窗口的DC均值（对照，饱和保留）',
 'control_ac_env_median':'同一稳定8秒窗口的204ms AC RMS包络中位数（对照）',
}

def choose_windows(ac,dc):
    a,d=envelopes(ac,dc);za,_,_=normalize_profile(a);zd,_,_=normalize_profile(d)
    candidates=[(s+125,e-125) for s,e in spans((za>.6)&(zd>.5)) if e-s>=2250]
    if not candidates:return None,None
    s,e=max(candidates,key=lambda v:v[1]-v[0]);mid=(s+e)//2;stable=(mid-1000,mid+1000)
    # Low-load proxy, not a claim that the motor is physically unloaded.
    onset=np.flatnonzero((za>.5)&(zd>.5))[0]
    low=(za<.2)&(zd<.2);low[max(0,onset-50):]=False
    choices=[]
    for s,e in spans(low):
        s,e=s+25,e-25
        if e-s<250:continue
        s=max(s,e-1000);quarter=(e-s)//4
        # Reject low-amplitude portions of a continuing entry ramp, using signal-only drift.
        if all(abs(np.median(z[e-quarter:e])-np.median(z[s:s+quarter]))<=.08 for z in (za,zd)):
            choices.append((s,e))
    idle=None
    if choices:
        s,e=max(choices,key=lambda v:v[1]-v[0]);idle=(max(s,e-1000),e)
    return stable,idle

def spectrum(x):
    # Common 1Hz grid for load and short baseline segments; equal Welch window.
    return welch(x,fs=FS,nperseg=250,noverlap=125,detrend='linear')

def band(freq,p,lo,hi):
    # Fractional-bin overlap avoids discontinuous energy jumps when a peak shifts across a bin boundary.
    df=freq[1]-freq[0]
    weights=np.maximum(0,np.minimum(freq+df/2,hi)-np.maximum(freq-df/2,lo))
    return float(np.sum(p*weights))

def component_features(ac,dc,idle=None):
    ac=np.asarray(ac,float);dc=np.asarray(dc,float);n=len(ac);t=np.arange(n)/FS
    f={k:np.nan for k in DEFS};y=detrend(ac);rms=float(np.sqrt(np.mean(ac**2)))
    env=np.sqrt(np.maximum(uniform_filter1d(ac**2,51,mode='reflect'),0))[30:-30]
    er=detrend(env);mad=float(1.4826*np.median(abs(er-np.median(er))))
    ef,ep=welch(er,fs=FS,nperseg=len(er),detrend=False)
    f.update(control_ac_rms=rms,control_dc_mean=float(np.mean(dc)),control_ac_env_median=float(np.median(env)),
        env_detrended_mad=mad,env_detrended_iqr=float(np.quantile(er,.75)-np.quantile(er,.25)),
        env_low_power=band(ef,ep,.25,1),env_mid_power=band(ef,ep,1,4),env_relative_mad=mad/max(np.median(env),1e-12),
        env_lag025_correlation=float(np.corrcoef(er[:-62],er[62:])[0,1]) if np.std(er)>1e-12 else np.nan)
    if idle is not None:
        ia,idc=idle
        f.update(net_ac_rms=rms-np.sqrt(np.mean(ia**2)),net_ac_power=np.mean(ac**2)-np.mean(ia**2),
                 net_dc_mean=np.mean(dc)-np.mean(idc),net_dc_median=np.median(dc)-np.median(idc))
    hz,psd=welch(ac,fs=FS,nperseg=n,detrend='linear');mask=(hz>=5)&(hz<=30)
    f0=float(hz[mask][np.argmax(psd[mask])]);prom=float(psd[mask].max()/max(np.median(psd[mask]),1e-20))
    if prom<8:return f,dict(frequency=f0,prominence=prom,periodic_valid=False)
    def design(freq):
        return np.column_stack([np.ones(n),t-t.mean()]+[u(2*np.pi*k*freq*t) for k in (1,2,3) for u in (np.sin,np.cos)])
    def objective(freq):
        x=design(freq);b=np.linalg.lstsq(x,ac,rcond=None)[0];return float(np.mean((ac-x@b)**2))
    opt=minimize_scalar(objective,bounds=(max(5,f0-FS/n),min(30,f0+FS/n)),method='bounded',options={'xatol':1e-5})
    f0=float(opt.x);x=design(f0);b=np.linalg.lstsq(x,ac,rcond=None)[0];res=ac-x@b
    rr=float(np.sqrt(np.mean(res**2)));yr=float(np.sqrt(np.mean(y**2)))
    for k in (1,2,3):f[f'period_h{k}_amplitude']=float(np.hypot(b[2*k],b[2*k+1]))
    f.update(period_residual_rms=rr,period_residual_mad=float(1.4826*np.median(abs(res-np.median(res)))),
             period_residual_q90=float(np.quantile(abs(res),.9)),period_residual_fraction=rr/max(yr,1e-12))
    cycles=[]
    for j in range(int(t[-1]*f0)):
        ct=(j+np.arange(32)/32)/f0
        if ct[-1]<=t[-1]:cycles.append(np.interp(ct,t,y))
    cycles=np.asarray(cycles);cr=np.sqrt(np.mean(cycles**2,axis=1))
    f.update(cycle_amplitude_cv=float(np.std(cr)/max(np.mean(cr),1e-12)),cycle_amplitude_iqr=float(np.quantile(cr,.75)-np.quantile(cr,.25)),
        cycle_amplitude_median=float(np.median(cr)),cycle_adjacent_difference=float(np.sqrt(np.mean(np.diff(cycles,axis=0)**2))/max(yr,1e-12)),
        cycle_template_residual=float(np.sqrt(np.mean((cycles-cycles.mean(axis=0))**2))/max(yr,1e-12)))
    freq,p=spectrum(ac);bp=lambda lo,hi:band(freq,p,lo,hi)
    peak=bp(f0-.75,f0+.75);h2=bp(2*f0-.75,2*f0+.75);h3=bp(3*f0-.75,3*f0+.75)
    bgmask=(abs(freq-f0)>=2)&(abs(freq-f0)<4);bg=float(np.median(p[bgmask]))
    near=abs(hz-f0)<2;weight=np.maximum(psd[near]-bg,0)
    width=float(np.sqrt(np.sum(weight*(hz[near]-f0)**2)/max(weight.sum(),1e-20)))
    side=bp(f0-3,f0-1)+bp(f0+1,f0+3);floor=float(np.median(p[(freq>=5)&(freq<100)]))
    f.update(peak_power=peak,peak_background_corrected_power=peak-bg*1.5,
        peak_width_hz=width,peak_concentration=peak/max(bp(5,100),1e-20),peak_h2_power=h2,peak_h3_power=h3,
        peak_h23_ratio=(h2+h3)/max(peak,1e-20),peak_sideband_power=side,peak_sideband_ratio=side/max(peak,1e-20),peak_spectral_floor=floor)
    if idle is not None:
        ih,ip=spectrum(idle[0]);f.update(net_carrier_power=peak-band(ih,ip,f0-.75,f0+.75),
            net_harmonic_power=h2+h3-band(ih,ip,2*f0-.75,2*f0+.75)-band(ih,ip,3*f0-.75,3*f0+.75),
            net_spectral_floor=floor-float(np.median(ip[(ih>=5)&(ih<100)])))
    return f,dict(frequency=f0,prominence=prom,periodic_valid=True)

def scale_for(data,f):
    ranges=data.groupby('case')[f].apply(lambda s:s.quantile(.9)-s.quantile(.1)).dropna()
    return float(np.sqrt(np.mean(ranges**2))) if len(ranges) else np.nan

def pair_metrics(data,f,pairs,scale):
    rows=[]
    for a,b in pairs:
        r=compare(data.loc[data.case==a,['VB',f]].values,data.loc[data.case==b,['VB',f]].values)
        if r is None:continue
        r.update(feature=f,pair=f'{a}/{b}',scale=scale,low_variation=r['overlap_swing']<=.05*scale+1e-12)
        r['ratio']=r['raw_gap']/r['overlap_swing'] if r['overlap_swing']>.05*scale+1e-12 else np.nan
        r['scaled_gap']=r['raw_gap']/scale if scale>1e-12 else np.nan;rows.append(r)
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(exist_ok=True)
    src=BASE.parent/'mill_numpy/mill.npy';source_hash=hashlib.sha256(src.read_bytes()).hexdigest()
    # Candidate definitions and fixed thresholds saved before looking at relations to VB.
    protocol=dict(source_sha256=source_hash,definitions=DEFS,fs=FS,stable_seconds=8,half_seconds=4,
        idle_min_seconds=1,idle_stationarity='first/last-quarter median difference <=8% of run q90-q05 for both AC envelope and DC level',retains_dc_saturation=True,source_quality=str(OLD/'quality_decisions.csv'),
        selector='TRAIN ONLY: coverage>=0.8, >=5 defined rhos, >=80% directional agreement, median signed rho>=0.5, split-half coverage>=0.8, median abs half difference / train within-case range<=0.15; then lowest train all-condition pair gap/swing, top5',
        caveat='Each pool has only one case per condition; train ranking is across conditions. Matched-condition cross-pool pairs are transfer diagnostics only, never selection scores. Dataset previously explored; not untouched external validation.',
        peak='dominant 5-30Hz with prominence>=8; 3-harmonic fit; not assigned a mechanical identity',
        repeatability='first vs last 4s of same stable8s, same idle reference; temporal consistency proxy, not independent sensor repeatability',
        known_limitation='One-second Welch windows give 1Hz resolution, limit low-load spectral variance; compare spectral fine structure cautiously; no full-run widths split into halves.')
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    old=pd.read_csv(OLD/'candidate_features.csv');allowed=set(zip(old.case,old.run));rows=[];halves=[];quality=[];signals={}
    records=sorted(np.load(src,allow_pickle=True).tolist(),key=lambda r:(int(r['case']),int(r['run'])))
    for rec in records:
        key=(int(rec['case']),int(rec['run']))
        if key not in allowed:continue
        ac=np.asarray(rec['smcAC'],float).ravel();dc=np.asarray(rec['smcDC'],float).ravel();stable,idle=choose_windows(ac,dc)
        meta={k:rec[k] for k in ['case','run','VB','time','DOC','feed','material']}
        vals={k:np.nan for k in DEFS};q=dict(case=key[0],run=key[1],stable_available=stable is not None,idle_available=idle is not None)
        if stable:
            s,e=stable;ref=(ac[slice(*idle)],dc[slice(*idle)]) if idle else None
            vals,diag=component_features(ac[s:e],dc[s:e],ref);q.update(diag,stable_start=s/FS,stable_end=e/FS,idle_start=idle[0]/FS if idle else np.nan,idle_end=idle[1]/FS if idle else np.nan,dc_rail_fraction=float(np.mean(dc[s:e]>=DC_RAIL-1e-9)))
            first,_=component_features(ac[s:s+1000],dc[s:s+1000],ref);last,_=component_features(ac[e-1000:e],dc[e-1000:e],ref)
            halves.append(dict(case=key[0],run=key[1],VB=rec['VB'],**{f:abs(first[f]-last[f]) for f in DEFS}))
            signals[key]=(ac,dc,stable,idle)
        rows.append(dict(**meta,**vals));quality.append(q)
    new=pd.DataFrame(rows);half=pd.DataFrame(halves);qc=pd.DataFrame(quality)
    new.to_csv(OUT/'new_features.csv',index=False);half.to_csv(OUT/'half_differences.csv',index=False);qc.to_csv(OUT/'window_quality.csv',index=False)
    data=old.merge(new,on=['case','run','VB','time','DOC','feed','material'],how='left',validate='one_to_one')
    data.to_csv(OUT/'all_features.csv',index=False)
    data=data[data.case.isin(ACTIVE)&data.VB.notna()];half=half[half.case.isin(ACTIVE)&half.VB.notna()]
    features=list(old.columns[7:])+list(DEFS);corr=relations(data,features,ACTIVE);corr.to_csv(OUT/'case_correlations.csv',index=False)
    # Freeze selection in both directions before the seven matched-condition comparisons.
    selected=[];trainrows=[]
    for fold,cases in POOLS.items():
        train=data[data.case.isin(cases)];hh=half[half.case.isin(cases)]
        for f in DEFS:
            scale=scale_for(train,f);v=corr[(corr.feature==f)&corr.case.isin(cases)].rho.dropna();direction=1 if (v>0).sum()>=(v<0).sum() else -1
            pr=pair_metrics(train,f,list(combinations(cases,2)),scale)
            score=float(pr.ratio.median()) if len(pr) else np.nan
            rep=float(hh[f].median()/scale) if scale>1e-12 else np.nan;coverage=train[f].notna().mean();repcoverage=hh[f].notna().sum()/len(train)
            strength=float((v*direction).median()) if len(v) else np.nan;agree=float((v*direction>0).mean()) if len(v) else 0
            eligible=bool(coverage>=.8 and len(v)>=5 and agree>=.8 and strength>=.5 and repcoverage>=.8 and rep<=.15 and np.isfinite(score))
            trainrows.append(dict(fold=fold,feature=f,coverage=coverage,defined=len(v),direction=direction,agreement=agree,median_signed_rho=strength,repeatability=rep,repeatability_coverage=repcoverage,train_ratio=score,eligible=eligible,scale=scale))
        ranked=sorted([r for r in trainrows if r['fold']==fold and r['eligible']],key=lambda r:(r['train_ratio'],r['feature']))[:5]
        selected.extend(ranked)
    pd.DataFrame(trainrows).to_csv(OUT/'train_selection_scores.csv',index=False)
    (OUT/'frozen_shortlists.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    stats=[];details=[];transfer=[]
    for f in features:
        scale=scale_for(data,f);p=pair_metrics(data,f,PAIRS,scale)
        if len(p):details.extend(p.to_dict('records'))
        v=corr[corr.feature==f].rho.dropna();direction=1 if (v>0).sum()>=(v<0).sum() else -1
        stats.append(dict(feature=f,group='old' if f not in DEFS else ('control' if f.startswith('control_') else 'new'),coverage=data[f].notna().mean(),
            pairs=len(p),informative_pairs=int(p.ratio.notna().sum()) if len(p) else 0,median_ratio=p.ratio.median() if len(p) else np.nan,
            worst_ratio=p.ratio.max() if len(p) else np.nan,median_scaled_gap=p.scaled_gap.median() if len(p) else np.nan,
            median_abs_rho=v.abs().median(),direction=direction,direction_cases=int((v*direction>0).sum()),defined_cases=len(v),
            repeatability=float(half[f].median()/scale) if f in DEFS and scale>1e-12 else np.nan,
            repeatability_coverage=half[f].notna().sum()/len(data) if f in DEFS else np.nan))
    for r in selected:
        testcases=[c for c in ACTIVE if c not in POOLS[r['fold']]];v=corr[(corr.feature==r['feature'])&corr.case.isin(testcases)].rho.dropna()
        p=pair_metrics(data,r['feature'],PAIRS,r['scale']);h=half[half.case.isin(testcases)]
        transfer.append(dict(**r,test_defined=len(v),test_direction_cases=int((v*r['direction']>0).sum()),test_signed_rho=(v*r['direction']).median(),
            test_repeatability=h[r['feature']].median()/r['scale'],matched_transfer_ratio=p.ratio.median() if len(p) else np.nan,transfer_pairs=len(p)))
    summary=pd.DataFrame(stats).sort_values('median_ratio');summary.to_csv(OUT/'comparison.csv',index=False)
    pd.DataFrame(details).to_csv(OUT/'pair_details.csv',index=False);pd.DataFrame(transfer).to_csv(OUT/'shortlist_transfer.csv',index=False)
    # Common-row comparison avoids giving features with missing difficult records an unfair advantage.
    matched=[]
    for f in DEFS:
        subset=data.dropna(subset=[f])
        for baseline in ['ac_wave_rms','dc_level_p90','control_ac_rms','control_dc_mean']:
            sub=subset.dropna(subset=[baseline]);a=pair_metrics(sub,f,PAIRS,scale_for(sub,f));b=pair_metrics(sub,baseline,PAIRS,scale_for(sub,baseline))
            if len(a) and len(b):
                both=a[['pair','ratio']].merge(b[['pair','ratio']],on='pair',suffixes=('_new','_baseline')).dropna()
                matched.append(dict(feature=f,baseline=baseline,common_records=len(sub),pairs=len(both),new_ratio=both.ratio_new.median(),baseline_ratio=both.ratio_baseline.median(),pairs_better=int((both.ratio_new<both.ratio_baseline).sum())))
    pd.DataFrame(matched).to_csv(OUT/'common_support_comparison.csv',index=False)
    # All new features plus familiar old controls; raw shared y axes, no case centering.
    shown=list(DEFS)+['ac_wave_rms','dc_level_p90','ac_env_plateau_median']
    for f in shown:
        fig,axs=plt.subplots(2,4,figsize=(16,7),sharey=True,layout='constrained')
        for ax,(a,b) in zip(axs.flat,PAIRS):
            for case in (a,b):
                d=data[data.case==case].sort_values('VB');ax.plot(d.VB,d[f],'.-',label=str(case),lw=1)
            p=pair_metrics(data,f,[(a,b)],scale_for(data,f))
            if len(p):
                r=p.iloc[0];ax.axvspan(r.overlap_low,r.overlap_high,alpha=.07,color='green');ax.set_title(f'{a}/{b}: gap/swing={r.ratio:.3f}'+(' [sparse]' if r['sparse'] else ''),fontsize=9)
            else:ax.set_title(f'{a}/{b}: unavailable')
            ax.set_xlabel('VB (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax=axs.flat[-1];ax.axis('off');r=summary[summary.feature==f].iloc[0]
        ax.text(0,1,f'Median gap/swing: {r.median_ratio:.3f}\nWorst: {r.worst_ratio:.3f}\nCoverage: {r.coverage:.0%}\nInformative pairs: {r.informative_pairs}/7\nDirection: {r.direction_cases}/{r.defined_cases}\nMedian abs rho: {r.median_abs_rho:.3f}\nHalf difference / scale: {r.repeatability:.3f}\n\nRaw feature values, common y axis.\nNo case centering.\nGreen: overlapping measured VB.',va='top',transform=ax.transAxes)
        fig.suptitle(f);fig.savefig(OUT/(f+'.png'),dpi=115);plt.close(fig)
    # Window diagnostic for first accepted run of each case (not selected by wear relation).
    fig,axs=plt.subplots(4,4,figsize=(17,12),layout='constrained')
    for case,ax in zip(range(1,17),axs.flat):
        keys=[k for k in signals if k[0]==case]
        if not keys:continue
        key=keys[0];ac,dc,stable,idle=signals[key];a,_=envelopes(ac,dc);t=np.arange(len(ac))/FS
        ax.plot(t,a,lw=.7);ax.axvspan(stable[0]/FS,stable[1]/FS,color='green',alpha=.2)
        if idle:ax.axvspan(idle[0]/FS,idle[1]/FS,color='blue',alpha=.15)
        ax.set_title(f'case {case}, run {key[1]}');ax.set_xlabel('seconds');ax.grid(alpha=.2)
    fig.suptitle('Window audit: AC RMS | green=stable8s; blue=pre-entry low-load reference');fig.savefig(OUT/'window_audit.png',dpi=120);plt.close(fig)
    trs=[]
    for r in summary.itertuples():
        path=r.feature+'.png' if r.feature in shown else '../current_calibration_consistency_20261013/'+r.feature+'.png'
        label=f'<a href="{path}">{r.feature}</a>'
        trs.append(f'<tr><td>{r.group}</td><td>{label}</td><td>{r.median_ratio:.3f}</td><td>{r.worst_ratio:.3f}</td><td>{r.coverage:.0%}</td><td>{r.informative_pairs}/7</td><td>{r.direction_cases}/{r.defined_cases}</td><td>{r.median_abs_rho:.3f}</td><td>{r.repeatability:.3f}</td></tr>')
    page='''<!doctype html><meta charset="utf-8"><title>电流成分特征探索</title><style>body{font:15px system-ui;margin:30px;line-height:1.6}p{max-width:1100px}table{border-collapse:collapse}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}td:nth-child(2){text-align:left}th{position:sticky;top:0;background:#eaf0f5;cursor:pointer}input{padding:8px;width:420px}</style>
<h1>电流成分：新特征与原61项比较</h1><p>35个新候选＋3个同窗口对照，共38项；与旧61项一起比较。DC饱和保留，异常剔除继承上一轮。相同VB差异除以重叠区间特征变化幅度，越小越接近；不能单凭此排名判定磨损信息有用。低变化对、缺失记录、随机波动会影响排名。</p>
<p>每run选定稳定8秒，并比较前后各4秒得到时间一致性指标。空载参照实际为切入前低负载代理，不保证物理空载。频率来自数据主峰，未认定为刀齿或转频。表中旧特征重复性留空（nan），不能与整段宽度等直接切半比较。</p>
<p><a href="RESULTS.html">结论、筛选与局限</a> · <a href="comparison.csv">全部99项结果</a> · <a href="common_support_comparison.csv">相同有效记录上的公平比较</a> · <a href="shortlist_transfer.csv">训练侧筛选与迁移</a> · <a href="window_audit.png">窗口检查</a> · <a href="protocol.json">全部特征定义与规则</a> · <a href="pair_details.csv">逐对明细</a></p>
<input placeholder="筛选特征名或 new / old / control" oninput="document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.innerText.toLowerCase().includes(this.value.toLowerCase()))"><table><thead><tr>'''+''.join('<th>'+x+'</th>' for x in ['组别','特征（点开原值图）','差异/变化↓','最差一对↓','覆盖率','有效对','多数方向case数','中位|rho|','半段差异/尺度↓'])+'</tr></thead><tbody>'+''.join(trs)+'''</tbody></table><script>document.querySelectorAll('th').forEach((h,i)=>h.onclick=()=>{h.asc=!h.asc;let b=document.querySelector('tbody');[...b.rows].sort((a,c)=>{let x=a.cells[i].innerText,y=c.cells[i].innerText;let v=i<2?x.localeCompare(y):parseFloat(x)-parseFloat(y);return h.asc?v:-v}).forEach(r=>b.append(r))})</script>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    assert hashlib.sha256(src.read_bytes()).hexdigest()==source_hash
    assert len(new)==len(old)==164 and len(data)==137 and len(DEFS)==38 and len(summary)==99
    audit=dict(new_candidates=35,controls=3,old_candidates=61,total=99,accepted_records=164,labeled_records=137,stable_records=int(qc.stable_available.sum()),idle_records=int(qc.idle_available.sum()),source_unchanged=True,plots=len(shown),selected=len(selected))
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit));print(summary[summary.group!='old'].to_string(index=False));print('TRANSFER');print(pd.DataFrame(transfer).to_string(index=False))

if __name__=='__main__':main()
