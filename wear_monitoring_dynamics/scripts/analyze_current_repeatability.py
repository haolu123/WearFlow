"""Descriptive cross-case repeatability of all 50 current features; no training."""
import json,itertools,html
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wear_monitoring_dynamics.config import project_path
from wear_monitoring_dynamics.data.records import load_table
from wear_monitoring_dynamics.data.feature_archive import attach_features
from wear_monitoring_dynamics.data.sensor_quality import apply_quality
R=project_path('wear_monitoring_dynamics/outputs/current_repeatability_20261012');R.mkdir(exist_ok=True);(R/'figures').mkdir(exist_ok=True)
P=R.with_name('current_top18_preprocessing_20261010');m=pd.read_csv(P/'study_manifest.csv');cfg=json.loads((Path(m.iloc[0].run_dir)/'provenance.json').read_text())['config'];t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']));t=apply_quality(attach_features(t,cfg['data']['feature_archive']),cfg['data']['sensor_quality']);names=cfg['data']['feature_archive']['names'];ix=[j for j,n in enumerate(names) if n.startswith(('smcAC_','smcDC_'))];assert len(ix)==50
meta=[];curves={};corr=[];raw=[]
for c in range(1,17):
 mask=(t['case']==c)&t['sensor_valid']&np.isfinite(t['VB']);allmask=t['case']==c;j=np.flatnonzero(allmask)[0];cond=tuple(t[k][j] for k in ['material','DOC','feed']);v=t['VB'][mask];x=t['sensor'][mask][:,ix];df=pd.DataFrame(x,columns=[names[i] for i in ix]);df.insert(0,'VB',v);df.insert(0,'run',t['run'][mask]);df.insert(0,'case',c);raw.append(df);curve=df.groupby('VB')[[names[i] for i in ix]].mean();curves[c]=curve
 meta.append(dict(case=c,material=int(cond[0]),DOC=cond[1],feed=cond[2],records=int(allmask.sum()),valid_labels=len(v),unique_vb=len(curve),vbmin=v.min(),vbmax=v.max()))
 for k,i in enumerate(ix):corr.append(dict(case=c,feature=names[i],rho=float(spearmanr(v,x[:,k]).statistic) if len(curve)>=3 and np.ptp(x[:,k])>0 else np.nan))
meta=pd.DataFrame(meta);meta.to_csv(R/'case_coverage.csv',index=False);pd.concat(raw).to_csv(R/'labeled_current_features.csv',index=False);corr=pd.DataFrame(corr);corr.to_csv(R/'within_case_correlations.csv',index=False)
coverage=[];rows=[];globaliqr={f:np.subtract(*np.percentile(pd.concat(raw)[f],[75,25])) for f in [names[i] for i in ix]}
for a,b in itertools.combinations(range(1,17),2):
 ma=meta[meta.case==a].iloc[0];mb=meta[meta.case==b].iloc[0];same=all(ma[k]==mb[k] for k in ['material','DOC','feed']);lo=max(ma.vbmin,mb.vbmin);hi=min(ma.vbmax,mb.vbmax);width=max(0,hi-lo);ca,cb=curves[a],curves[b];na=int(((ca.index>=lo)&(ca.index<=hi)).sum());nb=int(((cb.index>=lo)&(cb.index<=hi)).sum());eligible=len(ca)>=3 and len(cb)>=3 and width>=.1 and min(na,nb)>=3
 coverage.append(dict(case_a=a,case_b=b,same_condition=same,overlap_low=lo,overlap_high=hi,overlap_width=width,points_a=na,points_b=nb,eligible=eligible))
 if not eligible:continue
 grid=np.linspace(lo,hi,51)
 for f in [names[i] for i in ix]:
  x=np.interp(grid,ca.index,ca[f]);y=np.interp(grid,cb.index,cb[f]);gap=np.mean(abs(x-y));centered=np.mean(abs((x-x.mean())-(y-y.mean())));span=(np.ptp(x)+np.ptp(y))/2;scale=globaliqr[f];rho=spearmanr(x,y).statistic if np.ptp(x)>0 and np.ptp(y)>0 else np.nan
  rows.append(dict(feature=f,case_a=a,case_b=b,pair=f'{a}-{b}',same_condition=same,gap_iqr=gap/scale if scale>1e-12 else np.nan,gap_wear_span=gap/span if span>max(1e-12,scale*1e-6) else np.nan,centered_gap_iqr=centered/scale if scale>1e-12 else np.nan,offset_iqr=abs(np.mean(x-y))/scale if scale>1e-12 else np.nan,curve_rho=rho,overlap_width=width))
coverage=pd.DataFrame(coverage);coverage.to_csv(R/'pair_coverage.csv',index=False);pairs=pd.DataFrame(rows);pairs.to_csv(R/'pair_feature_metrics.csv',index=False)
summary=[]
for f in [names[i] for i in ix]:
 rr=corr[corr.feature==f].rho.dropna();s=dict(feature=f,channel=f.split('_')[0],kind='frequency' if '_psd_' in f else 'time',cases_with_rho=len(rr),mean_abs_rho=rr.abs().mean(),median_rho=rr.median(),positive_cases=int((rr>0).sum()),negative_cases=int((rr<0).sum()),strong_positive=int((rr>=.6).sum()),strong_negative=int((rr<=-.6).sum()))
 for same,label in [(True,'same'),(False,'different')]:
  q=pairs[(pairs.feature==f)&(pairs.same_condition==same)]
  for metric in ['gap_iqr','gap_wear_span','centered_gap_iqr','curve_rho']:s[label+'_'+metric]=q[metric].median()
  s[label+'_pairs']=len(q);s[label+'_gap_iqr_max']=q.gap_iqr.max()
 summary.append(s)
summary=pd.DataFrame(summary).sort_values('same_gap_iqr');summary.to_csv(R/'feature_summary.csv',index=False)
samepairs=coverage[coverage.same_condition];conditions=[(int(q.case_a),int(q.case_b)) for q in samepairs.itertuples()]
for f in [names[i] for i in ix]:
 fig,axs=plt.subplots(2,4,figsize=(16,7),layout='constrained')
 for ax,(a,b) in zip(axs.flat,conditions):
  for c,color in [(a,'tab:blue'),(b,'tab:orange')]:q=curves[c];ax.plot(q.index,q[f],'o-',color=color,label=f'case {c}',markersize=3)
  ma=meta[meta.case==a].iloc[0];ax.set(title=f'{a} vs {b}: M{int(ma.material)}, DOC{ma.DOC}, feed{ma.feed}',xlabel='VB (mm)',ylabel=f);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4))
 fig.suptitle(f+' | raw, unscaled, unclipped; duplicate VB averaged');fig.savefig(R/'figures'/f'{f}.png',dpi=110);plt.close(fig)
order=summary.feature.tolist();heat=pairs[pairs.same_condition].pivot(index='feature',columns='pair',values='gap_iqr').reindex(order)
fig,ax=plt.subplots(figsize=(10,15),layout='constrained');im=ax.imshow(np.log10(1+heat.values),aspect='auto',cmap='magma');ax.set_yticks(range(50),heat.index,fontsize=7);ax.set_xticks(range(len(heat.columns)),heat.columns);ax.set(title='Same-condition discrepancy at matched VB',xlabel='Repeated case pair');fig.colorbar(im,ax=ax,label='log10(1 + mean absolute gap / pooled IQR)');fig.savefig(R/'same_condition_heatmap.png',dpi=140);plt.close(fig)
h=corr.pivot(index='feature',columns='case',values='rho').reindex(order);fig,ax=plt.subplots(figsize=(12,15),layout='constrained');im=ax.imshow(h.values,vmin=-1,vmax=1,aspect='auto',cmap='coolwarm');ax.set_yticks(range(50),h.index,fontsize=7);ax.set_xticks(range(16),h.columns);fig.colorbar(im,ax=ax,label='Within-case Spearman(feature, VB)');fig.savefig(R/'correlation_heatmap.png',dpi=140);plt.close(fig)
page=['<!doctype html><meta charset="utf-8"><title>电流特征跨case一致性</title><style>body{font:16px sans-serif;margin:35px;max-width:1500px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:7px}img{max-width:100%}summary{cursor:pointer;padding:10px;background:#eee}a{color:#1760a5}</style><h1>50个电流特征：跨case一致性</h1><p>全部16case；6只有一个VB点，保留展示但不能评估趋势。原始特征未裁剪、未z-score。相同VB重叠范围内比较；全数据统计只作探索性诊断。</p><p><a href="RESULTS.md">方法与结论</a> · <a href="feature_summary.csv">全部50维汇总</a> · <a href="pair_coverage.csv">配对覆盖</a></p><h2>汇总：越小的gap表示同VB数值越接近；相关性不等于标定一致</h2>',summary.round(4).to_html(index=False),'<h2>同工况差异热图</h2><img src="same_condition_heatmap.png"><h2>各case内相关性</h2><img src="correlation_heatmap.png"><h2>全部50维原始曲线（点击展开，八对相同工况）</h2>']
for f in order:page.append(f'<details><summary>{html.escape(f)}</summary><img loading="lazy" src="figures/{f}.png"></details>')
(R/'index.html').write_text('\n'.join(page),encoding='utf-8')
print('COVERAGE',coverage.groupby('same_condition').agg(total=('eligible','size'),eligible=('eligible','sum')).to_string());print('TOP STRONG');print(summary[summary.mean_abs_rho>=.6].head(14).round(3).to_string(index=False));print('WORST');print(summary.tail(10).round(3).to_string(index=False));print('ALL SUMMARY');print(summary[['feature','mean_abs_rho','positive_cases','negative_cases','same_gap_iqr','different_gap_iqr','same_gap_wear_span']].round(3).to_string(index=False))

