"""Read-only model/data diagnosis; outputs only, no training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wear_monitoring_dynamics.config import project_path
from wear_monitoring_dynamics.data.records import load_table
from wear_monitoring_dynamics.data.feature_archive import attach_features
from wear_monitoring_dynamics.data.sensor_quality import apply_quality
from wear_monitoring_dynamics.data.preprocessing import transform
R=project_path('wear_monitoring_dynamics/outputs/current_top18_preprocessing_20261010')
O=R/'paired_cases_20261012';O.mkdir(exist_ok=True)
p=pd.read_csv(R/'test_predictions.csv');p=p[p.case.isin([3,4,10,11])&p.method.isin(['clip_zscore','ct_baseline'])].copy()
p['error']=p.predicted_vb_mm-p.vb_mm
p['stage']=pd.cut(p.vb_mm,[-np.inf,.2,.4,np.inf],labels=['VB<=0.2','0.2<VB<=0.4','VB>0.4'])
st=p.groupby(['case','stage','method'],observed=True).agg(mae=('absolute_error','mean'),bias=('error','mean'),rows=('error','size')).reset_index();st['points']=st.rows//5;st.to_csv(O/'stage_errors.csv',index=False)
record=p.groupby(['case','run','time','vb_mm','method']).agg(pred=('predicted_vb_mm','mean'),sd=('predicted_vb_mm','std'),mae=('absolute_error','mean')).reset_index();record.to_csv(O/'record_predictions.csv',index=False)
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,case in zip(axs.flat,[3,11,4,10]):
 q=record[record.case==case];real=q.drop_duplicates('run').sort_values('time');ax.plot(real.time,real.vb_mm,'ko-',label='Measured VB')
 for method,color in [('ct_baseline','tab:blue'),('clip_zscore','tab:orange')]:
  s=q[q.method==method].sort_values('time');ax.plot(s.time,s.pred,'o-',color=color,label=method);ax.fill_between(s.time,s.pred-s.sd,s.pred+s.sd,color=color,alpha=.15)
 ax.set(title=f'Case {case}',xlabel='Dataset time',ylabel='VB (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(O/'predictions.png',dpi=160);plt.close(fig)
manifest=pd.read_csv(R/'study_manifest.csv');features=[];fit=[];wear=[]
for fold,test,partner in [('fold_a',11,3),('fold_b',3,11),('fold_a',10,4),('fold_b',4,10)]:
 d=Path(manifest[(manifest.fold==fold)&(manifest.role=='final')&(manifest.name=='clip_zscore')].iloc[0].run_dir);cfg=json.loads((d/'provenance.json').read_text())['config'];s=json.loads((d/'scalers.json').read_text());t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']));t=apply_quality(attach_features(t,cfg['data']['feature_archive']),cfg['data']['sensor_quality']);z=transform(t,s)['sensor'];ix=cfg['monitor']['selected_indices'];names=cfg['data']['feature_archive']['names']
 masks={c:(t['case']==c)&np.isfinite(t['VB'])&t['sensor_valid'] for c in [test,partner]}
 fig,axs=plt.subplots(6,3,figsize=(15,20),layout='constrained')
 for ax,j in zip(axs.flat,ix):
  rho={}
  for c,color in [(partner,'tab:blue'),(test,'tab:orange')]:
   a=masks[c];v=t['VB'][a];x=z[a,j];rho[c]=pd.Series(v).corr(pd.Series(x),method='spearman');ax.plot(v,x,'o-',color=color,label=f'Case {c} '+('train' if c==partner else 'test'))
  a=masks[partner];ref=pd.DataFrame({'vb':t['VB'][a],'x':z[a,j]}).groupby('vb').x.mean();b=masks[test]&(t['VB']>=ref.index.min())&(t['VB']<=ref.index.max());gap=z[b,j]-np.interp(t['VB'][b],ref.index,ref.values)
  features.append(dict(fold=fold,test_case=test,train_partner=partner,feature=names[j],train_rho=rho[partner],test_rho=rho[test],abs_gap=np.abs(gap).mean(),signed_gap=gap.mean(),overlap_points=int(b.sum())))
  ax.set(title=f'{names[j]}\nrho train={rho[partner]:.2f}, test={rho[test]:.2f}',xlabel='Measured VB (mm)',ylabel='Train-scaled feature');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.suptitle(f'{fold}: training case {partner} vs test case {test}; actual 18 selected features',fontsize=15)
 fig.savefig(O/f'features_{partner}_to_{test}.png',dpi=125);plt.close(fig)
 # Same model's fitting performance on the paired training case, versus test case.
 for row in manifest[(manifest.fold==fold)&(manifest.role=='final')&(manifest.name=='clip_zscore')].itertuples():
  rd=Path(row.run_dir);conf=json.loads((rd/'provenance.json').read_text())['config']
  for method,path in [('clip_zscore',rd),('ct_baseline',Path(conf['monitor']['fallback_checkpoint']).parent)]:
   e=pd.read_csv(path/'epoch_predictions.csv.gz');e=e[(e.epoch==500)&e.case.isin([test,partner])].copy();e['ae']=abs(e.predicted_vb_mm-e.target_vb_mm)
   for (c,split),q in e.groupby(['case','split']):fit.append(dict(fold=fold,case=int(c),split=split,method=method,seed=row.seed,mae=q.ae.mean()))
 # True wear difference at common times, interpolation within measured coverage only.
 a=masks[partner];b=masks[test]&(t['time']>=t['time'][a].min())&(t['time']<=t['time'][a].max());estimated=np.interp(t['time'][b],t['time'][a],t['VB'][a]);wear.append(dict(train_partner=partner,test_case=test,n=int(b.sum()),mean_test_minus_partner_VB_at_same_time=np.mean(t['VB'][b]-estimated),mean_abs_difference=np.mean(abs(t['VB'][b]-estimated))))
f=pd.DataFrame(features);f.to_csv(O/'feature_pair_statistics.csv',index=False);ft=pd.DataFrame(fit).drop_duplicates();ft.to_csv(O/'train_test_fit.csv',index=False);pd.DataFrame(wear).to_csv(O/'wear_time_differences.csv',index=False)
print('STAGES');print(st.round(4).to_string(index=False));print('FIT');print(ft.groupby(['fold','case','split','method']).mae.mean().round(4).to_string());print('FEATURES');print(f.sort_values('abs_gap',ascending=False).groupby('test_case').head(4).round(3).to_string(index=False));print('WEAR');print(pd.DataFrame(wear).round(4).to_string(index=False))

