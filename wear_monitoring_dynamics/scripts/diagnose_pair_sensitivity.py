import json
from pathlib import Path
import numpy as np,pandas as pd,torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wear_monitoring_dynamics.config import project_path
from wear_monitoring_dynamics.data.records import load_table
from wear_monitoring_dynamics.data.feature_archive import attach_features
from wear_monitoring_dynamics.data.sensor_quality import apply_quality
from wear_monitoring_dynamics.data.preprocessing import transform
from wear_monitoring_dynamics.data.monitor_batch import make_monitor_batch
from wear_monitoring_dynamics.models.registry import build_model
R=project_path('wear_monitoring_dynamics/outputs/current_top18_preprocessing_20261010');O=R/'paired_cases_20261012';m=pd.read_csv(R/'study_manifest.csv');rows=[];torch.set_num_threads(1)
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,(fold,test,partner) in zip(axs.flat,[('fold_a',11,3),('fold_b',3,11),('fold_a',10,4),('fold_b',4,10)]):
 for row in m[(m.fold==fold)&(m.role=='final')&(m.name=='clip_zscore')].itertuples():
  d=Path(row.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config'];s=json.loads((d/'scalers.json').read_text());t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']));t=apply_quality(attach_features(t,cfg['data']['feature_archive']),cfg['data']['sensor_quality']);batch=make_monitor_batch(cfg,t,s,'cpu');model=build_model(cfg);model.load_state_dict(torch.load(d/'last.pt',map_location='cpu',weights_only=True)['model']);model.eval();names=cfg['data']['feature_archive']['names'];ix=cfg['monitor']['selected_indices'];key='smcAC_psd_band_0_5_fraction' if fold=='fold_a' else 'smcAC_psd_band_5_15_fraction';j=names.index(key)
  a=(t['case']==partner)&np.isfinite(t['VB'])&t['sensor_valid'];b=(t['case']==test)&np.isfinite(t['VB'])&t['sensor_valid'];z=batch['sensor'].numpy();ref=pd.DataFrame(z[a],index=t['VB'][a]).groupby(level=0).mean();ids=np.flatnonzero(b&(t['VB']>=ref.index.min())&(t['VB']<=ref.index.max()))
  for mode,replace in [('original',[]),('one_fraction',[j]),('all18',ix)]:
   mod=dict(batch);mod['sensor']=batch['sensor'].clone()
   for k in replace:mod['sensor'][ids,k]=torch.tensor(np.interp(t['VB'][ids],ref.index,ref[k]),dtype=mod['sensor'].dtype)
   with torch.no_grad():pred=model.estimate(mod,ids).numpy().ravel()*s['wear'][1]+s['wear'][0]
   rows.append(dict(case=test,train_partner=partner,seed=row.seed,mode=mode,n=len(ids),mae=np.abs(pred-t['VB'][ids]).mean(),bias=np.mean(pred-t['VB'][ids])))
  if row.seed==0:
   for case,color,mask in [(partner,'tab:blue',a),(test,'tab:orange',b)]:ax.plot(t['VB'][mask],z[mask,j],'o-',color=color,label=f'Case {case}')
   ax.set(title=f'{partner} (train) -> {test} (test)\n{key}',xlabel='Measured VB (mm)',ylabel='Train-scaled feature');ax.legend();ax.grid(alpha=.2)
fig.savefig(O/'frequency_shift.png',dpi=160);plt.close(fig)
a=pd.DataFrame(rows);a.to_csv(O/'oracle_feature_substitution.csv',index=False);print(a.groupby(['case','mode']).agg(mae=('mae','mean'),n=('n','first')).round(5).to_string())

