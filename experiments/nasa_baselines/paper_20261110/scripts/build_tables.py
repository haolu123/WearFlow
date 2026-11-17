"""Frozen paper tables: complete fusion system and independently evaluated baselines."""
from wearflow_paths import ROOT as _WF_ROOT
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
PAPER=Path(__file__).resolve().parents[1];SUITE=PAPER.parent;OUT=PAPER/'tables'
SOURCE=Path(str(_WF_ROOT / 'wear_monitoring_dynamics/outputs/current12_abcd_from100_20261019/predictions.csv.gz'))
CK=['fold','case','run'];FK=['fold','seed','case','origin_run','target_run','horizon']
EXTF=['sun2020_lstm_module','Kim_TF_cubic_mean_NASA','persistence_sensor100','linear2_sensor100','linear2_nonnegative_sensor100']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(df,task,scope):
 z=df.copy();z['ae']=abs(z.pred_mm-z.target_mm);z['se']=(z.pred_mm-z.target_mm)**2;allc=[];out=[]
 for h,g in z.groupby('horizon'):
  for region,mask in [('overall',np.ones(len(g),bool)),('fold_a',g.fold=='fold_a'),('fold_b',g.fold=='fold_b'),('fold_a_in',(g.fold=='fold_a')&(g.target_mm<=.74)),('fold_a_out',(g.fold=='fold_a')&(g.target_mm>.74))]:
   q=g[mask];c=q.groupby(['method','fold','case','seed']).agg(mae_mm=('ae','mean'),mse_mm2=('se','mean'),targets=('ae','size')).reset_index();c['rmse_mm']=np.sqrt(c.mse_mm2);c['task']=task;c['scope']=scope;c['horizon']=h;c['region']=region;allc.append(c)
   for m,s in c.groupby('method'):
    av=s.groupby(['fold','case'])[['mae_mm','rmse_mm']].mean();out.append(dict(method=m,task=task,scope=scope,horizon=h,region=region,mae_mm=av.mae_mm.mean(),case_macro_rmse_mm=av.rmse_mm.mean(),cases=len(av),seed_target_rows=int(s.targets.sum())))
 return pd.DataFrame(out),pd.concat(allc,ignore_index=True)
def main():
 OUT.mkdir(exist_ok=True);raw=pd.read_csv(SOURCE);fuse=raw[(raw.protocol=='fusion')&(raw.split=='test_diagnostic')&(((raw.group=='A')&(raw.epoch==0))|((raw.group!='A')&(raw.epoch==100)))].copy();fuse['method']='fusion_'+fuse.group;fuse=fuse.rename(columns={'target_vb_mm':'target_mm','predicted_vb_mm':'pred_mm'})
 original=fuse[fuse.task.isin(['current','future'])].copy();original.to_csv(OUT/'original_fusion_predictions.csv',index=False)
 maps=[]
 for fold in ['fold_a','fold_b']:
  a=dict(np.load(SUITE/'data'/fold/'final/arrays.npz'));maps.append(pd.DataFrame({'fold':fold,'case':a['case'],'run':a['run'],'time':a['time'],'truth':a['y_mm']}))
 truth=pd.concat(maps);assert not truth.duplicated(CK).any()
 fc=fuse[fuse.task=='current'].copy();fc['run']=fc.target_run
 for (m,s),g in fc.groupby(['method','seed']):assert len(g)==128 and not g.duplicated(CK).any()
 ckeys=set(map(tuple,fc[fc.method=='fusion_A'][CK].to_numpy()));assert len(ckeys)==128
 for _,g in fc.groupby(['method','seed']):assert set(map(tuple,g[CK].to_numpy()))==ckeys
 extc=pd.read_csv(SUITE/'summary/current_predictions.csv');cur=pd.concat([fc,extc],ignore_index=True);cur['horizon']=0;cur['in_common']=[tuple(k) in ckeys for k in cur[CK].to_numpy()];cur['original_fusion']=cur.method.str.startswith('fusion_')
 j=cur.merge(truth,on=CK,validate='many_to_one');assert len(j)==len(cur);np.testing.assert_allclose(j.target_mm,j.truth,atol=1e-12,rtol=0);assert np.isfinite(cur.pred_mm).all()
 assert not cur.duplicated(['method','seed']+CK).any()
 commonc=cur[cur.in_common];assert all(len(g)==128 for _,g in commonc.groupby(['method','seed']))
 future=fuse[fuse.task=='future'].copy();e=pd.read_csv(SUITE/'summary/future_predictions.csv');future=pd.concat([future,e[e.method.isin(EXTF)]],ignore_index=True);assert set(future.method)==set(['fusion_'+s for s in 'ABCD']+EXTF)
 for (m,s,h),g in future[future.method.str.startswith('fusion_')].groupby(['method','seed','horizon']):assert len(g)=={1:114,2:100,3:86}[h]
 assert not future.duplicated(['method']+FK).any()
 fkeys=set(map(tuple,future[future.method=='fusion_A'][FK].to_numpy()))
 for m in ['fusion_B','fusion_C','fusion_D']:assert set(map(tuple,future[future.method==m][FK].to_numpy()))==fkeys
 target=truth.rename(columns={'run':'target_run','time':'target_time','truth':'target_truth'});origin=truth.rename(columns={'run':'origin_run','time':'origin_time','truth':'origin_target_mm'})
 future=future.drop(columns=['target_time','origin_time','target_truth','origin_target_mm','origin_pred_mm'],errors='ignore')
 future=future.merge(target,on=['fold','case','target_run'],how='left',validate='many_to_one').merge(origin,on=['fold','case','origin_run'],how='left',validate='many_to_one');np.testing.assert_allclose(future.target_mm,future.target_truth,atol=1e-12,rtol=0)
 actual=future.target_time-future.origin_time;assert (actual>0).all();known=future.dt.notna();np.testing.assert_allclose(future.loc[known,'dt'],actual[known],atol=1e-12,rtol=0);future['dt']=actual
 assert np.isfinite(future.origin_target_mm).all() and not np.isinf(future.pred_mm).any()
 # Explicit origin estimates for physical-increment plots; Kim is not an IVP.
 origin_frames=[]
 for m in future.method.unique():
  source=fc[fc.method==m] if m.startswith('fusion_') else extc[extc.method==('Kim_TF_raw_ensemble' if m=='Kim_TF_cubic_mean_NASA' else 'sensor12_100')]
  q=source[['fold','seed','case','run','pred_mm']].rename(columns={'run':'origin_run','pred_mm':'origin_pred_mm'});q['method']=m;origin_frames.append(q)
 future=future.merge(pd.concat(origin_frames),on=['method','fold','seed','case','origin_run'],how='left',validate='many_to_one');assert future.origin_pred_mm.notna().all()
 future['origin_kind']=np.where(future.method=='Kim_TF_cubic_mean_NASA','monitor_reference_not_polynomial_initial_condition',np.where(future.method.str.startswith('fusion_'),'fused_state','sensor100_monitor'))
 future['in_common']=False;future['original_fusion']=future.method.str.startswith('fusion_');coverage=[]
 for h,g in future.groupby('horizon'):
  sets=[]
  for m,q in g.groupby('method'):
   assert set(q.seed)==set(range(5));patterns=[]
   for seed,z in q.groupby('seed'):
    k=[x for x in FK if x!='seed'];patterns.append((set(map(tuple,z[np.isfinite(z.pred_mm)][k].to_numpy())),set(map(tuple,z[~np.isfinite(z.pred_mm)][k].to_numpy()))))
   assert all(p==patterns[0] for p in patterns)
   sets.append(set(map(tuple,q[np.isfinite(q.pred_mm)][FK].to_numpy())))
  common=set.intersection(*sets);assert common
  future.loc[g.index,'in_common']=[tuple(k) in common for k in g[FK].to_numpy()]
  for (m,s),z in future.loc[g.index].groupby(['method','seed']):coverage.append(dict(method=m,seed=int(s),horizon=int(h),provided=len(z),finite=int(np.isfinite(z.pred_mm).sum()),common=int(z.in_common.sum()),common_cases=z[z.in_common].case.nunique()))
 cur.to_csv(OUT/'current_predictions.csv',index=False);future.to_csv(OUT/'future_predictions.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
 summaries=[];cases=[];keys=[]
 for task,data in [('current',cur),('future',future)]:
  for scope,mask in [('all_available',np.isfinite(data.pred_mm)),('original_fusion',data.original_fusion),('current_common128' if task=='current' else 'all_methods_common',data.in_common)]:
   d=data[mask];s,c=metrics(d,task,scope);summaries.append(s);cases.append(c);cols=CK if task=='current' else FK
   k=d[['method','seed']+[x for x in cols if x!='seed']].copy();k['task']=task;k['scope']=scope
   if task=='current':k['target_run']=k.run;k['origin_run']=k.run;k['horizon']=0
   keys.append(k)
 summary=pd.concat(summaries);case=pd.concat(cases);summary.to_csv(OUT/'summary.csv',index=False);case.to_csv(OUT/'case_seed_metrics.csv',index=False);pd.concat(keys).to_csv(OUT/'target_keys.csv',index=False)
 boot=[]
 for task,scope in [('current','current_common128'),('future','all_methods_common')]:
  c=case[(case.task==task)&(case.scope==scope)&(case.region=='overall')]
  for h,g in c.groupby('horizon'):
   pairs=[('fusion_D',m) for m in sorted(g.method.unique()) if m!='fusion_D']
   if task=='future':pairs += [('fusion_A',m) for m in EXTF]
   for left,right in pairs:
    s=g[g.method.isin([left,right])].groupby(['method','fold','case']).mae_mm.mean();p=pd.concat([s.loc[left].rename('left'),s.loc[right].rename('right')],axis=1);assert not p.isna().any().any();delta=p.left-p.right;rng=np.random.default_rng(20261103);bs=np.zeros(10000)
    for fold,z in delta.groupby(level='fold'):
     v=z.to_numpy();bs+=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)*len(v)/len(p)
    low,high=np.quantile(bs,[.025,.975]);boot.append(dict(task=task,scope=scope,horizon=h,left=left,right=right,cases=len(p),delta_mae_mm=delta.mean(),ci_low=low,ci_high=high,improved_cases=int((delta<0).sum())))
 pd.DataFrame(boot).to_csv(OUT/'bootstrap.csv',index=False)
 old=pd.read_csv(SOURCE.parent/'summary.csv');old=old[(old.protocol=='fusion')&old.task.isin(['current','future'])].copy();old['method']='fusion_'+old.group;old['horizon']=old.horizon.fillna(0).astype(int)
 reproduced=summary[summary.scope=='original_fusion'].merge(old,on=['method','task','horizon'],suffixes=('','_old'));reproduced=reproduced[reproduced.region==reproduced.subset];assert len(reproduced)==len(old);np.testing.assert_allclose(reproduced.mae_mm,reproduced.mae,atol=1e-12,rtol=0);reproduced.to_csv(OUT/'original_fusion_reproduction.csv',index=False)
 pd.read_csv(SUITE/'summary/input_parameter_budget.csv').to_csv(OUT/'input_parameter_budget_reference.csv',index=False)
 display={'fusion_A':'A: independent monitor + ODE + fusion','fusion_B':'B: current-only adaptation + fusion','fusion_C':'C: frozen-ODE adaptation + fusion','fusion_D':'Proposed D: joint adaptation + fusion','sun2020_lstm_module':'Sun et al. (2020) LSTM module','Kim_TF_raw_ensemble':'Kim et al. (2026) raw Transformer ensemble','Kim_TF_cubic_mean_NASA':'Kim et al. (2026) cubic mean (VB extension)','gpr':'GPR','svr':'SVR','sensor12_100':'Sensor top12 MLP100','CT100':'Condition/time MLP100','Zhang_PINN_top12':'Zhang PINN (NASA adaptation)','sun_pigpr_causal_nasa':'Sun et al. (2024) PI-GPR (causal adaptation)','chen2025_pidggcn_causal_paper_printed':'Chen et al. (2025) PIDGGCN (printed)','chen2025_pidggcn_causal_calculus_corrected':'Chen et al. (2025) PIDGGCN (corrected)','persistence_sensor100':'Persistence','linear2_sensor100':'Two-point linear','linear2_nonnegative_sensor100':'Two-point nonnegative linear'}
 pd.DataFrame([dict(method=m,display_name=display.get(m,m)) for m in sorted(set(cur.method)|set(future.method))]).to_csv(OUT/'method_labels.csv',index=False)
 audit={'original_summary_rows_reproduced':len(reproduced),'source':str(SOURCE),'source_sha256':sha(SOURCE),'protocol_sha256':sha(PAPER/'TABLE_PROTOCOL.md'),'no_training':True,'proposed_method':'fusion_D','current_original_targets_per_seed':128,'current_labeled_initial_exclusions':138-128,'future_original_targets_per_seed':{'1':114,'2':100,'3':86},'targets_dt_origin_truth_verified':True,'seeds_coverage_identical':True,'deterministic_methods_not_replicated':True,'future_methods':sorted(future.method.unique()),'future_counts':pd.DataFrame(coverage).query("method=='fusion_D'").to_dict('records')}
 (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
 print(summary[(summary.region=='overall')&summary.scope.isin(['current_common128','all_methods_common'])].to_string(index=False))
if __name__=='__main__':main()









