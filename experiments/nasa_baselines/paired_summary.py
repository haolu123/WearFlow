"""Paired case bootstrap: seed-averaged errors, fold-stratified case resampling."""
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent/'summary'
KEYS=['fold','seed','case','origin_run','target_run','horizon']
def contrast(df,left,right,task,horizon=0):
 z=df[df.method.isin([left,right])].copy();assert np.isfinite(z.pred_mm).all()
 if task.startswith('future'):
  keys=set(map(tuple,z[z.method==left][KEYS].to_numpy()))&set(map(tuple,z[z.method==right][KEYS].to_numpy()));z=z[[tuple(k) in keys for k in z[KEYS].to_numpy()]]
  l=z[z.method==left];r=z[z.method==right];j=l.merge(r,on=KEYS,validate='one_to_one',suffixes=('_left','_right'));assert len(j)==len(l)==len(r)
  np.testing.assert_allclose(j.target_mm_left,j.target_mm_right,atol=1e-12,rtol=0);np.testing.assert_allclose(j.dt_left,j.dt_right,atol=1e-12,rtol=0)
 z['ae']=abs(z.pred_mm-z.target_mm);c=z.groupby(['method','fold','case','seed']).ae.mean().groupby(['method','fold','case']).mean();p=pd.concat([c.loc[left].rename('left'),c.loc[right].rename('right')],axis=1);assert not p.isna().any().any() and len(p)>0
 if task=='current':assert len(p)==14
 d=(p.left-p.right).to_numpy();rng=np.random.default_rng(20261103);bs=np.zeros(10000)
 for fold,g in p.groupby(level='fold'):
  v=(g.left-g.right).to_numpy();bs+=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)*len(v)/len(p)
 lo,hi=np.quantile(bs,[.025,.975]);return dict(task=task,horizon=horizon,left=left,right=right,case_count=len(p),paired_seed_targets=len(z[z.method==left]),delta_mae_mm=d.mean(),ci_low=lo,ci_high=hi,improved_cases=int((d<0).sum()))
def main():
 rows=[];cur=pd.read_csv(ROOT/'current_predictions.csv')
 for method in ['gpr','svr','Zhang_PINN_top12','sun_pigpr_causal_nasa','CT100','Kim_TF_raw_ensemble']+sorted(m for m in cur.method.unique() if m.startswith('chen2025_pidggcn')):rows.append(contrast(cur,'sensor12_100',method,'current'))
 abcd=pd.read_csv(ROOT.parent/'references/ours_current_predictions.csv')
 for right in ['current_only_extra100','frozen_ode_future_extra100']:rows.append(contrast(abcd,'joint_future_extra100',right,'current'))
 f=pd.read_csv(ROOT/'future_predictions.csv');f=f[np.isfinite(f.pred_mm)]
 for h in [1,2,3]:
  raw=f[f.horizon==h];sets=[set(map(tuple,q[KEYS].to_numpy())) for _,q in raw.groupby('method')];common=set.intersection(*sets);g=raw[[tuple(x) in common for x in raw[KEYS].to_numpy()]]
  for left,right in [('independent_monitor100_ode500','sun2020_lstm_module'),('joint_future_extra100','current_only_extra100'),('joint_future_extra100','frozen_ode_future_extra100'),('independent_monitor100_ode500','Kim_TF_cubic_mean_NASA')]:rows.append(contrast(g,left,right,'future_all_methods_common',h))
  rows.append(contrast(raw,'independent_monitor100_ode500','Kim_TF_cubic_mean_NASA','future_pairwise_common',h))
 out=pd.DataFrame(rows);out.to_csv(ROOT/'paired_case_bootstrap.csv',index=False);print(out.to_string(index=False))
if __name__=='__main__':main()



