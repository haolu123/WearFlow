"""E1/E2/E8/E5A: read-only checkpoint evaluation, no optimizer or training."""
from wearflow_paths import ROOT as _WF_ROOT
import sys, json, time, hashlib
from pathlib import Path
sys.dont_write_bytecode=True
PROJECT=Path(str(_WF_ROOT / ''))
sys.path.insert(0,str(PROJECT))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wear_monitoring_dynamics.models.dynamics import WearDynamics
OUT=Path(__file__).resolve().parent
PAPER=OUT.parent
SUITE=PAPER.parent
ABCD=PROJECT/'wear_monitoring_dynamics/outputs/current12_abcd_from100_20261019'
torch.set_num_threads(1)
START=time.perf_counter()
HASHES={}
def digest(p):
    HASHES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
def readpt(p):
    digest(p)
    return torch.load(p,map_location='cpu',weights_only=True)
def tensor(x):return torch.as_tensor(x,dtype=torch.float32)
def dynamic(state):
    m=WearDynamics(32,20,True)
    m.load_state_dict({k.removeprefix('dynamics.'):v for k,v in state.items() if k.startswith('dynamics.')})
    m.eval()
    return m
def region_sets(q):
    return {'overall':q,'fold_a':q[q.fold=='fold_a'],'fold_b':q[q.fold=='fold_b'],
            'fold_a_in':q[(q.fold=='fold_a')&(q.target_mm<=.74)],'fold_a_out':q[(q.fold=='fold_a')&(q.target_mm>.74)]}
def aggregate(q, groups, values):
    case=q.groupby(groups+['fold','case','seed'],dropna=False)[values].mean().reset_index()
    seedmean=case.groupby(groups+['fold','case'],dropna=False)[values].mean().reset_index()
    out=seedmean.groupby(groups,dropna=False)[values].mean().reset_index()
    n=seedmean.groupby(groups,dropna=False).size().rename('cases').reset_index()
    return case,out.merge(n,on=groups)

source=PAPER/'tables/future_predictions.csv';digest(source)
full=pd.read_csv(source)
fusion=full[full.method.isin(['fusion_A','fusion_B','fusion_C','fusion_D'])].copy()
base=fusion[fusion.method=='fusion_A'].copy()
assert len(base)>0
all_oracle=[]; rate_rows=[]; obs_rates=[]; boundaries=[]; audits=[]
for fold in ['fold_a','fold_b']:
    data=SUITE/'data'/fold/'final'
    digest(data/'arrays.npz');digest(data/'metadata.json')
    a=dict(np.load(data/'arrays.npz'));meta=json.loads((data/'metadata.json').read_text())
    sc=meta['scalers'];mu,sd=sc['wear'];T=sc['time_scale']
    lookup={(int(c),int(r)):i for i,(c,r) in enumerate(zip(a['case'],a['run']))}
    train=a['train_pairs'];dttrain=a['time'][train[:,1]]-a['time'][train[:,0]]
    assert np.all(dttrain>0)
    lo,hi=np.quantile(dttrain,[1/3,2/3]);maximum=dttrain.max();minimum=dttrain.min()
    boundaries.append(dict(fold=fold,lower_tertile=lo,upper_tertile=hi,train_min=minimum,train_max=maximum,train_pairs=len(train),time_scale=T))
    p=a['test_pairs'];i,j,h=p.T
    pairs=pd.DataFrame(dict(fold=fold,case=a['case'][i],origin_run=a['run'][i],target_run=a['run'][j],horizon=h,
        origin_target_mm=a['y_mm'][i],target_mm=a['y_mm'][j],dt=a['time'][j]-a['time'][i],origin_index=i,target_index=j))
    assert np.isfinite(pairs[['origin_target_mm','target_mm','dt']]).all().all()
    assert (pairs.dt>0).all()
    pairs['condition']= [str(tuple(map(float,v))) for v in a['condition_raw'][i]]
    pairs['dt_bin']=np.where(pairs.dt<=lo,'short',np.where(pairs.dt<=hi,'medium','long'))
    pairs['dt_support']=np.where((pairs.dt<minimum)|(pairs.dt>maximum),'outside_train','within_train')
    common=set(map(tuple,base[(base.fold==fold)&base.in_common][['case','origin_run','target_run','horizon']].to_numpy()))
    original=set(map(tuple,base[base.fold==fold][['case','origin_run','target_run','horizon']].to_numpy()))
    keys=list(map(tuple,pairs[['case','origin_run','target_run','horizon']].to_numpy()))
    pairs['common']= [k in common for k in keys];pairs['original_fusion']=[k in original for k in keys]
    pairs['true_increment']=pairs.target_mm-pairs.origin_target_mm
    pairs['observed_average_rate']=pairs.true_increment/pairs.dt
    obs_rates.append(pairs.copy())
    for seed in range(5):
        b=readpt(ABCD/f'runs/{fold}/seed{seed}/B/last.pt')
        c=readpt(ABCD/f'runs/{fold}/seed{seed}/C/last.pt')
        d=readpt(ABCD/f'runs/{fold}/seed{seed}/D/last.pt')
        for k,v in b['model'].items():
            if k.startswith('dynamics.'):assert torch.equal(v,c['model'][k])
        assert np.array_equal(b['scalers']['wear'],sc['wear']) and b['scalers']['time_scale']==T
        states={'warm_A':b['model'],'warm_D':d['model']}
        for arm in ['local_only','integral_only','both']:
            ck=readpt(SUITE/f'internal/results/{fold}/final/seed_{seed}/{arm}/model.pt')
            assert np.array_equal(ck['scalers']['wear'],sc['wear']) and ck['scalers']['time_scale']==T
            states['cold_'+arm]=ck['final']
        for name,state in states.items():
            model=dynamic(state)
            with torch.no_grad():
                y=model.propagate(tensor((pairs.origin_target_mm.to_numpy()[:,None]-mu)/sd),tensor(a['X_ct'][i,:4]),tensor(pairs.dt.to_numpy()[:,None]/T)).numpy().ravel()*sd+mu
            q=pairs.copy();q['seed']=seed;q['model']=name;q['oracle_future_mm']=y;q['pred_increment']=y-q.origin_target_mm
            q['error']=q.pred_increment-q.true_increment;q['abs_error']=q.error.abs();q['squared_error']=q.error**2
            all_oracle.append(q)
            if name in ['warm_A','warm_D']:
                saved=fusion[(fusion.fold==fold)&(fusion.seed==seed)&(fusion.method=='fusion_'+name[-1])]
                ix=np.array([lookup[(int(r.case),int(r.origin_run))] for r in saved.itertuples()])
                with torch.no_grad():
                    replay=model.propagate(tensor((saved.origin_pred_mm.to_numpy()[:,None]-mu)/sd),tensor(a['X_ct'][ix,:4]),tensor(saved.dt.to_numpy()[:,None]/T)).numpy().ravel()*sd+mu
                maxerr=float(np.max(abs(replay-saved.pred_mm.to_numpy())))
                assert maxerr<4e-6,(fold,seed,name,maxerr)
                audits.append(dict(fold=fold,seed=seed,model=name,deploy_replay_max_abs=maxerr))
            # Rate visualization is restricted to condition-specific TRAINING VB support.
            fit=np.flatnonzero(a['train_mask']&np.isfinite(a['y_mm']))
            for condition in np.unique(a['condition_raw'][fit],axis=0):
                inds=fit[np.all(a['condition_raw'][fit]==condition,axis=1)]
                low,high=a['y_mm'][inds].min(),a['y_mm'][inds].max()
                grid=np.linspace(low,high,60)
                cond=np.tile(a['X_ct'][inds[0],:4],(len(grid),1))
                with torch.no_grad():rates=model.rhs(tensor((grid[:,None]-mu)/sd),tensor(cond)).numpy().ravel()*sd/T
                assert (rates>=0).all()
                rate_rows.extend(dict(fold=fold,seed=seed,model=name,condition=str(tuple(map(float,condition))),vb=float(v),rate=float(r),train_vb_min=float(low),train_vb_max=float(high),training_records=len(inds)) for v,r in zip(grid,rates))
    print('Evaluated '+fold,flush=True)
oracle=pd.concat(all_oracle,ignore_index=True)
oracle.to_csv(OUT/'oracle_pair_predictions.csv.gz',index=False)
rates=pd.DataFrame(rate_rows);rates.to_csv(OUT/'rate_curves.csv.gz',index=False)
obs=pd.concat(obs_rates);obs.to_csv(OUT/'observed_interval_rates.csv',index=False)
bounds=pd.DataFrame(boundaries);bounds.to_csv(OUT/'time_boundaries_training_only.csv',index=False)
summaries=[];case_tables=[]
for scope,selector in [('all_labeled_pairs',np.ones(len(oracle),bool)),('original_fusion',oracle.original_fusion),('external_common',oracle.common)]:
    for region,q in region_sets(oracle[selector]).items():
        cs,sm=aggregate(q,['model','horizon'],['abs_error','error','squared_error'])
        sm['scope']=scope;sm['region']=region;cs['scope']=scope;cs['region']=region
        summaries.append(sm);case_tables.append(cs)
summary=pd.concat(summaries);summary.to_csv(OUT/'E1_oracle_summary.csv',index=False)
case=pd.concat(case_tables);case.to_csv(OUT/'E1_case_seed.csv',index=False)
# Seed averaging precedes stratified paired case bootstrap.
rng=np.random.default_rng(20261111);boot=[]
for (scope,region,h),q in case[case.region=='overall'].groupby(['scope','region','horizon']):
    p=q.groupby(['fold','case','model']).abs_error.mean().unstack('model')
    for method,reference in [('warm_D','warm_A'),('cold_integral_only','cold_local_only'),('cold_both','cold_integral_only')]:
        delta=p[method]-p[reference]
        draws=sum(rng.choice(v.to_numpy(),(10000,len(v)),replace=True).sum(1) for _,v in delta.groupby(level=0))/len(delta)
        l,u=np.quantile(draws,[.025,.975]);boot.append(dict(scope=scope,horizon=h,method=method,reference=reference,delta=delta.mean(),ci_low=l,ci_high=u,cases=len(delta)))
pd.DataFrame(boot).to_csv(OUT/'E1_case_bootstrap.csv',index=False)
# E2 exact signed-error and squared-error decomposition, with case-equal aggregation.
fusion['initial_error']=fusion.origin_pred_mm-fusion.origin_target_mm
fusion['increment_error']=(fusion.pred_mm-fusion.origin_pred_mm)-(fusion.target_mm-fusion.origin_target_mm)
fusion['future_error']=fusion.pred_mm-fusion.target_mm
assert np.max(abs(fusion.future_error-fusion.initial_error-fusion.increment_error))<1e-12
for v in ['initial','increment','future']:
    fusion[v+'_mae']=fusion[v+'_error'].abs();fusion[v+'_mse']=fusion[v+'_error']**2
fusion['cross_term']=2*fusion.initial_error*fusion.increment_error
fusion['opposite_sign']=(fusion.initial_error*fusion.increment_error<0).astype(float)
assert np.max(abs(fusion.future_mse-fusion.initial_mse-fusion.increment_mse-fusion.cross_term))<1e-12
fusion.to_csv(OUT/'E2_error_decomposition_pairs.csv.gz',index=False)
decomp=[];decompcases=[]
values=['initial_error','increment_error','future_error','initial_mae','increment_mae','future_mae','initial_mse','increment_mse','future_mse','cross_term','opposite_sign']
for scope,q in [('original_fusion',fusion),('external_common',fusion[fusion.in_common])]:
    for region,z in region_sets(q).items():
        cs,sm=aggregate(z,['method','horizon'],values);sm['scope']=scope;sm['region']=region;cs['scope']=scope;cs['region']=region
        decomp.append(sm);decompcases.append(cs)
decomp=pd.concat(decomp);decomp.to_csv(OUT/'E2_summary.csv',index=False)
pd.concat(decompcases).to_csv(OUT/'E2_case_seed.csv',index=False)
correlations=[]
for (method,fold,c,seed,h),q in fusion.groupby(['method','fold','case','seed','horizon']):
    corr=q.initial_error.corr(q.increment_error) if len(q)>2 and q.initial_error.std()>1e-12 and q.increment_error.std()>1e-12 else np.nan
    correlations.append(dict(method=method,fold=fold,case=c,seed=seed,horizon=h,pairs=len(q),pearson=corr))
pd.DataFrame(correlations).to_csv(OUT/'E2_within_case_correlations.csv',index=False)
# E5A intervals selected exclusively from training pair durations.
duration=[]
for fold,b in bounds.set_index('fold').iterrows():
    q=fusion[fusion.fold==fold].copy()
    q['dt_bin']=np.where(q.dt<=b.lower_tertile,'short',np.where(q.dt<=b.upper_tertile,'medium','long'))
    q['dt_support']=np.where((q.dt<b.train_min)|(q.dt>b.train_max),'outside_train','within_train')
    duration.append(q)
duration=pd.concat(duration);duration.to_csv(OUT/'E5A_deployment_pairs.csv.gz',index=False)
time_summaries=[]
for scope,z in [('original_fusion',duration),('external_common',duration[duration.in_common])]:
    for grouping in ['dt_bin','dt_support']:
        for region,q in region_sets(z).items():
            if len(q)==0:continue
            cs,sm=aggregate(q,['method',grouping],['future_mae','future_error','dt'])
            sm=sm.rename(columns={grouping:'interval_group'});sm['grouping']=grouping;sm['scope']=scope;sm['region']=region
            time_summaries.append(sm)
pd.concat(time_summaries).to_csv(OUT/'E5A_summary.csv',index=False)
counts=duration.groupby(['method','fold','seed','dt_bin','dt_support']).size().rename('pairs').reset_index();counts.to_csv(OUT/'E5A_coverage.csv',index=False)
cs,sm=aggregate(duration,['method','horizon','dt_bin'],['future_mae','future_error','dt'])
cs.to_csv(OUT/'E5A_horizon_bin_case_seed.csv',index=False);sm.to_csv(OUT/'E5A_horizon_bin_summary.csv',index=False)
# Stored Figure3 allegation checked against actual source data.
figdata=pd.read_csv(PAPER/'figures/fig03_h1_increment_records.csv')
assert np.allclose(figdata.true_increment,figdata.target_mm-figdata.origin_target_mm)
pd.DataFrame(audits).to_csv(OUT/'checkpoint_replay_audit.csv',index=False)
audit=dict(status='passed',training_started=False,seconds=time.perf_counter()-START,checkpoint_replays=len(audits),
           A_B_C_rate_identical=True,figure3_negative_measured_increments=int((figdata.true_increment<0).sum()),
           e2_identity_max_error=float(np.max(abs(fusion.future_error-fusion.initial_error-fusion.increment_error))),
           checkpoint_and_source_hashes=HASHES)
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
print('Evaluation complete',audit['seconds'],flush=True)
