"""Read-only replay of frozen NASA Fusion A/D; no training or new selection."""
import os,sys,json,hashlib,shutil,zipfile
from pathlib import Path
sys.dont_write_bytecode=True
from wearflow_paths import ROOT as PROJECT
ROOT=PROJECT
SUITE=ROOT/'experiments/nasa_baselines'
SUP=SUITE/'paper_20261110/supplement_evolution_20261111'
OUT=SUITE/'paper_20261110/figure6'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(PROJECT))
os.chdir(PROJECT)
import numpy as np
import pandas as pd
import torch
from wear_monitoring_dynamics.training.adaptation_from100 import prepare
from wear_monitoring_dynamics.training.recursive_adaptation import layout,rollout
torch.set_num_threads(1)
ABCD=PROJECT/'wear_monitoring_dynamics/outputs/current12_abcd_from100_20261019'
tasks=json.loads((ABCD/'planned_tasks.json').read_text())
rows=[];hashes={};checks=[]
saved=pd.read_csv(ABCD/'predictions.csv.gz')
e2=pd.read_csv(SUP/'report_source_data/raw/E2_error_decomposition_pairs.csv')
for task in tasks:
    if task['group']!='D':continue
    model,b,t,parent=prepare(task)
    sc=parent['scalers'];st=layout(t,parent['split']['test'],'cpu')
    ck=ABCD/f"runs/{task['fold']}/seed{task['seed']}/D/last.pt"
    for name,path in [('monitor100',Path(task['monitor100'])),('ode',Path(task['ode_dir'])/'last.pt'),('D',ck)]:
        hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for arm in ['A','D']:
        if arm=='D':model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=True)['model'])
        model.eval()
        with torch.no_grad():
            raw=model.estimate(b,np.arange(len(t['case'])))
            post=rollout(model,b,raw,st,[(task['alpha'],True)])[:,0].numpy()*sc['wear'][1]+sc['wear'][0]
            monitor=raw[:,0].numpy()*sc['wear'][1]+sc['wear'][0]
        for seq in st['seqs']:
            assert np.diff(post[seq]).min()>=-1e-7
            for idx in seq:
                rows.append(dict(case=int(t['case'][idx]),run=int(t['run'][idx]),time=float(t['time'][idx]),true_vb_mm=float(t['VB'][idx]),model='Fusion '+arm,seed=int(task['seed']),current_pred_mm=float(post[idx]),sensor_valid=bool(t['sensor_valid'][idx]),fold=task['fold'],is_initial=idx in st['initial'],is_labeled=bool(np.isfinite(t['VB'][idx])),included_current_metric=idx in set(st['ids']),raw_monitor_pred_mm=float(monitor[idx]),alpha=task['alpha'],monotonic_clamp=True,source_epoch=0 if arm=='A' else 100))
        q=pd.DataFrame([r for r in rows if r['fold']==task['fold'] and r['seed']==task['seed'] and r['model']=='Fusion '+arm])
        existing=saved[(saved.fold==task['fold'])&(saved.seed==task['seed'])&(saved.group==arm)&(saved.task=='current')]
        existing=existing[existing['split'].str.startswith('test')]
        merged=existing.merge(q,left_on=['case','target_run'],right_on=['case','run'],validate='one_to_one')
        assert len(merged)==len(existing)>0
        error=abs(merged.predicted_vb_mm-merged.current_pred_mm).max();assert error<4e-6,error
        origins=e2[(e2.fold==task['fold'])&(e2.seed==task['seed'])&(e2.method=='fusion_'+arm)]
        origins=origins.merge(q,left_on=['case','origin_run'],right_on=['case','run'],validate='many_to_one')
        origin_error=abs(origins.origin_pred_mm-origins.current_pred_mm).max();assert origin_error<4e-6,origin_error
        checks.append(dict(fold=task['fold'],seed=task['seed'],model=arm,saved_current_max_error=float(error),e2_origin_max_error=float(origin_error),saved_current_rows=len(merged)))
    print(task['fold'],task['seed'],flush=True)
data=pd.DataFrame(rows).sort_values(['case','model','seed','time'])
assert not data.duplicated(['case','run','model','seed']).any()
data.to_csv(OUT/'NASA_case_current_trajectories.csv',index=False)
focus=data[data.case.isin([11,14])]
focus.to_csv(OUT/'NASA_case11_case14_current_trajectories.csv',index=False)
groups=['case','run','time','model','fold','is_initial','is_labeled','included_current_metric','sensor_valid']
avg=data.groupby(groups,dropna=False).agg(true_vb_mm=('true_vb_mm','first'),current_pred_mean_mm=('current_pred_mm','mean'),current_pred_sd_mm=('current_pred_mm','std'),n_seeds=('seed','nunique')).reset_index()
assert (avg.n_seeds==5).all()
avg.to_csv(OUT/'NASA_case_current_trajectories_mean_sd.csv',index=False)
future=e2[e2.method.isin(['fusion_A','fusion_D'])&e2.case.isin([11,14])].copy()
future['signed_error_mm']=future.pred_mm-future.target_mm
future.to_csv(OUT/'NASA_case11_case14_future_pairs.csv',index=False)
h3=future[future.horizon==3]
h3.to_csv(OUT/'NASA_case11_case14_h3_signed_errors.csv',index=False)
h3.groupby(['case','fold','method','origin_run','origin_time','target_run','target_time','horizon']).agg(true_vb_mm=('target_mm','first'),pred_mean_mm=('pred_mm','mean'),pred_sd_mm=('pred_mm','std'),signed_error_mean_mm=('signed_error_mm','mean'),signed_error_sd_mm=('signed_error_mm','std'),n_seeds=('seed','nunique')).reset_index().to_csv(OUT/'NASA_case11_case14_h3_mean_sd.csv',index=False)
for name in ['E8_rate_lines_and_seed_sd.csv','E8_observed_interval_average_points.csv']:
    shutil.copy2(SUP/'report_source_data/figure_points'/name,OUT/name)
coverage=data.groupby(['case','fold','model','seed']).agg(records=('run','size'),labeled=('is_labeled','sum'),initials=('is_initial','sum'),scored=('included_current_metric','sum'),first_run=('run','min'),last_run=('run','max')).reset_index()
coverage.to_csv(OUT/'coverage.csv',index=False)
(OUT/'audit.json').write_text(json.dumps({'checks':checks,'hashes':hashes,'rows':len(data),'focus_rows':len(focus),'no_training':True},indent=2))
print(coverage[coverage.case.isin([11,14])].to_string(index=False));print('TOTAL',len(data),len(focus));print('times',focus[['case','run','time']].drop_duplicates().to_string(index=False))
