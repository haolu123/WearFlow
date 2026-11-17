"""Independent checks against pre-existing archives, plus identities and coverage."""
from wearflow_paths import ROOT as _WF_ROOT
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent;S=P.parent.parent
o=pd.read_csv(P/'oracle_pair_predictions.csv.gz')
old=pd.read_csv(S/'internal/results/final_predictions.csv.gz')
old=old[old.task.str.startswith('oracle_h') & old.arm.isin(['local_only','integral_only','both'])].copy()
old['model']='cold_'+old.arm
keys=['fold','case','seed','origin_run','target_run','horizon','model']
cmp=o[o.model.str.startswith('cold')].merge(old,on=keys,validate='one_to_one')
assert len(cmp)==len(old)==len(o[o.model.str.startswith('cold')])
diff=float(np.max(abs(cmp.oracle_future_mm-cmp.pred_mm)))
assert diff<4e-6,diff
original=pd.read_csv(str(_WF_ROOT / 'wear_monitoring_dynamics/outputs/current12_abcd_from100_20261019/oracle_summary.csv'))
s=pd.read_csv(P/'E1_oracle_summary.csv')
for group in ['A','D']:
    for h in [1,2,3]:
        val=s[(s.model=='warm_'+group)&(s.horizon==h)&(s.scope=='original_fusion')&(s.region=='overall')].abs_error.item()
        oldval=original[(original.group==group)&(original.horizon==h)&(original.subset=='overall')].mae.item()
        assert abs(val-oldval)<4e-6
for flag,expected in [('original_fusion',{1:114,2:100,3:86}),('common',{1:100,2:86,3:73})]:
    counts=o[o[flag]].groupby(['model','seed','horizon']).size()
    for (_,_,h),n in counts.items():assert n==expected[h]
e=pd.read_csv(P/'E2_summary.csv')
assert np.max(abs(e.future_mse-e.initial_mse-e.increment_mse-e.cross_term))<1e-12
assert np.isfinite(o[['oracle_future_mm','pred_increment','dt']]).all().all()
assert (o.dt>0).all()
figs=list(P.glob('*.png'));assert len(figs)==6
source=json.loads((P/'audit.json').read_text())['checkpoint_and_source_hashes']
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source.items())
result={'status':'passed','cold_archived_oracle_rows':len(cmp),'cold_archive_replay_max_abs_mm':diff,
        'warm_original_summary_reproduced':True,'original_and_common_coverage_verified':True,'source_hashes_unchanged':True,
        'aggregated_MSE_identity_max_error':float(np.max(abs(e.future_mse-e.initial_mse-e.increment_mse-e.cross_term))),
        'figures':len(figs),'training_started':False,
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file() and p.name!='verification.json'}}
(P/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='output_sha256'}))
