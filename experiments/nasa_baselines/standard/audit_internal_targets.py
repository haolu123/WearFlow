"""Independent final-output target/schema audit; does not alter shared results."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent


def main():
    path = ROOT/'internal/results/final_predictions.csv.gz'
    df = pd.read_csv(path)
    assert {'target_run','horizon','dt'} <= set(df.columns)
    assert set(df.arm) == {'current_only','local_only','integral_only','both'}
    assert set(df.seed) == set(range(5))
    assert set(df.task) == {'current','h1','h2','h3','oracle_h1','oracle_h2','oracle_h3','persistence_h1','persistence_h2','persistence_h3'}
    assert np.isfinite(df.pred_mm).all() and np.isfinite(df.target_mm).all()
    checks=[]
    future_refs=pd.read_csv(ROOT/'references/future_predictions.csv')
    for fold in ('fold_a','fold_b'):
        with np.load(ROOT/'data'/fold/'final/arrays.npz') as archive:
            a=dict(archive)
        lookup={(int(c),int(r)):i for i,(c,r) in enumerate(zip(a['case'],a['run']))}
        assert len(lookup)==len(a['case'])
        expected_current={(int(a['case'][i]),int(a['run'][i])) for i in np.flatnonzero(a['test_mask'])}
        expected_future={(int(a['case'][origin]),int(a['run'][origin]),int(a['run'][target]),int(h)):(float(a['y_mm'][target]),float(a['time'][target]-a['time'][origin])) for origin,target,h in a['test_pairs']}
        for (arm,seed), group in df[df.fold==fold].groupby(['arm','seed']):
            current=group[group.task=='current']
            assert len(current)==len(expected_current)
            assert set(zip(current.case,current.run))==expected_current
            for row in current.itertuples():
                i=lookup[row.case,row.run]
                assert abs(row.target_mm-a['y_mm'][i])<1e-12
                assert row.origin_run==row.run
                assert row.target_run==row.run and row.horizon==0 and row.dt==0
            for prefix in ('','oracle_','persistence_'):
                future=group[group.task.isin([prefix+'h1',prefix+'h2',prefix+'h3'])]
                keys=[(int(row.case),int(row.origin_run),int(row.run),int(row.task[-1])) for row in future.itertuples()]
                assert len(keys)==len(set(keys))==len(expected_future)
                assert set(keys)==set(expected_future)
                for key,row in zip(keys,future.itertuples()):
                    truth,dt=expected_future[key]
                    assert dt>0 and abs(row.target_mm-truth)<1e-12
                    assert row.target_run==key[2] and row.horizon==key[3]
                    assert abs(row.dt-dt)<1e-12
                if not prefix:
                    ref=future_refs[(future_refs.fold==fold)&(future_refs.seed==seed)]
                    refkeys=set(zip(ref.case,ref.origin_run,ref.target_run,ref.horizon))
                    assert set(keys)==refkeys
                    for row in ref.itertuples():
                        truth,dt=expected_future[row.case,row.origin_run,row.target_run,row.horizon]
                        assert abs(row.dt-dt)<1e-12 and abs(row.target_mm-truth)<1e-12
            checks.append(dict(fold=fold,arm=arm,seed=int(seed),current_targets=len(current),future_targets=len(expected_future),reference_keys_exact=True))
    for (arm,seed),g in df[df.task=='current'].groupby(['arm','seed']):
        assert len(g)==138
    result=dict(status='passed',checks=checks,rows=len(df),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        audit='independent exact target-key, target-label, actual-time comparison',
        current_per_arm_seed=138,estimated_and_oracle_and_persistence_distinct=True,
        future_schema={'task':'filter h1/h2/h3 for main estimated future','target_run':'explicit field equals run and exported target key','horizon':'explicit field equals task suffix and exported test_pairs horizon','dt':'explicit field equals original exported target_time minus origin_time'},
        explicit_target_run_horizon_dt_checked=True,
        current_target_run_equals_run_and_horizon_dt_zero=True)
    (OUT/'internal_targets_audit.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'status':'passed','groups':len(checks),'rows':len(df)}))


if __name__=='__main__':main()
