"""Independent arithmetic/key audit of final reports; never trains models."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'summary'

def case_mae(frame):
    errors = {}
    for row in frame.itertuples(index=False):
        key = (row.fold, int(row.case), int(row.seed))
        errors.setdefault(key, []).append(abs(row.pred_mm-row.target_mm))
    per_case = {}
    for (fold, case, seed), vals in errors.items():
        per_case.setdefault((fold, case), []).append(sum(vals)/len(vals))
    return sum(sum(v)/len(v) for v in per_case.values())/len(per_case)

def subset(frame, name):
    if name == 'overall': return frame
    if name in ('fold_a', 'fold_b'): return frame[frame.fold == name]
    a = frame[frame.fold == 'fold_a']
    return a[a.target_mm <= .74] if name == 'fold_a_in' else a[a.target_mm > .74]

def main():
    cur = pd.read_csv(OUT/'current_predictions.csv')
    future = pd.read_csv(OUT/'future_predictions.csv')
    csummary = pd.read_csv(OUT/'current_summary.csv')
    fsummary = pd.read_csv(OUT/'future_summary.csv')
    current_keys = ['fold', 'case', 'run']
    truth = pd.read_csv(ROOT/'references/current_predictions.csv')[current_keys+['target_mm']].drop_duplicates()
    for (method, seed), group in cur.groupby(['method','seed']):
        assert len(group) == 138 and not group.duplicated(current_keys).any()
        joined = group.merge(truth, on=current_keys, validate='one_to_one', suffixes=('','_ref'))
        assert len(joined) == 138
        np.testing.assert_allclose(joined.target_mm, joined.target_mm_ref, atol=1e-12, rtol=0)
        assert np.isfinite(group.pred_mm).all()
    for row in csummary.itertuples(index=False):
        q = subset(cur[cur.method == row.method], row.subset)
        assert abs(case_mae(q)-row.mae_mm) < 1e-12
    fk = ['fold','seed','case','origin_run','target_run','horizon']
    rawkim = pd.read_csv(ROOT/'kde/reports/future_predictions.csv.gz')
    kim = future[future.method == 'Kim_TF_cubic_mean_NASA']
    assert set(map(tuple, kim[fk].to_numpy())) == set(map(tuple, rawkim[fk].to_numpy()))
    assert kim.pred_mm.isna().sum() == rawkim.pred_mm.isna().sum()
    common_counts = {}
    for horizon in (1,2,3):
        g = future[(future.horizon == horizon) & np.isfinite(future.pred_mm)]
        sets = [set(map(tuple, q[fk].to_numpy())) for _, q in g.groupby('method')]
        common = set.intersection(*sets)
        common_counts[horizon] = len(common)
        for row in fsummary[fsummary.horizon == horizon].itertuples(index=False):
            q = g[g.method == row.method]
            if row.scope == 'all_methods_common':
                q = q[[tuple(k) in common for k in q[fk].to_numpy()]]
            q = subset(q, row.subset)
            assert abs(case_mae(q)-row.mae_mm) < 1e-12, row
    files = ['current_predictions.csv','future_predictions.csv','current_summary.csv','future_summary.csv','paired_case_bootstrap.csv']
    result = dict(status='passed', current_methods=int(cur.method.nunique()), current_targets_per_seed=138,
                  current_summary_rows=len(csummary), future_summary_rows=len(fsummary),
                  common_seed_target_rows=common_counts, kim_missing_forecasts=int(kim.pred_mm.isna().sum()),
                  scope='Independent case/seed arithmetic and target-key audit; bootstrap implementation reviewed separately.',
                  sha256={f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in files})
    (OUT/'final_independent_aggregation_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__': main()
