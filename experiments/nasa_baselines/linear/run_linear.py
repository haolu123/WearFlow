"""Fixed causal two-point linear trend; internal baseline, no training."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUITE = ROOT.parent


def extrapolate(history_vb, history_time, target_time, nonnegative=False):
    """Inputs contain exactly two past/current estimates, never future sensor."""
    assert len(history_vb) == len(history_time) == 2
    delta = history_time[1] - history_time[0]
    assert delta > 0 and target_time > history_time[1]
    slope = (history_vb[1] - history_vb[0]) / delta
    if nonnegative:
        slope = max(slope, 0.0)
    return history_vb[1] + slope * (target_time - history_time[1]), slope


def collect(data, monitor, fold, seed):
    rows = []
    for case in np.unique(data['case'][data['test_mask']]):
        idx = np.flatnonzero(data['test_mask'] & (data['case'] == case) & np.isfinite(data['y_mm']))
        idx = idx[np.argsort(data['time'][idx], kind='stable')]
        for position in range(1, len(idx) - 1):
            previous, origin = idx[position - 1:position + 1]
            for horizon in (1, 2, 3):
                if position + horizon >= len(idx):
                    continue
                target = idx[position + horizon]
                for nonnegative in (False, True):
                    pred, slope = extrapolate(monitor[[previous, origin]], data['time'][[previous, origin]], data['time'][target], nonnegative)
                    rows.append(dict(method='linear2_nonnegative_sensor100' if nonnegative else 'linear2_sensor100', fold=fold, seed=seed,
                        partition='test', initialization='estimated', case=int(case), previous_run=int(data['run'][previous]),
                        origin_run=int(data['run'][origin]), target_run=int(data['run'][target]), horizon=horizon,
                        previous_time=float(data['time'][previous]), origin_time=float(data['time'][origin]),
                        target_time=float(data['time'][target]), dt=float(data['time'][target]-data['time'][origin]),
                        history_previous_mm=float(monitor[previous]), history_origin_mm=float(monitor[origin]),
                        slope_mm_per_time=float(slope), target_mm=float(data['y_mm'][target]), pred_mm=float(pred)))
    return pd.DataFrame(rows)


def main():
    start = time.perf_counter()
    (ROOT/'status.json').write_text(json.dumps({'status':'running'}))
    assert np.isclose(extrapolate([.1,.2],[1.,3.],5.)[0], .3)
    assert np.isclose(extrapolate([.2,.1],[1.,3.],7.)[0], -.1)
    assert np.isclose(extrapolate([.2,.1],[1.,3.],7.,True)[0], .1)
    all_rows, hashes = [], {}
    for fold in ('fold_a','fold_b'):
        path = SUITE/'data'/fold/'final'/'arrays.npz'
        hashes[str(path.relative_to(SUITE))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as archive:
            data = dict(archive)
        for seed in range(5):
            refpath = SUITE/'references'/f'{fold}_seed{seed}_monitor.npz'
            hashes[str(refpath.relative_to(SUITE))] = hashlib.sha256(refpath.read_bytes()).hexdigest()
            with np.load(refpath) as reference:
                for key in ('case','run','time'):
                    np.testing.assert_array_equal(data[key], reference[key])
                monitor = reference['pred_mm'].copy()
            frame = collect(data, monitor, fold, seed)
            assert np.isfinite(frame.pred_mm).all()
            # Every origin-specific forecast remains unchanged when all later
            # sensor estimates and finite future labels are altered.
            for (case, origin_run), group in frame.groupby(['case','origin_run']):
                origin = np.flatnonzero((data['case']==case)&(data['run']==origin_run))[0]
                future = (data['case']==case)&(data['time']>data['time'][origin])
                altered = monitor.copy(); altered[future] += 12345.
                altered_data = dict(data); altered_data['y_mm'] = data['y_mm'].copy()
                valid_future = future & np.isfinite(altered_data['y_mm'])
                altered_data['y_mm'][valid_future] += 999.
                re = collect(altered_data, altered, fold, seed)
                re = re[(re.case==case)&(re.origin_run==origin_run)]
                np.testing.assert_array_equal(group.pred_mm.to_numpy(), re.pred_mm.to_numpy())
            all_rows.append(frame)
    predictions = pd.concat(all_rows, ignore_index=True)
    predictions['ae'] = abs(predictions.pred_mm-predictions.target_mm)
    keys=['fold','seed','case','origin_run','target_run','horizon']
    lstm = pd.read_csv(SUITE/'sun_lstm/results/future_predictions.csv')
    canonical = pd.read_csv(SUITE/'references/future_predictions.csv')
    coverage=[]
    for method, group in predictions.groupby('method'):
        assert not group.duplicated(keys).any()
        assert set(map(tuple,group[keys].to_numpy())) == set(map(tuple,lstm[keys].to_numpy()))
        joined=group.merge(canonical[keys+['dt','target_mm']],on=keys,validate='one_to_one',suffixes=('','_canonical'))
        assert len(joined)==len(group)
        np.testing.assert_allclose(joined.dt,joined.dt_canonical,atol=1e-12)
        np.testing.assert_allclose(joined.target_mm,joined.target_mm_canonical,atol=1e-12)
        for h, expected in ((1,110),(2,96),(3,83)):
            part=group[group.horizon==h]
            assert set(part.groupby('seed').size())=={expected}
            coverage.append(dict(method=method,horizon=h,pairs_per_seed=expected,cases=part.case.nunique()))
    predictions.to_csv(ROOT/'future_predictions.csv',index=False)
    case_seed=predictions.groupby(['method','fold','seed','case','horizon'],as_index=False).ae.mean()
    case_seed.to_csv(ROOT/'case_seed_metrics.csv',index=False)
    case_mean=case_seed.groupby(['method','fold','case','horizon'],as_index=False).ae.mean()
    rows=[]
    for (method,h),group in case_mean.groupby(['method','horizon']):
        for scope in ('overall','fold_a','fold_b'):
            part=group if scope=='overall' else group[group.fold==scope]
            rows.append(dict(method=method,horizon=h,scope=scope,case_macro_mae=part.ae.mean(),n_cases=len(part)))
    pd.DataFrame(rows).to_csv(ROOT/'summary.csv',index=False)
    audit=dict(status='passed',training_runs=0,hyperparameter_selection=False,
        formula_synthetic_tests=True,negative_prediction_allowed_raw=True,
        future_sensor_and_future_label_perturbation_invariance=True,history_same_labelled_positions_as_lstm=True,
        target_keys_exactly_match_lstm=True,dt_and_target_match_persistence=True,
        coverage=coverage,source_hashes=hashes,elapsed_seconds=time.perf_counter()-start)
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf8')
    (ROOT/'status.json').write_text(json.dumps({'status':'complete','rows':len(predictions),'elapsed_seconds':audit['elapsed_seconds']},indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':
    main()
