"""Postprocess only: no training, no execution identity changes.

Native model coordinate is original run/cut index. Future VB ensemble mean is
an evaluation extension; KDE applies exclusively to threshold-arrival times.
"""
import argparse,json,time,hashlib,inspect
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from method import MonotoneCubic,kde_mode
from raw_data import ROOT
OUT=ROOT/'formal'

def aggregate(a,ids,member_predictions,fold,seed,epoch):
    lookup={int(i):k for k,i in enumerate(ids)};member_predictions=np.asarray(member_predictions)
    assert member_predictions.shape==(10,len(ids))
    current=[];future=[];rul=[];fit_cache={}
    for i in np.flatnonzero(a['test_mask']):
        assert int(i) in lookup
        current.append(dict(method='Kim_TF_raw_ensemble',fold=fold,seed=seed,epoch=epoch,case=int(a['case'][i]),run=int(a['run'][i]),target_mm=float(a['y_mm'][i]),pred_mm=float(member_predictions[:,lookup[i]].mean()),members=10))
    for i,j,h in a['test_pairs']:
        base=dict(method='Kim_TF_cubic_mean_NASA',fold=fold,seed=seed,epoch=epoch,case=int(a['case'][i]),origin_run=int(a['run'][i]),target_run=int(a['run'][j]),horizon=int(h),dt=float(a['time'][j]-a['time'][i]),coordinate='original_cut_run',target_mm=float(a['y_mm'][j]))
        if i not in fit_cache:
            hist=[int(k) for k in ids if a['case'][k]==a['case'][i] and a['run'][k]<=a['run'][i]]
            hist.sort(key=lambda k:a['run'][k]);curves=[];errors=[]
            for m in range(10):
                try:curves.append(MonotoneCubic().fit(a['run'][hist],member_predictions[m,[lookup[k] for k in hist]]));errors.append('')
                except (ValueError,RuntimeError) as error:curves.append(None);errors.append(str(error))
            fit_cache[i]=(hist,curves,errors)
        hist,curves,errors=fit_cache[i];values=np.array([c.predict(a['run'][j]) if c else np.nan for c in curves]);valid=np.isfinite(values)
        base.update(history_records=len(hist),valid_members=int(valid.sum()),complete_ensemble=bool(valid.all()),pred_mm=float(values.mean()) if valid.all() else np.nan,
            negative_target_derivative_members=sum(c is not None and c.derivative(a['run'][j]) < -1e-8 for c in curves),failure='; '.join(sorted(set(e for e in errors if e))))
        future.append(base)
        # True future VB is supplied ONLY as a diagnostic threshold, never fitting.
        crossings=np.array([c.crossing(a['y_mm'][j]) if c else np.nan for c in curves]);good=np.isfinite(crossings)
        for bw in ['scott','literal_eq9']:
            mode=kde_mode(crossings,bw)
            rul.append(dict(fold=fold,seed=seed,epoch=epoch,case=base['case'],origin_run=base['origin_run'],target_run=base['target_run'],horizon=int(h),threshold_mm=float(a['y_mm'][j]),target_arrival_cut=float(a['run'][j]),pred_arrival_cut=float(mode),pred_remaining_cuts=float(mode-a['run'][i]),bandwidth=bw,valid_members=int(good.sum()),complete_ensemble=bool(good.all()),early_crossing_members=int(np.sum(crossings[good]<a['run'][i])),task='true_future_VB_threshold_time_diagnostic'))
    return pd.DataFrame(current),pd.DataFrame(future),pd.DataFrame(rul)

def group_report(fold,seed,epochs):
    dest=ROOT/'reports';group=dest/fold/f'ensemble_{seed}';group.mkdir(parents=True,exist_ok=True)
    a=dict(np.load(ROOT.parent/'data'/fold/'final/arrays.npz'))
    paths=[OUT/fold/f'ensemble_{seed}'/f'member_{m}' for m in range(10)]
    assert all((p/'completed.json').exists() for p in paths)
    semantic=hashlib.sha256(inspect.getsource(aggregate).encode()+(ROOT/'method.py').read_bytes()).hexdigest()
    started=time.perf_counter();final=None
    for count,epoch in enumerate(epochs,1):
        marker=group/f'epoch_{epoch:03d}_report.json';files=[p/f'epoch_{epoch:03d}_predictions.npz' for p in paths]
        identity=dict(semantic_hash=semantic,inputs=[hashlib.sha256(p.read_bytes()).hexdigest() for p in files])
        names=['current','future','threshold_time'];tables=None
        if marker.exists() and json.loads(marker.read_text()).get('identity')==identity:
            if epoch==100:tables=tuple(pd.read_csv(group/f'epoch_{epoch:03d}_{name}.csv.gz') for name in names)
        else:
            tick=time.perf_counter();packs=[dict(np.load(p)) for p in files];ids=packs[0]['record_indices'];assert all(np.array_equal(ids,p['record_indices']) for p in packs)
            tables=aggregate(a,ids,np.stack([p['pred_mm'] for p in packs]),fold,seed,epoch)
            for name,table in zip(names,tables):table.to_csv(group/f'epoch_{epoch:03d}_{name}.csv.gz',index=False,compression='gzip')
            marker.write_text(json.dumps(dict(identity=identity,seconds=time.perf_counter()-tick,epoch=epoch),indent=2))
        if epoch==100:final=tables
        (group/'progress.json').write_text(json.dumps(dict(completed_epochs=count,total_epochs=len(epochs),latest_epoch=epoch,elapsed_seconds=time.perf_counter()-started)))
    return fold,seed,final,time.perf_counter()-started

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--all-epochs',action='store_true');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
    dest=ROOT/'reports';dest.mkdir(exist_ok=True);started=time.perf_counter();final=[[],[],[]]
    epochs=list(range(1,101)) if args.all_epochs else [100]
    progress=dest/('all_epochs_status.json' if args.all_epochs else 'final_epoch_status.json');done=0
    progress.write_text(json.dumps(dict(status='running',completed_groups=0,total_groups=10,epochs_per_group=len(epochs))))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(group_report,fold,seed,epochs) for fold in ['fold_a','fold_b'] for seed in range(5)]
        for job in as_completed(jobs):
            fold,seed,tables,seconds=job.result();done+=1
            for accumulator,table in zip(final,tables):accumulator.append(table)
            progress.write_text(json.dumps(dict(status='running',completed_groups=done,total_groups=10,epochs_per_group=len(epochs),elapsed_seconds=time.perf_counter()-started)))
            print(f'{fold}/ensemble_{seed} report complete {seconds:.1f}s',flush=True)
    for name,parts in zip(['current','future','threshold_time'],final):
        if parts:pd.concat(parts).to_csv(dest/f'{name}_predictions.csv.gz',index=False,compression='gzip')
    if final[0]:
        current=pd.concat(final[0]);current['ae']=(current.pred_mm-current.target_mm).abs()
        cases=current.groupby(['fold','case','seed']).ae.mean().reset_index();cases.to_csv(dest/'current_case_seed.csv',index=False)
        mean=cases.groupby(['fold','case']).ae.mean();jsondata=dict(current_case_macro=float(mean.mean()),completed_ensemble_groups=len(final[0]),report_seconds=time.perf_counter()-started)
        future=pd.concat(final[1]);future['ae']=(future.pred_mm-future.target_mm).abs()
        coverage=future.groupby(['fold','seed','horizon']).agg(total_pairs=('case','size'),complete_pairs=('complete_ensemble','sum'),mean_valid_members=('valid_members','mean')).reset_index()
        coverage.to_csv(dest/'future_coverage.csv',index=False)
        fc=future[future.complete_ensemble].groupby(['fold','case','seed','horizon']).ae.mean().reset_index();fc.to_csv(dest/'future_case_seed.csv',index=False)
        fm=fc.groupby(['fold','case','horizon']).ae.mean().reset_index();fm.groupby('horizon').ae.agg(['mean','count']).to_csv(dest/'future_summary.csv')
        threshold=pd.concat(final[2]);threshold['absolute_cut_error']=(threshold.pred_arrival_cut-threshold.target_arrival_cut).abs()
        threshold.groupby(['fold','seed','bandwidth','horizon']).agg(total_pairs=('case','size'),complete_pairs=('complete_ensemble','sum'),mean_valid_members=('valid_members','mean')).to_csv(dest/'threshold_coverage.csv')
        tc=threshold[threshold.complete_ensemble].groupby(['bandwidth','fold','case','seed','horizon']).absolute_cut_error.mean().reset_index()
        tc.to_csv(dest/'threshold_case_seed.csv',index=False)
        jsonwrite=dest/'status.json';jsonwrite.write_text(json.dumps(jsondata,indent=2));print(json.dumps(jsondata))
    else:print('No complete 10-member ensemble available; no partial ensemble reported')
    progress.write_text(json.dumps(dict(status='complete',completed_groups=done,total_groups=10,epochs_per_group=len(epochs),elapsed_seconds=time.perf_counter()-started)))
if __name__=='__main__':main()
