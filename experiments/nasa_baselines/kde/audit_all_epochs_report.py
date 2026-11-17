"""Independent archive consistency audit, without retraining or refitting curves."""
import hashlib,inspect,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from report_raw_ensemble import aggregate
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports'

def main():
    start=time.perf_counter();semantic=hashlib.sha256(inspect.getsource(aggregate).encode()+(ROOT/'method.py').read_bytes()).hexdigest()
    assert len(list(REPORT.glob('fold_*/ensemble_*/epoch_*_report.json')))==1000
    assert len(list(REPORT.glob('fold_*/ensemble_*/epoch_*.csv.gz')))==3000
    final={k:[] for k in ['current','future','threshold_time']};groups=0;rowcounts=dict(current=0,future=0,threshold_time=0);maxerror=0.
    for fold in ['fold_a','fold_b']:
        a=dict(np.load(ROOT.parent/'data'/fold/'final/arrays.npz'));test=np.flatnonzero(a['test_mask']);pairs=a['test_pairs'];i,j,h=pairs.T
        for seed in range(5):
            group=REPORT/fold/f'ensemble_{seed}'
            for epoch in range(1,101):
                marker=json.loads((group/f'epoch_{epoch:03d}_report.json').read_text());assert marker['identity']['semantic_hash']==semantic
                paths=[ROOT/'formal'/fold/f'ensemble_{seed}'/f'member_{m}'/f'epoch_{epoch:03d}_predictions.npz' for m in range(10)]
                assert marker['identity']['inputs']==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
                packs=[dict(np.load(p)) for p in paths];ids=packs[0]['record_indices'];assert all(np.array_equal(ids,p['record_indices']) for p in packs)
                lookup={int(k):n for n,k in enumerate(ids)};mean=np.stack([p['pred_mm'] for p in packs]).mean(0)
                tables={name:pd.read_csv(group/f'epoch_{epoch:03d}_{name}.csv.gz') for name in final}
                for name,d in tables.items():
                    assert (d.fold==fold).all() and (d.seed==seed).all() and (d.epoch==epoch).all();rowcounts[name]+=len(d)
                c=tables['current'];assert len(c)==len(test) and not c.duplicated(['case','run']).any()
                np.testing.assert_array_equal(c.case,a['case'][test]);np.testing.assert_array_equal(c.run,a['run'][test]);np.testing.assert_allclose(c.target_mm,a['y_mm'][test],rtol=0,atol=1e-14)
                expected=mean[[lookup[int(k)] for k in test]];error=float(np.max(np.abs(c.pred_mm-expected)));maxerror=max(maxerror,error);np.testing.assert_allclose(c.pred_mm,expected,rtol=0,atol=1e-13)
                f=tables['future'];assert len(f)==len(pairs) and not f.duplicated(['case','origin_run','target_run','horizon']).any()
                for column,target in [('case',a['case'][i]),('origin_run',a['run'][i]),('target_run',a['run'][j]),('horizon',h)]:np.testing.assert_array_equal(f[column],target)
                np.testing.assert_allclose(f.target_mm,a['y_mm'][j],rtol=0,atol=1e-14);np.testing.assert_allclose(f.dt,a['time'][j]-a['time'][i],rtol=0,atol=1e-12)
                assert (f.dt>0).all();assert np.array_equal(f.complete_ensemble.values,f.valid_members.values==10);assert np.array_equal(f.pred_mm.notna().values,f.complete_ensemble.values)
                assert f.valid_members.between(0,10).all()
                for k,origin in enumerate(i):
                    n=int(np.sum((a['case'][ids]==a['case'][origin])&(a['run'][ids]<=a['run'][origin])))
                    assert f.history_records.iloc[k]==n
                    if n<3:assert not f.complete_ensemble.iloc[k]
                t=tables['threshold_time'];assert len(t)==2*len(pairs)
                for bw in ['scott','literal_eq9']:
                    q=t[t.bandwidth==bw]
                    for column,target in [('case',a['case'][i]),('origin_run',a['run'][i]),('target_run',a['run'][j]),('horizon',h)]:np.testing.assert_array_equal(q[column],target)
                    np.testing.assert_allclose(q.threshold_mm,a['y_mm'][j],rtol=0,atol=1e-14);assert np.array_equal(q.complete_ensemble.values,q.valid_members.values==10)
                    assert np.array_equal(q.pred_arrival_cut.notna().values,q.valid_members.values>0)
                    np.testing.assert_allclose(q.pred_remaining_cuts,q.pred_arrival_cut-q.origin_run,rtol=0,atol=1e-12,equal_nan=True)
                if epoch==100:
                    for name,d in tables.items():final[name].append(d)
                groups+=1
    for name,frames in final.items():
        left=pd.concat(frames,ignore_index=True);right=pd.read_csv(REPORT/f'{name}_predictions.csv.gz')
        keys=['fold','seed','case']+(['run'] if name=='current' else ['origin_run','target_run','horizon'])+(['bandwidth'] if name=='threshold_time' else [])
        pd.testing.assert_frame_equal(left.sort_values(keys).reset_index(drop=True),right.sort_values(keys).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=1e-12)
    result=dict(status='passed',markers=groups,csv_files=3000,input_archives_hashed=10000,row_counts=rowcounts,current_max_mean_error_mm=maxerror,final_epoch100_consistent=True,seconds=time.perf_counter()-start,
        scope='All report hashes/keys/targets/dt/member-mean/coverage/epoch100 consistency. Does not independently refit all cubics or recompute every KDE mode; no model training or inference repeated.')
    (REPORT/'all_epochs_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
