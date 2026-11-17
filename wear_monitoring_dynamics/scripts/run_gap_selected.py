"""Nested train-only gap/swing feature selection; paired CT vs CT+12/18."""
from pathlib import Path
from itertools import combinations
import json,time,hashlib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .explore_current_components import pair_metrics,scale_for,ACTIVE,POOLS
from .run_sensor_mechanisms import Net,fit

BASE=Path(__file__).resolve().parents[1]
SOURCE=BASE/'outputs/current_components_20261013/all_features.csv'
OUT=BASE/'outputs/gap_selected_training_20261013'
METHODS={'ct':0,'ct_top12':12,'ct_top18':18}

def rank_features(data,cases,features):
    train=data[data.case.isin(cases)];rows=[]
    for f in features:
        scale=scale_for(train,f);p=pair_metrics(train,f,list(combinations(sorted(cases),2)),scale)
        coverage=float(train[f].notna().mean());count=int(p.ratio.notna().sum()) if len(p) else 0
        score=float(p.ratio.median()) if count else np.nan
        eligible=bool(coverage>=.8 and count>=5 and scale>1e-10 and np.isfinite(score))
        rows.append(dict(feature=f,coverage=coverage,pairs=count,score=score,eligible=eligible))
    r=pd.DataFrame(rows).sort_values(['score','feature']);chosen=r[r.eligible].feature.head(18).tolist()
    assert len(chosen)==18
    return chosen,r

def inputs(data,cases,chosen):
    mask=data.case.isin(cases).values;train=data[mask]
    sensor=data[chosen].values.astype(float);med=np.nanmedian(sensor[mask],axis=0)
    sensor=np.where(np.isfinite(sensor),sensor,med)
    lo,hi=np.quantile(sensor[mask],[.005,.995],axis=0);sensor=np.clip(sensor,lo,hi)
    mu=sensor[mask].mean(0);sd=np.maximum(sensor[mask].std(0),1e-8);sensor=(sensor-mu)/sd
    cond=data[['DOC','feed']].values;cm=cond[mask].mean(0);cs=np.maximum(cond[mask].std(0),1e-8)
    intervals=[]
    for _,g in train.groupby('case'):
        ts=g.sort_values('run').time.values
        intervals.extend((ts[3:]-ts[:-3]).tolist())
    ts=float(np.median(intervals));assert ts>0
    ct=np.column_stack([(cond-cm)/cs,data.material==1,data.material==2,data.time/ts]).astype('float32')
    ym=float(train.VB.mean());ys=max(float(train.VB.std(ddof=0)),1e-8)
    y=torch.tensor(((data.VB.values-ym)/ys)[:,None],dtype=torch.float32)
    xs={name:torch.tensor(np.column_stack([sensor[:,:n],ct]) if n else ct,dtype=torch.float32) for name,n in METHODS.items()}
    state=dict(train_cases=sorted(cases),chosen=chosen,median=med.tolist(),low=lo.tolist(),high=hi.tolist(),sensor_mean=mu.tolist(),sensor_std=sd.tolist(),condition_mean=cm.tolist(),condition_std=cs.tolist(),time_scale=ts,wear_mean=ym,wear_std=ys)
    return xs,y,np.flatnonzero(mask),state

def macro(data,p,ids):
    e=abs(p[ids]-data.VB.values[ids]);c=data.case.values[ids]
    return float(np.mean([e[c==k].mean() for k in np.unique(c)]))

def main(out=None,selector=None,selection_description=None):
    OUT=out if out is not None else globals()['OUT']
    selector=selector if selector is not None else rank_features
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    raw=pd.read_csv(SOURCE);data=raw[raw.case.isin(ACTIVE)&raw.VB.notna()].reset_index(drop=True);features=list(raw.columns[7:])
    assert len(data)==137 and len(features)==99
    protocol=dict(epochs=500,final_seeds=list(range(5)),inner_seed=0,hidden=[64,32],optimizer='AdamW lr=.001 wd=.0001; full batch standardized VB MSE',
        selection=selection_description or 'Train-only median gap/swing across all within-train case pairs; coverage>=80%, >=5 informative pairs, scale>1e-10; ascending score then feature name; no rho filter or redundancy removal; top12 is prefix of top18',
        low_variation='Pair swing<=5% RMS of train within-case P90-P10 excluded; >=2 distinct observed VB points per side within overlap',
        caveat='No same-condition pairs exist within a training pool. Rankings mix conditions; held-out labels never rank features. Previously explored dataset, not independent confirmation.',
        arms=METHODS,inputs='CT is standardized DOC/feed, material one-hot(1,2), time / median train h3 interval; sensor appended before CT',
        preprocessing='Train labeled records only: median imputation, .5/99.5% winsorization, zscore. No case ID or initial VB. All arms same137 labeled records; no fallback.',
        primary='fixed500',secondary='per fold/arm epoch minimizing mean seven inner holdout case MAEs; final seed0..4',
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    data[['case','run','VB','time']].to_csv(OUT/'records.csv',index=False)
    manifest=[];selected_epochs={};start=time.time()
    for fold,pool in POOLS.items():
        test=[c for c in ACTIVE if c not in pool];testids=np.flatnonzero(data.case.isin(test))
        for holdout in pool+[None]:
            role='inner' if holdout else 'final';cases=[c for c in pool if c!=holdout]
            scope=OUT/fold/(f'inner_{holdout}' if holdout else 'final');scope.mkdir(parents=True,exist_ok=True)
            chosen,ranks=selector(data,cases,features);ranks.to_csv(scope/'ranking.csv',index=False)
            xs,y,trainids,scalers=inputs(data,cases,chosen)
            (scope/'scalers.json').write_text(json.dumps(scalers,indent=2),encoding='utf-8')
            validids=np.flatnonzero(data.case==holdout) if holdout else testids
            if holdout is None:
                for name in METHODS:
                    curves=[pd.read_csv(OUT/fold/f'inner_{c}'/'seed0'/name/'history.csv').validation_mae.values for c in pool]
                    selected_epochs[fold+'_'+name]=int(np.argmin(np.mean(curves,axis=0))+1)
            for seed in ([0] if holdout else range(5)):
                for name,n in METHODS.items():
                    d=scope/f'seed{seed}'/name;d.mkdir(parents=True,exist_ok=True)
                    identity=dict(fold=fold,role=role,holdout=holdout,seed=seed,name=name,chosen=chosen[:n],train_cases=cases,train_indices=trainids.tolist(),epochs=500)
                    ep=selected_epochs.get(fold+'_'+name,500)
                    if (d/'complete.json').exists():
                        assert json.loads((d/'identity.json').read_text())==identity
                    else:
                        (d/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
                        torch.manual_seed(seed);model=Net(n+5,[64,32]);x=xs[name];pred=np.empty((500,len(data)),dtype=np.float32);hist=[];selected_state=None
                        torch.save(model.state_dict(),d/'initial.pt')
                        def after(epoch,m,loss):
                            nonlocal selected_state
                            with torch.no_grad():p=(m(x).numpy().ravel()*scalers['wear_std']+scalers['wear_mean'])
                            pred[epoch-1]=p;hist.append(dict(epoch=epoch,loss=loss,train_mae=macro(data,p,trainids),validation_mae=macro(data,p,validids)))
                            if epoch==ep:selected_state={k:v.detach().clone() for k,v in m.state_dict().items()}
                        fit(model,x,y,trainids,500,after)
                        assert np.isfinite(pred).all()
                        np.savez_compressed(d/'predictions.npz',predictions=pred)
                        pd.DataFrame(hist).to_csv(d/'history.csv',index=False);torch.save(model.state_dict(),d/'last.pt');torch.save(selected_state,d/'selected.pt')
                        for file,epoch in [('last.pt',500),('selected.pt',ep)]:
                            check=Net(n+5,[64,32]);check.load_state_dict(torch.load(d/file,weights_only=True))
                            with torch.no_grad():p=check(x).numpy().ravel()*scalers['wear_std']+scalers['wear_mean']
                            np.testing.assert_allclose(p,pred[epoch-1],atol=1e-7,rtol=1e-6)
                        (d/'complete.json').write_text(json.dumps(dict(epochs=500,parameters=sum(v.numel() for v in model.parameters()),reload_verified=True,selected_epoch=ep)),encoding='utf-8')
                    manifest.append(dict(**identity,run_dir=str(d),selected_epoch=ep))
                    pd.DataFrame(manifest).to_csv(OUT/'manifest.csv',index=False)
                    (OUT/'status.json').write_text(json.dumps(dict(completed=len(manifest),planned=72,seconds=time.time()-start)),encoding='utf-8')
                    print(f'{len(manifest)}/72 {fold} {role} {holdout} seed{seed} {name}',flush=True)
    (OUT/'selected_epochs.json').write_text(json.dumps(selected_epochs,indent=2),encoding='utf-8')
    rows=[];predrows=[]
    for r in manifest:
        if r['role']!='final':continue
        ids=np.flatnonzero(~data.case.isin(POOLS[r['fold']]))
        with np.load(Path(r['run_dir'])/'predictions.npz') as z:pred=z['predictions']
        for mode,epoch in [('fixed500',500),('inner_selected',r['selected_epoch'])]:
            p=pred[epoch-1]
            for case in sorted(data.iloc[ids].case.unique()):
                ci=ids[data.iloc[ids].case.values==case];e=p[ci]-data.VB.values[ci]
                rows.append(dict(fold=r['fold'],name=r['name'],seed=r['seed'],mode=mode,epoch=epoch,case=case,mae=float(abs(e).mean()),mse=float((e**2).mean()),n=len(ci)))
                for i in ci:predrows.append(dict(fold=r['fold'],name=r['name'],seed=r['seed'],mode=mode,case=case,run=int(data.iloc[i].run),target=float(data.iloc[i].VB),prediction=float(p[i])))
    scores=pd.DataFrame(rows);scores.to_csv(OUT/'case_metrics.csv',index=False);pd.DataFrame(predrows).to_csv(OUT/'test_predictions.csv',index=False)
    summary=scores.groupby(['mode','name']).agg(case_macro_mae=('mae','mean'),case_macro_mse=('mse','mean')).reset_index();summary['rmse']=np.sqrt(summary.case_macro_mse);summary.to_csv(OUT/'summary.csv',index=False)
    foldsummary=scores.groupby(['mode','fold','name']).mae.mean().reset_index();foldsummary.to_csv(OUT/'fold_summary.csv',index=False)
    comparisons=[];rng=np.random.default_rng(20261013)
    for mode in ['fixed500','inner_selected']:
        v=scores[scores['mode']==mode].groupby(['case','name']).mae.mean().unstack()
        for name in ['ct_top12','ct_top18']:
            delta=(v[name]-v.ct).values;boots=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            comparisons.append(dict(mode=mode,name=name,delta_mae=float(delta.mean()),improvement_percent=float(-100*delta.mean()/v.ct.mean()),cases_better=int((delta<0).sum()),ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975))))
    pd.DataFrame(comparisons).to_csv(OUT/'paired_comparison.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,mode in zip(axs,['fixed500','inner_selected']):
        v=scores[scores['mode']==mode].groupby(['case','name']).mae.mean().unstack();v.plot.bar(ax=ax);ax.set_title(mode);ax.set_ylabel('Case MAE (mm), mean of 5 seeds');ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'case_comparison.png',dpi=140);plt.close(fig)
    assert len(manifest)==72 and len(scores)==420 and len(predrows)==4110
    (OUT/'audit.json').write_text(json.dumps(dict(runs=72,epochs=36000,records=137,all_checkpoint_reload=True,paired_case_metric_rows=len(scores),test_prediction_rows=len(predrows),elapsed_seconds=time.time()-start),indent=2),encoding='utf-8')
    print(summary.to_string(index=False));print(pd.DataFrame(comparisons).to_string(index=False))

if __name__=='__main__':main()
