"""Current+initial versus current+delta, with same-dimensional duplicate control."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..config import project_path, write_json, file_hash
from .run_sensor_mechanisms import context, Net, fit, predict_raw, OLD

ROOT=project_path('wear_monitoring_dynamics/outputs/initial_context_20261012')
PRIOR=ROOT.with_name('sensor_mechanisms_20261012')/'step23_controlled'
METHODS=['current_initial','current_delta','current_duplicate']


def initial_context(current,case,run,valid):
    """No wear labels accepted; never give earlier records a future reference."""
    initial=torch.zeros_like(current);anchor=np.full(len(case),-1,dtype=int)
    for c in np.unique(case):
        ids=np.flatnonzero((case==c)&valid)
        if not len(ids):continue
        i=ids[np.argmin(run[ids])]
        usable=(case==c)&(run>=run[i])&valid
        initial[usable]=current[i];anchor[usable]=i
    assert (anchor[valid]>=0).all()
    assert (run[anchor[valid]]<=run[valid]).all()
    return initial,anchor


def build_inputs(c):
    t=c['table'];current=c['sensor'][:,:18]
    initial,anchor=initial_context(current,t['case'],t['run'],c['valid'])
    delta=current-initial
    return dict(current_initial=torch.cat([current,initial,c['ct']],dim=1),
        current_delta=torch.cat([current,delta,c['ct']],dim=1),
        current_duplicate=torch.cat([current,current,c['ct']],dim=1)),anchor


def make_model(seed):
    torch.manual_seed(seed);return Net(41,[64,32])


def initial_to_delta_weights(model):
    """Exact change of coordinates for the first affine layer; audit, not training."""
    other=make_model(0);other.load_state_dict(model.state_dict())
    with torch.no_grad():
        w=model.net[0].weight
        other.net[0].weight[:,:18]=w[:,:18]+w[:,18:36]
        other.net[0].weight[:,18:36]=-w[:,18:36]
    return other


def run_one(c,name,x,anchor,out,epochs):
    r=c['row'];scope=f'{r.fold}/{r.role}_{int(r.holdout) if pd.notna(r.holdout) else "all"}/seed_{int(r.seed)}'
    d=out/'runs'/scope/name;d.mkdir(parents=True,exist_ok=True)
    identity=dict(name=name,epochs=epochs,seed=int(r.seed),source_run=r.run_dir,split=c['split'],train_indices=c['train'].tolist(),
        selected_features=c['cfg']['monitor']['selected_features'],input_dim=41,hidden=[64,32],lr=.001,weight_decay=.0001,device='cpu',
        representation='[current,extra,condition,time]; shared train-fitted clipped z-score; delta=current_z-initial_z')
    if (d/'complete.json').exists():
        assert json.loads((d/'identity.json').read_text())==identity
        return d
    model=make_model(int(r.seed));initial=model.state_dict()
    torch.save(initial,d/'initial.pt')
    write_json(d/'identity.json',identity);write_json(d/'scalers.json',c['scalers'])
    pred=np.empty((epochs,len(anchor)),dtype=np.float32);hist=[]
    np.save(d/'epoch0.npy',predict_raw(model,x,c))
    def after(epoch,m,loss):
        p=predict_raw(m,x,c);pred[epoch-1]=p;row=dict(epoch=epoch,loss=loss)
        for label,ids in c['evals'].items():
            if not len(ids):continue
            error=p[ids]-c['table']['VB'][ids];cases=c['table']['case'][ids]
            row[label+'_mae']=float(np.mean([abs(error[cases==k]).mean() for k in np.unique(cases)]))
            row[label+'_mse']=float(np.mean([(error[cases==k]**2).mean() for k in np.unique(cases)]))
        hist.append(row)
    started=time.perf_counter();fit(model,x,c['y'],c['train'],epochs,after)
    assert np.isfinite(pred).all()
    pd.DataFrame(hist).to_csv(d/'history.csv',index=False)
    t=c['table'];np.savez_compressed(d/'epoch_predictions.npz',predictions=pred,case=t['case'],run=t['run'],target=t['VB'],valid=c['valid'],anchor=anchor)
    torch.save(model.state_dict(),d/'last.pt')
    loaded=make_model(int(r.seed));loaded.load_state_dict(torch.load(d/'last.pt',weights_only=True))
    np.testing.assert_allclose(predict_raw(loaded,x,c),pred[-1],atol=1e-7,rtol=1e-6)
    write_json(d/'complete.json',dict(epochs=epochs,seconds=time.perf_counter()-started,parameters=sum(p.numel() for p in model.parameters()),reload_verified=True))
    return d


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    out=ROOT.with_name(ROOT.name+'_smoke') if args.smoke else ROOT;epochs=3 if args.smoke else 500
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    snapshot=out/'runner_snapshot.py'
    if snapshot.exists():assert file_hash(snapshot)==file_hash(__file__)
    else:snapshot.write_bytes(Path(__file__).read_bytes())
    write_json(out/'protocol.json',dict(models=METHODS,epochs=epochs,hidden_dims=[64,32],parameters=4801,
        reused=dict(ct_baseline='ct_standard',current_only='sensor_standard'),reused_study=str(PRIOR),
        first_reference='first sensor_valid record per case, independent of VB; causal availability mask',
        preprocessing='reuse train-only clipping/scalers and 18-feature selection; initial uses SAME transform as current; delta=current_z-initial_z; no separate delta scaling/clipping',
        inputs='18 current + 18 extra + 4 condition + 1 time; no anchor VB, case ID or future records as inputs',
        optimizer='AdamW lr=.001 wd=.0001, full batch standardized VB MSE',initialization='same numerical seed and identical initial weights across all 3 new arms',
        main='fixed500; secondary epoch selected by mean of 7 inner holdout cases, seed0, per fold/method; final seeds0..4',
        duplicate='[current,current] has no new information, same dimensionality/parameter count as requested arms',
        equivalence='[x,x0] and [x,x-x0] invertibly related; first-layer function equivalence is audited, optimization/weight decay need not be invariant',
        logging='all epoch predictions/metrics, initial and last weights, no every-epoch checkpoints',source_sha256=file_hash(__file__)))
    mf=pd.read_csv(OLD/'reused_sensor_manifest.csv');prior=pd.read_csv(PRIOR/'study_manifest.csv')
    if args.smoke:mf=mf[(mf.fold=='fold_a')&(mf.role=='inner')].head(1)
    rows=[];anchors=[];audits=[]
    try:
        for r in mf.itertuples():
            c=context(r);xs,anchor=build_inputs(c);t=c['table'];valid=c['valid']
            np.testing.assert_allclose((xs['current_initial'][:,:18]-xs['current_initial'][:,18:36]).numpy(),xs['current_delta'][:,18:36].numpy(),atol=1e-7)
            for case in c['split']['train']+c['split']['outer_test_reserved']:
                ids=np.flatnonzero((t['case']==case)&valid);first=ids[np.argmin(t['run'][ids])]
                assert all(anchor[ids]==first)
                anchors.append(dict(fold=r.fold,role=r.role,holdout=None if pd.isna(r.holdout) else int(r.holdout),seed=int(r.seed),case=int(case),
                    anchor_run=int(t['run'][first]),anchor_time=float(t['time'][first]),anchor_vb=float(t['VB'][first])))
            model=make_model(int(r.seed));mapped=initial_to_delta_weights(model)
            with torch.no_grad():equiv=float((model(xs['current_initial'])-mapped(xs['current_delta'])).abs().max())
            assert equiv<2e-6
            # Appending/changing future records cannot affect a preceding prediction input.
            changed=c['sensor'][:,:18].clone()
            first_ids=np.unique(anchor[anchor>=0]);nonfirst=np.ones(len(anchor),bool);nonfirst[first_ids]=False
            changed[nonfirst]+=100
            initial2,anchor2=initial_context(changed,t['case'],t['run'],valid)
            assert np.array_equal(anchor,anchor2)
            torch.testing.assert_close(initial2,xs['current_initial'][:,18:36],atol=0,rtol=0)
            audits.append(dict(fold=r.fold,role=r.role,holdout=None if pd.isna(r.holdout) else int(r.holdout),seed=int(r.seed),equivalence_max_error=equiv,causal=True))
            for name in METHODS:
                d=run_one(c,name,xs[name],anchor,out,epochs)
                rows.append(dict(name=name,fold=r.fold,role=r.role,holdout=None if pd.isna(r.holdout) else int(r.holdout),seed=int(r.seed),run_dir=str(d),reused=False))
                pd.DataFrame(rows).to_csv(out/'study_manifest.csv',index=False)
                write_json(out/'status.json',dict(status='running',new_completed=sum(not x['reused'] for x in rows),new_planned=len(mf)*3,last=name))
                print(f'{sum(not x["reused"] for x in rows)}/{len(mf)*3} {r.fold} {r.role} {r.holdout} seed{r.seed} {name}',flush=True)
            for oldname,newname in [('ct_standard','ct_baseline'),('sensor_standard','current_only')]:
                q=prior[(prior.name==oldname)&(prior.fold==r.fold)&(prior.role==r.role)&(prior.seed==r.seed)]
                q=q[q.holdout.isna()] if pd.isna(r.holdout) else q[q.holdout==r.holdout]
                assert len(q)==1;d=Path(q.iloc[0].run_dir);identity=json.loads((d/'identity.json').read_text())
                assert identity['train_indices']==c['train'].tolist() and identity['split']==c['split']
                rows.append(dict(name=newname,fold=r.fold,role=r.role,holdout=None if pd.isna(r.holdout) else int(r.holdout),seed=int(r.seed),run_dir=str(d),reused=True))
        pd.DataFrame(rows).to_csv(out/'study_manifest.csv',index=False)
        pd.DataFrame(anchors).to_csv(out/'anchor_records.csv',index=False)
        write_json(out/'input_audit.json',dict(passed=True,splits=audits,all_new_models_parameters=4801))
        write_json(out/'status.json',dict(status='smoke_completed' if args.smoke else 'training_completed',new_runs=len(mf)*3,reused_runs=len(mf)*2,epochs=epochs))
    except Exception as e:write_json(out/'failure.json',dict(error=str(e)));raise


if __name__=='__main__':main()
