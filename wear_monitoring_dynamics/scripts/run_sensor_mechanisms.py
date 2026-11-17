"""Controlled capacity, warm-start and frozen-residual experiment.

Standalone runner leaves legacy execution code and checkpoints unchanged.
Every epoch is stored as a compact prediction array. Only inner validation
selects epochs/alpha. All training populations are identical within a split.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from ..config import project_path, write_json, file_hash, digest
from ..data.records import load_table
from ..data.feature_archive import attach_features
from ..data.sensor_quality import apply_quality
from ..data.preprocessing import fit_scalers, transform
from ..data.splits import make_split
from ..models.registry import build_model
from .diagnose_sensor_learning import ROOT, OLD, md

MODELS={
    'ct_small':dict(sensor=False,hidden=[32,16],kind='scratch'),
    'sensor_small':dict(sensor=True,hidden=[32,16],kind='scratch'),
    'ct_standard':dict(sensor=False,hidden=[64,32],kind='scratch'),
    'sensor_standard':dict(sensor=True,hidden=[64,32],kind='scratch'),
    'sensor_matched':dict(sensor=True,hidden=[50,25],kind='scratch'),
    'ct_continue':dict(sensor=False,hidden=[64,32],kind='warm'),
    'sensor_warm':dict(sensor=True,hidden=[64,32],kind='warm'),
    'resid_ct_in':dict(sensor=False,hidden=[32,16],kind='residual',target='in'),
    'resid_sensor_in':dict(sensor=True,hidden=[32,16],kind='residual',target='in'),
    'resid_ct_oof':dict(sensor=False,hidden=[32,16],kind='residual',target='oof'),
    'resid_sensor_oof':dict(sensor=True,hidden=[32,16],kind='residual',target='oof'),
}
ALPHAS=[0.,.25,.5,1.]
BASE_TABLE=None


class Net(nn.Module):
    def __init__(self,width,hidden):
        super().__init__();layers=[]
        for h in hidden:layers.extend([nn.Linear(width,h),nn.SiLU()]);width=h
        self.net=nn.Sequential(*layers,nn.Linear(width,1))

    def forward(self,x):return self.net(x)


def make_net(sensor,hidden,seed):
    torch.manual_seed(seed)
    return Net(23 if sensor else 5,hidden)


def warm_start(base,sensor,seed):
    model=make_net(sensor,[64,32],seed)
    with torch.no_grad():
        model.net[0].weight.zero_()
        model.net[0].weight[:,-5:].copy_(base.net[0].weight)
        model.net[0].bias.copy_(base.net[0].bias)
        for k in [2,4]:model.net[k].load_state_dict(base.net[k].state_dict())
    return model


def context(row):
    global BASE_TABLE
    directory=Path(row.run_dir)
    cfg=json.loads((directory/'provenance.json').read_text())['config']
    if BASE_TABLE is None:
        raw,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))
        BASE_TABLE=attach_features(raw,cfg['data']['feature_archive'])
    table=apply_quality(BASE_TABLE,cfg['data']['sensor_quality'])
    split=make_split(row.fold,row.role,cfg['experiment'].get('holdout'))
    scalers=fit_scalers(table,split['train'],cfg['data']['sensor_preprocessing'])
    old=json.loads((directory/'scalers.json').read_text())
    assert digest(scalers)==digest(old),'Scaler reproduction failed'
    values=transform(table,scalers);idx=cfg['monitor']['selected_indices']
    assert len(idx)==18
    ct=torch.tensor(np.column_stack([values['condition'],values['time']]),dtype=torch.float32)
    sx=values['sensor'][:,idx].copy();sx[~table['sensor_valid']]=0
    sensor=torch.cat([torch.tensor(sx,dtype=torch.float32),ct],dim=1)
    y=torch.tensor(values['target'],dtype=torch.float32)
    labeled=np.isfinite(table['VB']);valid=table['sensor_valid']
    train=np.flatnonzero(labeled&valid&np.isin(table['case'],split['train']))
    evals={k:np.flatnonzero(labeled&np.isin(table['case'],cases)) for k,cases in
           [('train',split['train']),('validation',split['validation']),('test',split['outer_test_reserved'])]}
    # Same frozen original CT fallback for ALL arms, even CT-only controls.
    fallback_path=cfg['monitor']['fallback_checkpoint']
    state=torch.load(fallback_path,map_location='cpu',weights_only=True)
    assert state['scalers']['train_cases']==scalers['train_cases']
    original=transform(table,state['scalers'])
    batch={k:torch.tensor(v,dtype=torch.float32) for k,v in original.items()}
    oldmodel=build_model(state['config']);oldmodel.load_state_dict(state['model']);oldmodel.eval()
    with torch.no_grad():fallback=oldmodel.estimate(batch,np.arange(len(labeled))).numpy().ravel()*state['scalers']['wear'][1]+state['scalers']['wear'][0]
    return dict(cfg=cfg,table=table,split=split,scalers=scalers,ct=ct,sensor=sensor,y=y,train=train,evals=evals,
                valid=valid,fallback=fallback,row=row)


def predict_raw(model,x,c,base_raw=None):
    with torch.no_grad():v=model(x).numpy().ravel()*c['scalers']['wear'][1]
    if base_raw is None:v=v+c['scalers']['wear'][0]
    else:v=v+base_raw
    v[~c['valid']]=c['fallback'][~c['valid']]
    return v


def fit(model,x,target,ids,epochs,after=None):
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    loss_history=[]
    for epoch in range(1,epochs+1):
        model.train();opt.zero_grad(set_to_none=True)
        loss=(model(x[ids])-target[ids]).square().mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
        loss.backward();opt.step();loss_history.append(float(loss.detach()))
        if after is not None:after(epoch,model,float(loss.detach()))
    return loss_history


def crossfit(c,out,epochs):
    """Nested leave-case-out baselines: no held-out label fits a scaler/model."""
    table=c['table'];result=np.full(len(table['case']),np.nan);rows=[]
    for holdout in c['split']['train']:
        train_cases=[v for v in c['split']['train'] if v!=holdout]
        seed=int(c['row'].seed)
        key=digest(dict(train_cases=train_cases,seed=seed,epochs=epochs,valid_keys=list(map(int,np.flatnonzero(c['valid'])))))[:16]
        directory=out/'crossfit_baselines'/key
        directory.mkdir(parents=True,exist_ok=True)
        scalers=fit_scalers(table,train_cases)
        v=transform(table,scalers)
        x=torch.tensor(np.column_stack([v['condition'],v['time']]),dtype=torch.float32)
        y=torch.tensor(v['target'],dtype=torch.float32)
        ids=np.flatnonzero(np.isfinite(table['VB'])&c['valid']&np.isin(table['case'],train_cases))
        model=make_net(False,[64,32],seed)
        if (directory/'complete.json').exists():
            state=torch.load(directory/'last.pt',weights_only=True);model.load_state_dict(state)
        else:
            losses=fit(model,x,y,ids,epochs)
            torch.save(model.state_dict(),directory/'last.pt')
            pd.DataFrame({'epoch':range(1,epochs+1),'loss':losses}).to_csv(directory/'history.csv',index=False)
            write_json(directory/'scalers.json',scalers)
            write_json(directory/'complete.json',dict(train_cases=train_cases,seed=seed,epochs=epochs,train_indices=ids.tolist()))
        held=np.flatnonzero((table['case']==holdout)&c['valid']&np.isfinite(table['VB']))
        assert holdout not in scalers['train_cases'] and not np.isin(table['case'][ids], [holdout]).any()
        with torch.no_grad():result[held]=model(x[held]).numpy().ravel()*scalers['wear'][1]+scalers['wear'][0]
        rows.append(dict(holdout=int(holdout),train_cases=train_cases,directory=str(directory),n=len(held)))
    assert np.isfinite(result[c['train']]).all()
    return result,rows


def run_arm(c,name,out,epochs,base=None,oof=None):
    spec=MODELS[name];r=c['row'];scope=f'{r.fold}/{r.role}_{int(r.holdout) if pd.notna(r.holdout) else "all"}/seed_{int(r.seed)}'
    directory=out/'runs'/scope/name;directory.mkdir(parents=True,exist_ok=True)
    x=c['sensor'] if spec['sensor'] else c['ct']
    residual=spec['kind']=='residual'
    if spec['kind']=='warm':model=warm_start(base,spec['sensor'],int(r.seed))
    else:model=make_net(spec['sensor'],spec['hidden'],int(r.seed))
    base_raw=None
    if residual:
        # Start from exactly the base, not a random correction.
        nn.init.zeros_(model.net[-1].weight);nn.init.zeros_(model.net[-1].bias)
        base_raw=predict_raw(base,c['ct'],c)
        ref=oof if spec['target']=='oof' else base_raw
        target=torch.tensor(((c['table']['VB']-ref)/c['scalers']['wear'][1])[:,None],dtype=torch.float32)
    else:target=c['y']
    epoch0=predict_raw(model,x,c,base_raw)
    if spec['kind'] in ['warm','residual']:
        assert np.max(abs(epoch0-predict_raw(base,c['ct'],c)))<2e-6,'Baseline identity failed'
    ident=dict(name=name,spec=spec,epochs=epochs,seed=int(r.seed),split=c['split'],source_run=r.run_dir,
               train_indices=c['train'].tolist(),device='cpu',optimizer='AdamW reset',lr=.001,weight_decay=.0001)
    if (directory/'complete.json').exists():
        prior=json.loads((directory/'identity.json').read_text());assert prior==ident
        model.load_state_dict(torch.load(directory/'last.pt',weights_only=True))
        return model,directory
    write_json(directory/'identity.json',ident);write_json(directory/'scalers.json',c['scalers'])
    np.save(directory/'epoch0.npy',epoch0)
    n=len(epoch0);predictions=np.empty((epochs,n),np.float32);hist=[]
    def after(epoch,model,loss):
        raw=predict_raw(model,x,c,base_raw);predictions[epoch-1]=raw
        row=dict(epoch=epoch,loss=loss)
        for label,ids in c['evals'].items():
            if not len(ids):continue
            e=raw[ids]-c['table']['VB'][ids]
            cases=c['table']['case'][ids]
            row[label+'_mae']=np.mean([abs(e[cases==case]).mean() for case in np.unique(cases)])
            row[label+'_mse']=np.mean([(e[cases==case]**2).mean() for case in np.unique(cases)])
        hist.append(row)
    start=time.perf_counter();fit(model,x,target,c['train'],epochs,after)
    pd.DataFrame(hist).to_csv(directory/'history.csv',index=False)
    np.savez_compressed(directory/'epoch_predictions.npz',predictions=predictions,case=c['table']['case'],run=c['table']['run'],
                        target=c['table']['VB'],valid=c['valid'],base=np.array([]) if base_raw is None else base_raw)
    torch.save(model.state_dict(),directory/'last.pt')
    reloaded=make_net(spec['sensor'],spec['hidden'],int(r.seed));reloaded.load_state_dict(torch.load(directory/'last.pt',weights_only=True))
    assert np.allclose(predict_raw(reloaded,x,c,base_raw),predictions[-1],atol=1e-7,rtol=1e-6)
    assert np.isfinite(predictions).all()
    write_json(directory/'complete.json',dict(epochs=epochs,parameters=sum(p.numel() for p in model.parameters()),
               seconds=time.perf_counter()-start,reload_verified=True,epoch0_base_match=spec['kind'] in ['warm','residual']))
    return model,directory


def execute(out,epochs=500,smoke=False):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out.mkdir(parents=True,exist_ok=True)
    source=Path(__file__);snapshot=out/'runner_snapshot.py'
    if snapshot.exists():assert file_hash(snapshot)==file_hash(source),'Runner changed: use a new output directory'
    else:snapshot.write_bytes(source.read_bytes())
    manifest=pd.read_csv(OLD/'reused_sensor_manifest.csv')
    if smoke:manifest=manifest[(manifest.fold=='fold_a')&(manifest.role=='inner')].head(1)
    protocol=dict(epochs=epochs,models=MODELS,alphas=ALPHAS,scope='current VB, no ODE/history',
                  device='CPU single thread deterministic',training_population='same labeled valid sensor records for ALL arms',
                  selection='inner validation only; fixed500 primary, inner epoch secondary for non-residual; residual alpha at fixed500',
                  hidden_match='CT64/32 2497 parameters; sensor50/25 2501',
                  residual='frozen newly trained CT64/32 + zero-output initialized residual; in-sample and nested OOF targets',
                  fallback='same original split/seed CT on invalid sensor records for all arms',
                  continuation='500 additional epochs with fresh optimizer for BOTH CT and sensor warm arms',
                  crossfit='leave one case out WITHIN current training split; refit c/t/VB scalers; never outer holdout',
                  save='all epoch predictions/metrics, epoch0 predictions, last checkpoint; no per-epoch checkpoints',
                  source_sha256=file_hash(source))
    write_json(out/'protocol.json',protocol)
    rows=[]
    for r in manifest.itertuples():
        c=context(r);models={};oof=None
        for name in MODELS:
            if name.endswith('_oof') and oof is None:
                oof,cf=crossfit(c,out,epochs)
                scope=out/'oof_targets'/r.fold/f'{r.role}_{int(r.holdout) if pd.notna(r.holdout) else "all"}_seed{int(r.seed)}'
                scope.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(scope/'targets.npz',case=c['table']['case'],run=c['table']['run'],prediction=oof)
                write_json(scope/'audit.json',cf)
            model,d=run_arm(c,name,out,epochs,models.get('ct_standard'),oof)
            models[name]=model
            rows.append(dict(name=name,fold=r.fold,role=r.role,holdout=None if pd.isna(r.holdout) else int(r.holdout),seed=int(r.seed),run_dir=str(d)))
            pd.DataFrame(rows).to_csv(out/'study_manifest.csv',index=False)
            write_json(out/'status.json',dict(status='running',completed=len(rows),planned=len(manifest)*len(MODELS),last=name))
            print(f'{len(rows)}/{len(manifest)*len(MODELS)} {r.fold} {r.role} {r.holdout} seed{r.seed} {name}',flush=True)
    write_json(out/'status.json',dict(status='training_completed',runs=len(rows),epochs=epochs))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    out=ROOT/'step23_smoke' if args.smoke else ROOT/'step23_controlled'
    try:execute(out,3 if args.smoke else 500,args.smoke)
    except Exception as e:write_json(out/'failure.json',dict(error=str(e)));raise


if __name__=='__main__':main()
