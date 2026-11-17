"""Resumable fixed-protocol PIDGGCN NASA causal feature-graph experiment."""
from wearflow_paths import ROOT as _WF_ROOT
import argparse,copy,hashlib,json,os,sys,time,traceback
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from model import PIDGGCN,fit_condition_priors,physics_loss,own_scalar_time_derivative,HalfDecay
PROJECT=Path(str(_WF_ROOT / ''))
sys.path.insert(0,str(PROJECT))
from wear_monitoring_dynamics.models.registry import build_model
ROOT=Path(__file__).resolve().parent;SUITE=ROOT.parent
VARIANTS=('paper_printed','calculus_corrected')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def savejson(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2),encoding='utf8');tmp.replace(path)
def savecheckpoint(path,obj):
    tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
def state_hash(model):return hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.state_dict().values())).hexdigest()


def fallback(a,meta,seed):
    table=pd.read_csv(PROJECT/'wear_monitoring_dynamics/outputs/current12_fusion_20261013/reused_ct_manifest.csv')
    mask=(table.fold==meta['fold'])&(table.role==meta['role'])&(table.seed==seed)
    if meta['role']=='inner':mask &= table.holdout==meta['split']['validation'][0]
    rows=table[mask];assert len(rows)==1
    path=Path(rows.iloc[0].run_dir)/'last.pt';ck=torch.load(path,map_location='cpu',weights_only=True)
    for key in ('wear','condition','time_scale'):np.testing.assert_array_equal(ck['scalers'][key],meta['scalers'][key])
    assert ck['scalers']['train_cases']==sorted(meta['split']['train'])
    batch={'sensor':torch.tensor(a['sensor_z'],dtype=torch.float32),'condition':torch.tensor(a['X_ct'][:,:4],dtype=torch.float32),'time':torch.tensor(a['X_ct'][:,4:5],dtype=torch.float32)}
    model=build_model(ck['config']);model.load_state_dict(ck['model']);model.eval()
    with torch.no_grad():pred=model.estimate(batch,torch.arange(len(a['case']))).flatten().numpy().astype(np.float64)
    return pred,dict(path=str(path),sha256=sha(path),epoch=ck.get('epoch'))


def prepare(path,seed):
    a=dict(np.load(path/'arrays.npz'));meta=json.loads((path/'metadata.json').read_text());mask=a['train_mask']
    pool,priors=fit_condition_priors(a['time'][mask],a['y_mm'][mask],a['case'][mask],a['condition_raw'][mask],meta['scalers']['time_scale'])
    fb,fbmeta=fallback(a,meta,seed)
    cases=sorted(set(meta['split']['train']+meta['split']['validation']+meta['split']['test']))
    sequences={}
    for case in cases:
        ids=np.flatnonzero((a['case']==case)&a['sensor_valid'])
        ids=ids[np.argsort(a['time'][ids],kind='stable')]
        assert len(ids) and np.all(np.diff(a['time'][ids])>0)
        sequences[case]=ids
    return a,meta,pool,priors,fb,fbmeta,sequences


def evaluate(model,a,meta,sequences,fb):
    model.eval();pred=np.full(len(a['case']),np.nan)
    with torch.no_grad():
        for case,ids in sequences.items():
            pred[ids]=model(torch.tensor(a['X'][ids],dtype=torch.float64)).numpy()
            invalid=(a['case']==case)&~a['sensor_valid'];pred[invalid]=fb[invalid]
    metrics={}
    for split in ('train','validation','test'):
        mask=a[split+'_mask'];assert np.isfinite(pred[mask]).all()
        errors=np.abs(pred[mask]-a['y_z'][mask])*meta['scalers']['wear'][1]
        metrics[split+'_mae_mm']=float(errors.mean()) if len(errors) else None
        metrics[split+'_case_macro_mae_mm']=float(np.mean([errors[a['case'][mask]==c].mean() for c in np.unique(a['case'][mask])])) if len(errors) else None
    return pred,metrics


def train_epoch(model,optimizer,a,meta,sequences,priors,variant,weights):
    model.train();optimizer.zero_grad(set_to_none=True);parts=[]
    cases=meta['split']['train']
    for case in cases:
        ids=sequences[case];label=np.flatnonzero(a['train_fit_mask'][ids]);assert len(label)
        # Scalar leaves isolate exact own-time partials without severing
        # recurrent hidden or prior feature paths from parameter gradients.
        leaves=[torch.tensor(v,dtype=torch.float64,requires_grad=True) for v in a['X'][ids,-1]]
        u=torch.stack(leaves);x=torch.cat([torch.tensor(a['X'][ids,:-1],dtype=torch.float64),u[:,None]],1)
        pred=model(x);derivative=own_scalar_time_derivative(pred,leaves,label)
        target=torch.tensor(a['y_z'][ids[label]],dtype=torch.float64)
        prior=priors[tuple(a['condition_raw'][ids[0]])]
        assert prior['time_scale']==meta['scalers']['time_scale']
        losses=physics_loss(pred[label],target,derivative,u[label],prior['abc'],meta['scalers']['wear'][1],variant)
        total=weights[0]*losses[0]+weights[1]*(losses[1]+losses[2])
        assert torch.isfinite(total),('nonfinite loss',case)
        (total/len(cases)).backward();parts.append([float(p.detach()) for p in losses])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'nonfinite parameter gradient'
    norm=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
    optimizer.step()
    return np.mean(parts,axis=0),norm


def execute_run(path,seed,variant,out,epochs):
    folder=out/path.parent.name/path.name/f'seed_{seed}'/variant;folder.mkdir(parents=True,exist_ok=True)
    a,meta,pool,priors,fb,fbmeta,sequences=prepare(path,seed)
    identity=dict(data_hash=sha(path/'arrays.npz'),metadata_hash=sha(path/'metadata.json'),model_hash=sha(ROOT/'model.py'),runner_hash=sha(Path(__file__)),seed=seed,variant=variant,epochs=epochs,
      dtype='float64',optimizer='RMSprop',lr=.001,alpha=.99,eps=1e-8,momentum=0.,weight_decay=0.,case_equal=True,dropout=.5,
      schedule=dict(warmup=50,check_interval=2),fallback=fbmeta)
    if (folder/'identity.json').exists():assert json.loads((folder/'identity.json').read_text())==identity,'run identity changed'
    else:savejson(folder/'identity.json',identity)
    if (folder/'completed.json').exists():return json.loads((folder/'completed.json').read_text())
    savejson(folder/'physical_priors.json',dict(pool=pool,groups=list(priors.values())))
    torch.manual_seed(seed);model=PIDGGCN().double();initial=copy.deepcopy(model.state_dict());initialhash=state_hash(model)
    optimizer=torch.optim.RMSprop(model.parameters(),lr=.001);schedule=HalfDecay(100,True)
    history=[];predictions=[];last=-1;elapsed_previous=0.
    checkpoint=folder/'checkpoint.pt'
    if checkpoint.exists():
        ck=torch.load(checkpoint,weights_only=False,map_location='cpu');assert ck['identity']==identity
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng'])
        schedule.data,schedule.physics=ck['weights'];history=ck['history'];predictions=ck['predictions'];last=ck['epoch'];elapsed_previous=ck['elapsed_seconds']
        assert ck['initial_hash']==initialhash
    started=time.perf_counter();elapsed=elapsed_previous
    for epoch in range(last+1,epochs+1):
        row=dict(epoch=epoch,lambda_data=schedule.data,lambda_physics=schedule.physics)
        if epoch:
            loss,norm=train_epoch(model,optimizer,a,meta,sequences,priors,variant,(schedule.data,schedule.physics))
            row.update(loss_data=float(loss[0]),loss_physics=float(loss[1]),loss_monotonic=float(loss[2]),gradient_norm=norm)
            schedule.step(epoch,loss[0],loss[1]+loss[2])
        pred,metrics=evaluate(model,a,meta,sequences,fb);row.update(metrics);history.append(row);predictions.append(pred)
        elapsed=elapsed_previous+time.perf_counter()-started
        savecheckpoint(checkpoint,dict(identity=identity,initial=initial,initial_hash=initialhash,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),weights=(schedule.data,schedule.physics),history=history,predictions=predictions,epoch=epoch,elapsed_seconds=elapsed))
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
        savejson(out/'active_run.json',dict(run=str(folder.relative_to(out)),epoch=epoch,total_epochs=epochs,elapsed_seconds=elapsed))
    restored=PIDGGCN().double();restored.load_state_dict(torch.load(checkpoint,weights_only=False)['model'])
    re,_=evaluate(restored,a,meta,sequences,fb);np.testing.assert_array_equal(re,predictions[-1])
    np.savez_compressed(folder/'epoch_predictions.npz',pred_z=np.array(predictions),case=a['case'],run=a['run'],time=a['time'],target_mm=a['y_mm'],sensor_valid=a['sensor_valid'],train_mask=a['train_mask'],validation_mask=a['validation_mask'],test_mask=a['test_mask'])
    done=dict(status='complete',epochs=epochs,seconds=elapsed,initial_hash=initialhash,reload_exact=True,parameters=sum(p.numel() for p in model.parameters()),fallback=fbmeta)
    savejson(folder/'completed.json',done);return done


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--execute',action='store_true');args=parser.parse_args();assert args.smoke or args.execute
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=ROOT/('runner_smoke' if args.smoke else 'formal');out.mkdir(exist_ok=True)
    tasks=[]
    for path in sorted((SUITE/'data').glob('fold_*/*')):
        meta=json.loads((path/'metadata.json').read_text())
        for seed in ([0] if meta['role']=='inner' else range(5)):
            for variant in VARIANTS:tasks.append((path,seed,variant))
    if args.smoke:tasks=[(SUITE/'data/fold_a/inner_1',0,v) for v in VARIANTS]
    pd.DataFrame([dict(split=str(p.relative_to(SUITE/'data')),seed=s,variant=v) for p,s,v in tasks]).to_csv(out/'study_manifest.csv',index=False)
    (out/'process_id.txt').write_text(str(os.getpid()));started=time.perf_counter();completed=0
    savejson(out/'status.json',dict(status='running',completed=0,total=len(tasks)))
    try:
        for path,seed,variant in tasks:
            done=execute_run(path,seed,variant,out,3 if args.smoke else 100);completed+=1
            savejson(out/'status.json',dict(status='running',completed=completed,total=len(tasks),elapsed_seconds=time.perf_counter()-started))
            print(completed,len(tasks),path.parent.name,path.name,seed,variant,done['seconds'],flush=True)
        savejson(out/'status.json',dict(status='complete',completed=completed,total=len(tasks),elapsed_seconds=time.perf_counter()-started))
    except Exception:
        (out/'execution_error.log').write_text(traceback.format_exc(),encoding='utf8')
        savejson(out/'status.json',dict(status='failed',completed=completed,total=len(tasks)));raise


if __name__=='__main__':main()
