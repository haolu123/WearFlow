"""Resumable random-initialization mechanism experiment, fixed 100 epochs."""
from wearflow_paths import ROOT as _WF_ROOT
import copy,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from mechanism import *
from wear_monitoring_dynamics.models.registry import build_model
OUT=Path(__file__).parent/'results'
PROJECT=Path(str(_WF_ROOT / ''))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def savejson(path,obj):path.write_text(json.dumps(obj,indent=2),encoding='utf8')
def fallback(batch,meta,seed):
    table=pd.read_csv(PROJECT/'wear_monitoring_dynamics/outputs/current12_fusion_20261013/reused_ct_manifest.csv')
    select=(table.fold==meta['fold'])&(table.role==meta['role'])&(table.seed==seed)
    if meta['role']=='inner':select &= table.holdout==meta['split']['validation'][0]
    rows=table[select];assert len(rows)==1
    path=Path(rows.iloc[0].run_dir)/'last.pt'; state=torch.load(path,map_location='cpu',weights_only=True)
    for key in ('wear','condition','time_scale'):assert np.array_equal(state['scalers'][key],meta['scalers'][key])
    assert state['scalers']['train_cases']==sorted(meta['split']['train'])
    model=build_model(state['config']);model.load_state_dict(state['model']);model.eval()
    with torch.no_grad():batch['fallback_vb_z']=model.estimate(batch,torch.arange(len(batch['time']))).detach()
    batch['sensor'][~batch['sensor_valid']]=0
    return {'path':str(path),'sha256':sha(path),'epoch':state.get('epoch')}
def eval_epoch(model,batch,a,pairs):
    model.eval()
    with torch.no_grad():
        raw=model.estimate(batch,torch.arange(len(a['case']))).flatten().numpy()
        f=model.forecast(batch,pairs).flatten().numpy()
        origin,target=pairs[:,0],pairs[:,1]
        oracle=model.dynamics.propagate(batch['target'][origin],batch['condition'][origin],batch['time'][target]-batch['time'][origin]).flatten().numpy()
    return raw,f,oracle

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    tasks=[]
    for path in sorted((SUITE/'data').glob('*/*')):
        if not (path/'arrays.npz').exists():continue
        meta=json.loads((path/'metadata.json').read_text())
        for seed in ([0] if meta['role']=='inner' else range(5)):
            for arm in ARMS:tasks.append((path,seed,arm))
    pd.DataFrame([{'split':str(p.relative_to(SUITE/'data')),'seed':s,'arm':a} for p,s,a in tasks]).to_csv(OUT/'study_manifest.csv',index=False)
    started=time.perf_counter();complete=0
    sources={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('mechanism.py'),PROJECT/'wear_monitoring_dynamics/losses/derivative.py',PROJECT/'wear_monitoring_dynamics/models/feature_mlp.py',PROJECT/'wear_monitoring_dynamics/models/dynamics.py',SUITE/'internal_protocol.md',PROJECT/'release/CondWear-ODE/models/vb_sensor_condition_ode.py']}
    for path,seed,arm in tasks:
        folder=OUT/path.parent.name/path.name/f'seed_{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
        a,meta,batch,obs,trainpairs=load_split(path);fb=fallback(batch,meta,seed)
        identity={'fallback':fb,'sources':sources,'data_hash':sha(path/'arrays.npz'),'metadata_hash':sha(path/'metadata.json'),'seed':seed,'arm':arm,'epochs':100,'lr':.001,'wd':.0001}
        if (folder/'completed.json').exists():
            assert json.loads((folder/'identity.json').read_text())==identity
            complete+=1;continue
        savejson(folder/'identity.json',identity)
        allpairs=torch.tensor(np.concatenate([a[k+'_pairs'] for k in ('train','validation','test')]),dtype=torch.long)
        assert torch.all(torch.tensor(a['case'])[allpairs[:,0]]==torch.tensor(a['case'])[allpairs[:,1]])
        assert torch.equal(batch['condition'][allpairs[:,0]],batch['condition'][allpairs[:,1]])
        torch.manual_seed(seed);model=Mechanism();initial=copy.deepcopy(model.state_dict());initialhash=state_hash(model)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        current=[];future=[];oracle=[];history=[];diagnostics=[];start=time.perf_counter()
        for epoch in range(101):
            if epoch:
                model.train()
                optimizer.zero_grad(set_to_none=True); losses=components(model,batch,obs,trainpairs,arm)
                if epoch in (1,100):
                    norms={};vectors={}
                    for name,loss in losses.items():vectors[name],norms[name]=gradients(loss,model)
                    cos={f'{x}:{y}':float(torch.nn.functional.cosine_similarity(vectors[x],vectors[y],dim=0)) for x in vectors for y in vectors if x<y}
                    diagnostics.append({'epoch':epoch,'norms':norms,'cosines':cos})
                total=sum(losses.values());assert torch.isfinite(total);total.backward()
                assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                optimizer.step();history.append({'epoch':epoch,**{k:float(v.detach()) for k,v in losses.items()}})
            raw,f,o=eval_epoch(model,batch,a,allpairs);current.append(raw);future.append(f);oracle.append(o)
            if epoch==0:history.append({'epoch':0})
            for split_name in ('train','validation','test'):
                mask=a[split_name+'_mask']; pm=mask[allpairs[:,0].numpy()]
                history[-1][split_name+'_current_mae_mm']=float(np.abs(raw[mask]-a['y_z'][mask]).mean()*meta['scalers']['wear'][1]) if mask.any() else None
                history[-1][split_name+'_future_mae_mm']=float(np.abs(f[pm]-a['y_z'][allpairs[:,1].numpy()[pm]]).mean()*meta['scalers']['wear'][1]) if pm.any() else None
        elapsed=time.perf_counter()-start
        rate_change=max(float((model.state_dict()[k]-v).abs().max()) for k,v in initial.items() if k.startswith('dynamics'))
        assert (rate_change==0)==(arm=='current_only')
        torch.save({'initial':initial,'final':model.state_dict(),'scalers':meta['scalers'],'epoch':100,'identity':identity},folder/'model.pt')
        reloaded=Mechanism();reloaded.load_state_dict(torch.load(folder/'model.pt',weights_only=True)['final'])
        restored=eval_epoch(reloaded,batch,a,allpairs)
        assert all(np.array_equal(x,y) for x,y in zip(restored,(current[-1],future[-1],oracle[-1])))
        np.savez_compressed(folder/'epoch_predictions.npz',train_observations=obs.numpy(),train_pairs_used=trainpairs.numpy(),current_z=np.array(current),future_z=np.array(future),oracle_z=np.array(oracle),pairs=allpairs.numpy(),case=a['case'],run=a['run'],time=a['time'],target_mm=a['y_mm'],train_mask=a['train_mask'],validation_mask=a['validation_mask'],test_mask=a['test_mask'],sensor_valid=a['sensor_valid'])
        pd.DataFrame(history).to_csv(folder/'history.csv',index=False);savejson(folder/'gradients.json',diagnostics)
        savejson(folder/'completed.json',{'seconds':elapsed,'initial_hash':initialhash,'fallback':fb,'rate_change':rate_change,'epochs':100,'reload_exact':True})
        complete+=1;savejson(OUT/'status.json',{'status':'running','completed':complete,'total':len(tasks),'elapsed_seconds':time.perf_counter()-started})
        print(complete,len(tasks),str(folder.relative_to(OUT)),round(elapsed,2),flush=True)
    savejson(OUT/'status.json',{'status':'complete','completed':complete,'total':len(tasks),'elapsed_seconds':time.perf_counter()-started})
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);(OUT/'execution_error.log').write_text(traceback.format_exc());raise







