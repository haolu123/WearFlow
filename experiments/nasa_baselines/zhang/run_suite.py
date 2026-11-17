"""Frozen author-parameter NASA transfer: 24 runs, fixed last epoch 100."""
from wearflow_paths import ROOT as _WF_ROOT
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import Monitor,ZhangPhysics,paired_loss
from smoke import adjacent
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'formal'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):p.write_text(json.dumps(d,indent=2),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1)
    source={str(p):sha(p) for p in [ROOT/'model.py',ROOT/'smoke.py',ROOT/'run_suite.py',Path(str(_WF_ROOT / 'wear_monitoring_dynamics/models/zhang.py'))]}
    protocol=dict(method='Zhang_PINN_top12',epochs=100,alpha=1.,beta=.75,batch_size=16,optimizer='Adam',weight_decay=0.,monitor_lr=[.01,.003,.0001],physics_lr=.001,warmup=10,scheduler_after_epoch=True,source=source)
    frozen=OUT/'protocol.json'
    if frozen.exists():assert json.loads(frozen.read_text())==protocol,'Frozen source/protocol mismatch'
    else:save(frozen,protocol)
    jobs=[(fold,path.name,seed) for fold in ('fold_a','fold_b') for path in sorted((ROOT.parent/'data'/fold).iterdir()) if path.is_dir() for seed in (range(5) if path.name=='final' else [0])]
    finals=[];manifest=[];started=time.perf_counter()
    for fold,split,seed in jobs:
        data=ROOT.parent/'data'/fold/split
        identity=hashlib.sha256(json.dumps(dict(protocol=protocol,data=sha(data/'arrays.npz'),metadata=sha(data/'metadata.json'),seed=seed),sort_keys=True).encode()).hexdigest()
        out=OUT/f'{fold}_{split}_seed{seed}';out.mkdir(exist_ok=True)
        done=out/'summary.json'
        if done.exists():
            summary=json.loads(done.read_text());assert summary['identity']==identity
        else:
            torch.manual_seed(seed);np.random.seed(seed)
            a=dict(np.load(data/'arrays.npz'));meta=json.loads((data/'metadata.json').read_text())
            for label in ('train','validation','test'):assert not np.any(a[label+'_mask']&~a['sensor_valid']),'Frozen CT fallback not implemented: invalid labeled sensor'
            x=torch.tensor(a['X'],dtype=torch.float32);y=torch.tensor(a['y_z'],dtype=torch.float32).reshape(-1,1)
            pairs=adjacent(a);assert len(pairs)>0
            pd.DataFrame(pairs,columns=['left_index','right_index']).to_csv(out/'train_pairs.csv',index=False)
            monitor,physics=Monitor(),ZhangPhysics(17)
            optimizer=torch.optim.Adam([{'params':monitor.parameters(),'lr':.01},{'params':physics.parameters(),'lr':.001}])
            mean,std=meta['scalers']['wear'];hist=[];rows=[];tick=time.perf_counter()
            for epoch in range(1,101):
                monitor.train();physics.train();losses=np.zeros(4);lr=optimizer.param_groups[0]['lr']
                for batch in torch.randperm(len(pairs)).split(16):
                    left,right=pairs[batch.numpy()].T;optimizer.zero_grad()
                    loss,terms=paired_loss(monitor,physics,x[left],x[right],y[left],y[right])
                    assert torch.isfinite(loss);loss.backward()
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(monitor.parameters())+list(physics.parameters()))
                    optimizer.step();losses+=np.array([v.item() for v in (loss,*terms)])*len(batch)/len(pairs)
                monitor.eval()
                with torch.no_grad():pred=monitor(x).numpy().ravel()*std+mean
                record=dict(epoch=epoch,lr_monitor=lr,lr_physics=.001,**dict(zip(['loss','supervised','residual','direction'],losses)))
                for label in ('train','validation','test'):
                    idx=np.flatnonzero(a[label+'_mask'])
                    if not len(idx):continue
                    frame=pd.DataFrame(dict(case=a['case'][idx],ae=np.abs(pred[idx]-a['y_mm'][idx])))
                    record[label+'_mae_mm']=float(frame.ae.mean());record[label+'_case_macro_mae_mm']=float(frame.groupby('case').ae.mean().mean())
                    for i in idx:rows.append(dict(method='Zhang_PINN_top12',fold=fold,seed=seed,epoch=epoch,split=label,case=int(a['case'][i]),run=int(a['run'][i]),time=float(a['time'][i]),target_mm=float(a['y_mm'][i]),pred_mm=float(pred[i]),sensor_valid=True))
                hist.append(record);index=epoch-1
                optimizer.param_groups[0]['lr']=(.01+(.003-.01)*index/9 if index<10 else .0001+.5*(.003-.0001)*(1+math.cos(math.pi*(index-10)/90)))
            pd.DataFrame(hist).to_csv(out/'history.csv',index=False)
            pd.DataFrame(rows).to_csv(out/'predictions.csv.gz',index=False,compression='gzip')
            torch.save(dict(monitor=monitor.state_dict(),physics=physics.state_dict(),epoch=100,identity=identity),out/'last.pt')
            reload=Monitor();reload.load_state_dict(torch.load(out/'last.pt',weights_only=True)['monitor']);reload.eval()
            with torch.no_grad():torch.testing.assert_close(monitor(x),reload(x),rtol=0,atol=0)
            summary=dict(identity=identity,epochs=100,elapsed_seconds=time.perf_counter()-tick,reload_exact=True,pairs=len(pairs),source=source)
            save(done,summary)
        manifest.append(dict(fold=fold,split=split,seed=seed,identity=identity,epochs=100,run_dir=str(out),elapsed_seconds=summary['elapsed_seconds']))
        pd.DataFrame(manifest).to_csv(OUT/'study_manifest.csv',index=False)
        save(OUT/'status.json',dict(status='running',completed=len(manifest),total=24))
        if split=='final':
            frame=pd.read_csv(out/'predictions.csv.gz');finals.append(frame[(frame.epoch==100)&(frame.split=='test')])
    df=pd.concat(finals,ignore_index=True);df['ae']=(df.pred_mm-df.target_mm).abs()
    for seed,g in df.groupby('seed'):assert len(g)==138 and not g.duplicated(['case','run']).any()
    df.to_csv(OUT/'current_predictions.csv',index=False)
    cases=df.groupby(['method','fold','seed','case']).ae.mean().reset_index();cases.to_csv(OUT/'case_mae.csv',index=False)
    summary=dict(overall_case_macro=float(cases.ae.mean()),fold_case_macro=cases.groupby('fold').ae.mean().to_dict(),training_seconds=sum(x['elapsed_seconds'] for x in manifest),invocation_seconds=time.perf_counter()-started)
    save(OUT/'results.json',summary);save(OUT/'status.json',dict(status='complete',completed=24,total=24));print(json.dumps(summary))
if __name__=='__main__':main()
