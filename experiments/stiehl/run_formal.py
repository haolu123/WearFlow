"""Frozen Stiehl experiment. Atomic per-epoch checkpoint is the recovery authority."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from data_protocol import ROOT, records, pairs, transform
from models import WearModel, objective

OUT = ROOT / 'formal'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save_json(path, data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    os.replace(temp,path)

def save_predictions(path, data):
    temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**data)
    os.replace(temp,path)

def checkpoint(path,data):
    temp=path.with_suffix('.tmp')
    torch.save(data,temp)
    os.replace(temp,path)

def setup(split):
    frame,raw,_=records()
    scaler=json.loads((ROOT/'splits'/f'{split}.json').read_text())
    x,valid=transform(frame,raw,scaler)
    assert valid.all(), 'Invalid sensors require audited fallback; do not silently impute.'
    roles=np.full(len(frame),'test',dtype='U10')
    roles[frame.tool.isin(scaler['train_tools'])]='train'
    if scaler['validation_tool'] is not None:
        roles[frame.tool==scaler['validation_tool']]='validation'
    pp=pairs(frame)
    ti=np.flatnonzero(roles=='train')
    train_frame=frame.iloc[ti].reset_index(drop=True)
    tp=pairs(train_frame)
    y=(frame.VB_mm.to_numpy()-scaler['wear_mean'])/scaler['wear_std']
    return {'frame':frame,'scaler':scaler,'x':torch.tensor(x),'y':torch.tensor(y,dtype=torch.float32),
            'roles':roles,'pairs':pp,'train_index':ti,'train_tools':torch.tensor(train_frame.tool.to_numpy()),
            'train_pairs':torch.tensor(tp[['origin','target']].to_numpy()),
            'train_dt':torch.tensor(tp.actual_seconds.to_numpy()/1000,dtype=torch.float32)}

def evaluate(model,d,mode):
    pp=d['pairs'];i=pp.origin.to_numpy();j=pp.target.to_numpy()
    scale=d['scaler']['wear_std'];mean=d['scaler']['wear_mean']
    with torch.no_grad():
        z=model(d['x']); current=z.numpy()*scale+mean
        if mode in ['local','integral']:
            dt=torch.tensor(pp.actual_seconds.to_numpy()/1000,dtype=torch.float32)
            future=model.propagate(z[i],dt).numpy()*scale+mean
            oracle=model.propagate(d['y'][i],dt).numpy()*scale+mean
        elif mode=='time':
            future=current[j];oracle=np.full(len(pp),np.nan,dtype=np.float32)
        else:
            future=current[i];oracle=np.full(len(pp),np.nan,dtype=np.float32)
    result={'current_mm':current,'future_mm':future,'oracle_mm':oracle}
    if mode=='current':
        slope=np.zeros(len(current),dtype=np.float32)
        for _,g in d['frame'].groupby('tool'):
            ix=g.index.to_numpy()
            slope[ix[1:]]=np.diff(current[ix])/np.diff(g.cumulated_tool_contact_time.to_numpy())
        result['linear_mm']=current[i]+slope[i]*pp.actual_seconds.to_numpy()
    assert np.isfinite(current).all() and np.isfinite(future).all()
    if mode in ['local','integral']:assert np.isfinite(oracle).all()
    return result

def metrics(pred,d):
    rows=[];frame=d['frame'];pp=d['pairs'];truth=frame.VB_mm.to_numpy()
    for role in np.unique(d['roles']):
        for tool in frame.loc[d['roles']==role,'tool'].unique():
            mask=frame.tool.to_numpy()==tool
            error=pred['current_mm'][mask]-truth[mask]
            rows.append({'role':role,'tool':int(tool),'task':'current','horizon':0,'n':int(mask.sum()),'mae':float(abs(error).mean()),'rmse':float(np.sqrt(np.mean(error**2)))})
            for h in [240,480,720]:
                m=(pp.tool.to_numpy()==tool)&(pp.requested_seconds.to_numpy()==h)
                for key in ['future_mm','oracle_mm','linear_mm']:
                    if key not in pred or not np.isfinite(pred[key][m]).all():continue
                    error=pred[key][m]-truth[pp.target.to_numpy()[m]]
                    rows.append({'role':role,'tool':int(tool),'task':key,'horizon':h,'n':int(m.sum()),'mae':float(abs(error).mean()),'rmse':float(np.sqrt(np.mean(error**2)))})
    return rows

def identity(split,method,seed):
    files=[Path(__file__),ROOT/'models.py',ROOT/'data_protocol.py',ROOT/'protocol.json',ROOT/'features.npz',ROOT/'splits'/f'{split}.json']
    obj={'split':split,'method':method,'seed':int(seed),'epochs':100,'sources':{p.name:sha(p) for p in files}}
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def train(split,method,seed,limit=100):
    d=setup(split);folder=OUT/split/f'{method}_seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    token=identity(split,method,seed)
    torch.manual_seed(seed);model=WearModel(method=='time')
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    initial={k:v.clone() for k,v in model.state_dict().items()}
    last=folder/'last.pt';start_epoch=0;seconds=0.
    if last.exists():
        state=torch.load(last,weights_only=False)
        assert state['identity']==token,'Identity changed: refusing to reuse or overwrite training.'
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);torch.set_rng_state(state['rng'])
        start_epoch=state['epoch']+1;seconds=state['elapsed_seconds']
        initial=state['initial']
    save_json(folder/'identity.json',{'identity':token,'split':split,'method':method,'seed':seed,'scaler':d['scaler'],'effective_parameters':sum(p.numel() for p in model.monitor.parameters())+(sum(p.numel() for p in model.rate.parameters()) if method in ['local','integral'] else 0)})
    ti=d['train_index']
    for epoch in range(start_epoch,limit+1):
        clock=time.perf_counter();loss_value=None
        if epoch>0:
            model.train();opt.zero_grad()
            loss,_,_=objective(model,d['x'][ti],d['y'][ti],d['train_tools'],d['train_pairs'],d['train_dt'],method)
            assert torch.isfinite(loss)
            loss.backward();opt.step();loss_value=float(loss.detach())
        model.eval();pred=evaluate(model,d,method)
        save_predictions(folder/f'epoch_{epoch:03}.npz',pred)
        save_json(folder/f'epoch_{epoch:03}.json',{'epoch':epoch,'loss':loss_value,'metrics':metrics(pred,d)})
        seconds+=time.perf_counter()-clock
        checkpoint(last,{'identity':token,'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'initial':initial,'epoch':epoch,'elapsed_seconds':seconds})
        save_json(OUT/'active_run.json',{'split':split,'method':method,'seed':seed,'epoch':epoch,'elapsed_seconds':seconds})
    if limit==100:
        reloaded=WearModel(method=='time');state=torch.load(last,weights_only=False);reloaded.load_state_dict(state['model']);reloaded.eval()
        original=np.load(folder/'epoch_100.npz');again=evaluate(reloaded,d,method)
        for key in again:assert np.array_equal(original[key],again[key],equal_nan=True)
        rate_changed=any(not torch.equal(initial[k],model.state_dict()[k]) for k in initial if k.startswith('rate.'))
        assert rate_changed==(method in ['local','integral'])
        assert len(list(folder.glob('epoch_*.npz')))==101
        save_json(folder/'complete.json',{'identity':token,'epochs':100,'seconds':seconds,'reload_exact':True,'rate_changed':rate_changed})
    return folder

def gpr(split):
    folder=OUT/split/'gpr';folder.mkdir(parents=True,exist_ok=True)
    token=identity(split,'gpr',0)
    if (folder/'complete.json').exists():
        assert json.loads((folder/'complete.json').read_text())['identity']==token
        return
    import joblib
    d=setup(split);frame=d['frame'];selected=[]
    for tool in d['scaler']['train_tools']:
        g=frame[frame.tool==tool];t=g.cumulated_tool_contact_time.to_numpy()
        selected.extend(g.index.to_numpy()[np.unique(np.searchsorted(t,np.linspace(t.min(),t.max(),128)))])
    gp=GaussianProcessRegressor(kernel=ConstantKernel(1)*RBF(1)+WhiteKernel(.01),alpha=1e-6,normalize_y=False,n_restarts_optimizer=0,random_state=0)
    clock=time.perf_counter();gp.fit(d['x'].numpy()[selected],d['y'].numpy()[selected])
    pred=gp.predict(d['x'].numpy())*d['scaler']['wear_std']+d['scaler']['wear_mean']
    joblib.dump(gp,folder/'model.joblib')
    np.testing.assert_array_equal(gp.predict(d['x'].numpy()),joblib.load(folder/'model.joblib').predict(d['x'].numpy()))
    save_predictions(folder/'predictions.npz',{'current_mm':pred})
    save_json(folder/'complete.json',{'identity':token,'seconds':time.perf_counter()-clock,'train_indices':list(map(int,selected)),'kernel':str(gp.kernel_),'reload_exact':True})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke-resume',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if args.smoke_resume:
        # The same run continues exactly: smoke does not change identity or data.
        train('test9_inner','integral',0,limit=1)
        train('test9_inner','integral',0,limit=2)
        print('Atomic epoch resume passed; epochs 0..2 retained for formal continuation.',flush=True)
        return
    (OUT/'process_id.txt').write_text(str(os.getpid()))
    started=time.time();manifest=pd.read_csv(ROOT/'study_manifest.csv')
    # Inner diagnostics first; the fixed test protocol is not changed thereafter.
    manifest['order']=~manifest.split.str.endswith('inner')
    manifest=manifest.sort_values(['order','split','seed','method'])
    complete=0
    for row in manifest.itertuples():
        train(row.split,row.method,int(row.seed));complete+=1
        save_json(OUT/'status.json',{'stage':'neural_training','completed':complete,'total':len(manifest),'session_seconds':time.time()-started})
        print(f'Completed {complete}/{len(manifest)} {row.split} {row.method} seed{row.seed}',flush=True)
    for split in manifest.split.unique():gpr(split)
    save_json(OUT/'status.json',{'stage':'training_complete_report_pending','completed':complete,'total':len(manifest),'gpr_completed':18,'session_seconds':time.time()-started})
    print('All formal models completed. Run report_formal.py for final audit and aggregation.',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True)
        (OUT/'execution_error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
