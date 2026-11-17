"""Warm-start current-monitor and physical-state ODE single-shot adaptation."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from ..config import project_path,write_json,file_hash,digest
from ..models.registry import build_model,build_dynamics
from ..data.records import load_table
from ..data.feature_archive import attach_features
from ..data.sensor_quality import apply_quality
from ..data.monitor_batch import make_monitor_batch
from ..data.datasets import observations,pair_tensors
from ..evaluation.epoch_logging import EpochRecorder
from ..training.reproducibility import setup,synchronize

class SingleShot(nn.Module):
    def __init__(self,monitor,dynamics):
        super().__init__();self.monitor=monitor;self.dynamics=dynamics
    def estimate(self,batch,indices,time=None):
        return self.monitor.estimate(batch,indices,time)
    def forecast(self,batch,pairs):
        a,b=pairs[:,0],pairs[:,1]
        return self.dynamics.propagate(self.estimate(batch,a),batch['condition'][a],batch['time'][b]-batch['time'][a])


def load_parent(sensor_dir,ode_dir,device):
    s=torch.load(Path(sensor_dir)/'last.pt',map_location=device,weights_only=True);o=torch.load(Path(ode_dir)/'last.pt',map_location=device,weights_only=True)
    assert s['split']==o['split'] and s['data_signature']==o['data_signature']
    for key in ['wear','condition','time_scale']:assert s['scalers'][key]==o['scalers'][key],key
    cfg=s['config'];table,signature=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']));assert signature==s['data_signature']
    table=apply_quality(table,cfg['data'].get('sensor_quality'));table=attach_features(table,cfg['data']['feature_archive'])
    batch=make_monitor_batch(cfg,table,s['scalers'],device)
    monitor=build_model(cfg).to(device);monitor.load_state_dict(s['model']);ode=build_dynamics(o['config']['dynamics']).to(device);ode.load_state_dict(o['model'])
    return SingleShot(monitor,ode).to(device),batch,table,s


def execute(task,directory,epochs=500):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    identity=dict(task=task,epochs=epochs,learning_rate=1e-4,weight_decay=1e-4,sensor_hash=file_hash(Path(task['sensor_dir'])/'last.pt'),ode_hash=file_hash(Path(task['ode_dir'])/'last.pt'),execution_hash=file_hash(__file__))
    if (directory/'status.json').exists() and json.loads((directory/'status.json').read_text())['status']=='completed':
        assert json.loads((directory/'identity.json').read_text())==identity,'Completed run identity changed'
        return
    write_json(directory/'identity.json',identity);write_json(directory/'status.json',dict(status='training'))
    start=time.perf_counter();device=setup(task['seed'],'cuda',True);torch.set_num_threads(1)
    model,batch,table,parent=load_parent(task['sensor_dir'],task['ode_dir'],device);split=parent['split'];sc=parent['scalers']
    write_json(directory/'split.json',split);write_json(directory/'scalers.json',sc)
    model.dynamics.requires_grad_(task['mode']=='joint')
    obs=observations(table,split['train']);obs=obs[table['sensor_valid'][obs]];pairs=pair_tensors(table,split['train'],[1,2,3],device)
    initial_ode={k:v.detach().clone() for k,v in model.dynamics.state_dict().items()};initial_monitor={k:v.detach().clone() for k,v in model.monitor.state_dict().items()}
    # Audit future gradients through the frozen ODE as well as joint dynamics.
    model.zero_grad(set_to_none=True);future_loss=(model.forecast(batch,pairs)-batch['target'][pairs[:,1]]).square().mean();future_loss.backward()
    mg=sum(float(p.grad.square().sum()) for p in model.monitor.parameters() if p.grad is not None)**.5
    dg=sum(float(p.grad.square().sum()) for p in model.dynamics.parameters() if p.grad is not None)**.5
    assert mg>0 and (dg>0 if task['mode']=='joint' else dg==0)
    # Changing future-only sensor must not change this pair's forecast.
    pair=pairs[:1];target=int(pair[0,1]);origin=int(pair[0,0]);assert target!=origin
    with torch.no_grad():
        before=model.forecast(batch,pair).clone();saved=batch['sensor'][target].clone();batch['sensor'][target]+=10
        after=model.forecast(batch,pair);batch['sensor'][target]=saved;torch.testing.assert_close(before,after,rtol=0,atol=0)
    write_json(directory/'gradient_audit.json',dict(future_to_monitor_norm=mg,future_to_ode_norm=dg,no_future_sensor=True,scaler_units_equal=True))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    with EpochRecorder(directory,table,batch,sc,split,[1,2,3],False) as recorder:
        initial=recorder.record(model,0);write_json(directory/'epoch0_metrics.json',initial)
        for epoch in range(1,epochs+1):
            synchronize(device);ts=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
            current=(model.estimate(batch,obs)-batch['target'][obs]).square().mean()
            if task['lambda_future']:
                future=(model.forecast(batch,pairs)-batch['target'][pairs[:,1]]).square().mean();loss=current+task['lambda_future']*future
            else:future=torch.zeros((),device=device);loss=current
            assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);optimizer.step()
            row=dict(epoch=epoch,loss=float(loss.detach()),loss_current=float(current.detach()),loss_future=float(future.detach()));row.update(recorder.record(model,epoch));synchronize(device);row['seconds']=time.perf_counter()-ts
            pd.DataFrame([row]).to_csv(directory/'history.csv',index=False,mode='w' if epoch==1 else 'a',header=epoch==1)
            if task['role']=='final':
                ep=directory/'epochs';ep.mkdir(exist_ok=True);torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},ep/f'epoch_{epoch:04d}.pt')
    state={k:v.detach().cpu() for k,v in model.state_dict().items()};torch.save(dict(model=state,epoch=epochs,task=task,scalers=sc,split=split),directory/'last.pt')
    monitor_change=max(float((v-initial_monitor[k]).abs().max()) for k,v in model.monitor.state_dict().items() if v.is_floating_point())
    ode_change=max(float((v-initial_ode[k]).abs().max()) for k,v in model.dynamics.state_dict().items())
    assert monitor_change>0 and (ode_change>0 if task['mode']=='joint' else ode_change==0)
    model.eval()
    with torch.no_grad():before=model.forecast(batch,pairs)
    restored,_,_,_=load_parent(task['sensor_dir'],task['ode_dir'],device);restored.load_state_dict(torch.load(directory/'last.pt',map_location=device,weights_only=True)['model']);restored.eval()
    with torch.no_grad():after=restored.forecast(batch,pairs)
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    write_json(directory/'metrics.json',dict(epochs=epochs,seconds=time.perf_counter()-start,monitor_max_change=monitor_change,ode_max_change=ode_change,reload_verified=True,epoch0_recorded=True))
    write_json(directory/'status.json',dict(status='completed'));print('COMPLETE '+task['id'],flush=True)

