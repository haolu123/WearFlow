"""100-epoch adaptations from monitor epoch100 and independent ODE epoch500.
Training is single-shot; fixed sensor fusion is an evaluation-only deployment rule.
"""
import csv,gzip,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..config import file_hash,write_json
from .reproducibility import setup
from .single_shot_adaptation import load_parent
from ..data.datasets import observations,pair_tensors
from ..evaluation.epoch_logging import EpochRecorder
from .recursive_adaptation import layout,record,rollout


def prepare(task):
    model,b,t,parent=load_parent(task['sensor_dir'],task['ode_dir'],'cpu')
    ck=torch.load(task['monitor100'],map_location='cpu',weights_only=True)
    assert ck['epoch']==100
    model.monitor.load_state_dict(ck['model'])
    return model,b,t,parent


def execute(task,directory,epochs=100):
    d=Path(directory);d.mkdir(parents=True,exist_ok=True)
    identity=dict(task=task,epochs=epochs,learning_rate=1e-4,weight_decay=1e-4,
                  monitor100_hash=file_hash(task['monitor100']),ode_hash=file_hash(Path(task['ode_dir'])/'last.pt'),execution_hash=file_hash(__file__))
    if (d/'status.json').exists() and json.loads((d/'status.json').read_text())['status']=='completed':
        assert json.loads((d/'identity.json').read_text())==identity
        return
    write_json(d/'identity.json',identity);write_json(d/'status.json',dict(status='training'))
    started=time.perf_counter();setup(task['seed'],'cpu',True);torch.set_num_threads(1)
    model,b,t,parent=prepare(task);sp=parent['split'];sc=parent['scalers']
    write_json(d/'split.json',sp);write_json(d/'scalers.json',sc)
    model.dynamics.requires_grad_(task['group']=='D')
    initial={k:v.clone() for k,v in model.state_dict().items()}
    obs=observations(t,sp['train']);obs=obs[t['sensor_valid'][obs]]
    pairs=pair_tensors(t,sp['train'],[1,2,3],'cpu')
    model.zero_grad(set_to_none=True)
    (model.forecast(b,pairs)-b['target'][pairs[:,1]]).square().mean().backward()
    norms={name:sum(float(p.grad.square().sum()) for p in branch.parameters() if p.grad is not None)**.5 for name,branch in [('monitor',model.monitor),('ode',model.dynamics)]}
    assert norms['monitor']>0 and (norms['ode']>0 if task['group']=='D' else norms['ode']==0)
    pair=pairs[:1];j=int(pair[0,1]);saved=b['sensor'][j].clone()
    with torch.no_grad():
        y=model.forecast(b,pair);b['sensor'][j]+=10;torch.testing.assert_close(y,model.forecast(b,pair),rtol=0,atol=0);b['sensor'][j]=saved
    write_json(d/'gradient_audit.json',dict(future_to_monitor=norms['monitor'],future_to_ode=norms['ode'],no_future_sensor=True,future_loss_used=task['group']!='B'))
    structures={name:layout(t,cases,'cpu') for name,cases in [('train',sp['train']),('test_diagnostic',sp['test'])]}
    configs=[(task['alpha'],True)]
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4)
    with EpochRecorder(d,t,b,sc,sp,[1,2,3],False) as rawrec, gzip.open(d/'fusion_epoch_predictions.csv.gz','wt',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['epoch','split','fusion','task','case','origin_run','target_run','horizon','target_vb_mm','predicted_vb_mm'])
        rawrec.record(model,0);record(model,b,t,sc,structures,configs,0,writer)
        for epoch in range(1,epochs+1):
            ts=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
            current=(model.estimate(b,obs)-b['target'][obs]).square().mean()
            future=(model.forecast(b,pairs)-b['target'][pairs[:,1]]).square().mean() if task['group']!='B' else torch.zeros(())
            loss=current+future;assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            row=dict(epoch=epoch,loss=float(loss.detach()),loss_current=float(current.detach()),loss_future=float(future.detach()))
            row.update(rawrec.record(model,epoch))
            row.update({'fusion_'+k:v for k,v in record(model,b,t,sc,structures,configs,epoch,writer).items()})
            stream.flush();row['seconds']=time.perf_counter()-ts
            pd.DataFrame([row]).to_csv(d/'history.csv',index=False,mode='w' if epoch==1 else 'a',header=epoch==1)
            ep=d/'epochs';ep.mkdir(exist_ok=True);torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},ep/f'epoch_{epoch:04d}.pt')
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    changes={prefix:max(float(abs(v-initial[k]).max()) for k,v in state.items() if k.startswith(prefix+'.')) for prefix in ['monitor','dynamics']}
    assert changes['monitor']>0 and (changes['dynamics']>0 if task['group']=='D' else changes['dynamics']==0)
    torch.save(dict(model=state,split=sp,scalers=sc,task=task,epoch=epochs),d/'last.pt')
    restored,_,_,_=prepare(task);restored.load_state_dict(torch.load(d/'last.pt',map_location='cpu',weights_only=True)['model'])
    model.eval();restored.eval()
    with torch.no_grad():torch.testing.assert_close(model.forecast(b,pairs),restored.forecast(b,pairs),rtol=0,atol=0)
    write_json(d/'metrics.json',dict(epochs=epochs,seconds=time.perf_counter()-started,changes=changes,reload_verified=True))
    write_json(d/'status.json',dict(status='completed'));print('COMPLETE '+task['id'],flush=True)

