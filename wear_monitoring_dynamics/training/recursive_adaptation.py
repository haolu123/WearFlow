"""Differentiable case-batched propagation/correction; physical-scale-equivalent z space."""

import json,time,gzip,csv

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from ..config import write_json,file_hash

from ..data.datasets import observations,pair_tensors

from .single_shot_adaptation import load_parent

from .reproducibility import setup,synchronize



CONFIGS=[(a,m) for a in [.25,.5,.75] for m in [False,True]]



def layout(table,cases,device):

    seqs=[np.flatnonzero(table['case']==c) for c in cases]

    seqs=[s[np.argsort(table['time'][s])] for s in seqs]

    assert all(np.all(np.diff(table['time'][s])>0) for s in seqs)

    initial={int(s[0]) for s in seqs};ids=observations(table,cases);ids=np.array([i for i in ids if i not in initial],dtype=int)

    pairs=pair_tensors(table,cases,[1,2,3],device)

    mask=torch.tensor([int(i) not in initial for i in pairs[:,0]],device=device);pairs=pairs[mask]

    steps=[]

    for pos in range(1,max(map(len,seqs))):

        active=[s for s in seqs if len(s)>pos];steps.append((np.array([s[pos-1] for s in active]),np.array([s[pos] for s in active])))

    return dict(seqs=seqs,initial=initial,ids=ids,pairs=pairs,steps=steps,monitor_ids=observations(table,cases))





def rollout(model,batch,raw,structure,configs):

    """No target VB access; later sensors cannot change earlier posteriors."""

    n=len(configs);states=list(raw.expand(-1,n).unbind(0));alphas=torch.tensor([a for a,m in configs],device=raw.device,dtype=raw.dtype)[None,:];clamp=torch.tensor([m for a,m in configs],device=raw.device)[None,:]

    for prev,ids in structure['steps']:

        old=torch.stack([states[i] for i in prev]);conditions=batch['condition'][ids].repeat_interleave(n,dim=0)

        dt=(batch['time'][ids]-batch['time'][prev]).repeat_interleave(n,dim=0)

        prior=model.dynamics.propagate(old.reshape(-1,1),conditions,dt).reshape(-1,n)

        gain=alphas*batch['sensor_valid'][ids,None];post=prior+gain*(raw[ids]-prior)

        post=torch.where(clamp,torch.maximum(old,post),post)

        for i,v in zip(ids,post.unbind(0)):states[i]=v

    return torch.stack(states)





def future(model,batch,states,pairs):

    n=states.shape[1];a,b=pairs[:,0],pairs[:,1]

    return model.dynamics.propagate(states[a].reshape(-1,1),batch['condition'][a].repeat_interleave(n,0),(batch['time'][b]-batch['time'][a]).repeat_interleave(n,0)).reshape(-1,n)





def case_loss(error,cases):

    return torch.stack([error[cases==c].square().mean() for c in np.unique(cases)]).mean()





def prepare(task,device):

    model,b,t,parent=load_parent(task['sensor_dir'],task['ode_dir'],device)

    state=torch.load(task['start_checkpoint'],map_location=device,weights_only=True)

    assert state['split']==parent['split'] and state['scalers']==parent['scalers']

    model.load_state_dict(state['model']);return model,b,t,parent





def record(model,b,t,sc,structures,configs,epoch,writer):

    model.eval();row={};n=len(configs)

    with torch.no_grad():

        raw=model.estimate(b,np.arange(len(t['case'])))

        for split,st in structures.items():

            states=rollout(model,b,raw,st,configs);f=future(model,b,states,st['pairs'])

            for name,indices,targets,horizons,values in [('monitor',st['monitor_ids'],st['monitor_ids'],np.zeros(len(st['monitor_ids']),int),raw[st['monitor_ids']].expand(-1,n)),('current',st['ids'],st['ids'],np.zeros(len(st['ids']),int),states[st['ids']]),('future',st['pairs'][:,0].cpu().numpy(),st['pairs'][:,1].cpu().numpy(),st['pairs'][:,2].cpu().numpy(),f)]:

                pred=values.cpu().numpy()*sc['wear'][1]+sc['wear'][0];truth=t['VB'][targets];assert np.isfinite(pred).all()

                for k,(alpha,clamp) in enumerate(configs):

                    label=f'a{alpha:g}_m{int(clamp)}';err=abs(pred[:,k]-truth);case_mae=np.mean([err[t['case'][indices]==c].mean() for c in np.unique(t['case'][indices])]);row[f'{split}_{label}_{name}_mae']=float(case_mae)

                    writer.writerows(zip([epoch]*len(indices),[split]*len(indices),[label]*len(indices),[name]*len(indices),t['case'][indices].astype(int),t['run'][indices].astype(int),t['run'][targets].astype(int),horizons,truth,pred[:,k]))

    return row





def execute(task,directory,epochs):

    d=Path(directory);d.mkdir(parents=True,exist_ok=True);identity=dict(task=task,epochs=epochs,lr=1e-4,weight_decay=1e-4,start_hash=file_hash(task['start_checkpoint']),code_hash=file_hash(__file__))

    if (d/'status.json').exists() and json.loads((d/'status.json').read_text())['status']=='completed':

        assert json.loads((d/'identity.json').read_text())==identity;return

    write_json(d/'identity.json',identity);write_json(d/'status.json',dict(status='training'));start=time.perf_counter();dev=setup(task['seed'],task.get('device','cuda'),True);torch.set_num_threads(1)

    model,b,t,parent=prepare(task,dev);sp=parent['split'];sc=parent['scalers'];structures={name:layout(t,cases,dev) for name,cases in [('train',sp['train']),('validation',sp['validation']),('test_diagnostic',sp['outer_test_reserved'])] if cases};st=structures['train']

    configs=CONFIGS if task['mode']=='single' else [(task['alpha'],task['clamp'])];optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)

    # One full-case gradient and causal audit, with labels entirely absent from rollout.

    model.zero_grad(set_to_none=True);raw=model.estimate(b,np.arange(len(t['case'])));states=rollout(model,b,raw,st,configs);last=int(st['seqs'][0][-1]);first=int(st['seqs'][0][0])

    initial_grad=torch.autograd.grad(states[last].sum(),raw,retain_graph=True)[0][first].abs().sum();assert torch.isfinite(initial_grad)

    (states[st['ids']]-b['target'][st['ids']]).square().mean().backward();norms={name:sum(float(p.grad.square().sum()) for p in branch.parameters() if p.grad is not None)**.5 for name,branch in [('monitor',model.monitor),('ode',model.dynamics)]};assert all(v>0 for v in norms.values())

    with torch.no_grad():

        modified=raw.detach().clone();modified[last]+=100;changed=rollout(model,b,modified,st,configs);ids=st['seqs'][0][:-1];torch.testing.assert_close(changed[ids],states.detach()[ids],rtol=0,atol=0)

        zero=rollout(model,b,raw.detach(),st,[(0.,False)]);one=rollout(model,b,raw.detach(),st,[(1.,False)])

        valid=st['ids'][t['sensor_valid'][st['ids']]];torch.testing.assert_close(one[valid],raw.detach()[valid],rtol=1e-5,atol=1e-6)

        for a,z in st['steps']:

            expected=model.dynamics.propagate(zero[a],b['condition'][z],b['time'][z]-b['time'][a]);torch.testing.assert_close(zero[z],expected)

    write_json(d/'gradient_audit.json',dict(**norms,initial_to_last_gradient=float(initial_grad),no_future_sensor=True,endpoints=True))

    with gzip.open(d/'epoch_predictions.csv.gz','wt',newline='') as stream:

        w=csv.writer(stream);w.writerow(['epoch','split','fusion','task','case','origin_run','target_run','horizon','target_vb_mm','predicted_vb_mm']);record(model,b,t,sc,structures,configs,0,w)

        valid=st['monitor_ids'][t['sensor_valid'][st['monitor_ids']]];pairs=st['pairs']

        for epoch in range(1,epochs+1):

            synchronize(dev);ts=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True);raw=model.estimate(b,np.arange(len(t['case'])))

            states=raw if task['mode']=='single' else rollout(model,b,raw,st,configs)

            lmonitor=case_loss(raw[valid]-b['target'][valid],t['case'][valid]);lcurrent=case_loss(states[st['ids']]-b['target'][st['ids']],t['case'][st['ids']]);fp=future(model,b,states,pairs);lfuture=case_loss(fp-b['target'][pairs[:,1]],t['case'][pairs[:,0].cpu().numpy()]);loss=lmonitor+lcurrent+lfuture

            assert torch.isfinite(loss);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);optimizer.step()

            row=dict(epoch=epoch,loss=float(loss.detach()),monitor_loss=float(lmonitor.detach()),current_loss=float(lcurrent.detach()),future_loss=float(lfuture.detach()));row.update(record(model,b,t,sc,structures,configs,epoch,w));stream.flush();synchronize(dev);row['seconds']=time.perf_counter()-ts;pd.DataFrame([row]).to_csv(d/'history.csv',mode='w' if epoch==1 else 'a',header=epoch==1,index=False)

            if task['role']=='final':

                ep=d/'epochs';ep.mkdir(exist_ok=True);torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},ep/f'epoch_{epoch:04d}.pt')

    torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},split=sp,scalers=sc,task=task,epoch=epochs),d/'last.pt')

    restored,_,_,_=prepare(task,dev);restored.load_state_dict(torch.load(d/'last.pt',map_location=dev,weights_only=True)['model']);model.eval();restored.eval()

    with torch.no_grad():

        a=rollout(model,b,model.estimate(b,np.arange(len(t['case']))),st,configs);z=rollout(restored,b,restored.estimate(b,np.arange(len(t['case']))),st,configs);torch.testing.assert_close(a,z,rtol=0,atol=0)

    write_json(d/'metrics.json',dict(epochs=epochs,seconds=time.perf_counter()-start,reload_verified=True));write_json(d/'status.json',dict(status='completed'));print('COMPLETE '+task['id'],flush=True)

