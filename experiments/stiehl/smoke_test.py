"""Development-only functional checks and timing. Never evaluates outer test tool 9."""
import argparse
import copy
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from data_protocol import ROOT,records,pairs,transform
from models import WearModel,objective

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=3);args=parser.parse_args()
    torch.set_num_threads(1)
    start=time.time();frame,x,_=records()
    scaler=json.loads((ROOT/'splits/test9_inner.json').read_text())
    xx,valid=transform(frame,x,scaler)
    assert scaler['train_tools']==list(range(1,8))
    assert valid.all(), 'Implement and audit frozen time-only fallback before training invalid sensors.'
    train=np.flatnonzero(frame.tool.isin(scaler['train_tools']))
    val=np.flatnonzero(frame.tool==8)
    # Exclude held-out test tool before constructing tensors or losses.
    train_frame=frame.iloc[train].reset_index(drop=True)
    pp=pairs(train_frame)
    tx=torch.tensor(xx[train]);vy=torch.tensor(frame.VB_mm.to_numpy()[val],dtype=torch.float32)
    ty=torch.tensor((train_frame.VB_mm.to_numpy()-scaler['wear_mean'])/scaler['wear_std'],dtype=torch.float32)
    tt=torch.tensor(train_frame.tool.to_numpy());pi=torch.tensor(pp[['origin','target']].to_numpy())
    dt=torch.tensor(pp.actual_seconds.to_numpy()/1000,dtype=torch.float32)
    out=ROOT/'smoke';out.mkdir(exist_ok=True)
    torch.manual_seed(0);base=WearModel();initial=copy.deepcopy(base.state_dict())
    rows=[];audits={}
    for mode in ['time','current','local','integral']:
        torch.manual_seed(0);model=WearModel(mode=='time')
        if mode!='time':model.load_state_dict(initial)
        before=copy.deepcopy(model.state_dict())
        opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
        for epoch in range(1,args.epochs+1):
            clock=time.time();opt.zero_grad()
            loss,cur,aux=objective(model,tx,ty,tt,pi,dt,mode)
            loss.backward()
            gradient={branch:float(sum(p.grad.square().sum() for p in getattr(model,branch).parameters() if p.grad is not None).sqrt()) if any(p.grad is not None for p in getattr(model,branch).parameters()) else 0 for branch in ['monitor','rate']}
            assert torch.isfinite(loss)
            opt.step()
            with torch.no_grad():
                pred=model(torch.tensor(xx[val]))*scaler['wear_std']+scaler['wear_mean']
                mae=(pred-vy).abs().mean().item()
            rows.append({'method':mode,'epoch':epoch,'current_loss':cur.item(),'auxiliary_loss':aux.item(),'validation_mae_mm':mae,'seconds':time.time()-clock,**{f'gradient_{k}':v for k,v in gradient.items()}})
        torch.save({'model':model.state_dict(),'scaler':scaler},out/f'{mode}.pt')
        clone=WearModel(mode=='time');clone.load_state_dict(torch.load(out/f'{mode}.pt',weights_only=False)['model'])
        assert torch.equal(clone(tx[:20]),model(tx[:20]))
        changed=any(not torch.equal(before[k],model.state_dict()[k]) for k in before if k.startswith('rate.'))
        assert changed==(mode in ['local','integral'])
        if mode in ['local','integral']:
            z=ty[:20].clone().requires_grad_(True)
            result=model.propagate(z,torch.ones(20)*.72)
            initial_gradient=torch.autograd.grad(result.sum(),z)[0]
            assert torch.isfinite(initial_gradient).all() and (initial_gradient.abs()>0).all()
            assert (result>=z).all()
            rk4diff=(result-model.propagate(z,torch.ones(20)*.72,steps=40)).abs().max().item()*scaler['wear_std']
        else:rk4diff=None
        audits[mode]={'reload_exact':True,'rate_changed':changed,'rk4_20_vs_40_max_mm':rk4diff}
        print(mode,rows[-1],flush=True)
    # A deterministic uniform-in-time subset keeps exact GPR cost bounded.
    chosen=[]
    for tool,g in train_frame.groupby('tool'):
        times=g.cumulated_tool_contact_time.to_numpy()
        ix=np.unique(np.searchsorted(times,np.linspace(times.min(),times.max(),128)))
        chosen.extend(g.index.to_numpy()[ix])
    gp=GaussianProcessRegressor(kernel=ConstantKernel(1)*RBF(1)+WhiteKernel(.01),alpha=1e-6,normalize_y=False,n_restarts_optimizer=0,random_state=0)
    clock=time.time();gp.fit(tx.numpy()[chosen],ty.numpy()[chosen]);pred=gp.predict(xx[val])*scaler['wear_std']+scaler['wear_mean']
    gp_audit={'train_subset_n':len(chosen),'seconds':time.time()-clock,'validation_mae_mm':float(np.mean(np.abs(pred-vy.numpy()))),'kernel':str(gp.kernel_)}
    pd.DataFrame(rows).to_csv(out/'epoch_metrics.csv',index=False)
    report={'epochs':args.epochs,'train_tools':list(range(1,8)),'validation_tool':8,'outer_test_tool_never_evaluated':9,'train_records':len(train),'train_pairs':len(pp),'audits':audits,'gpr':gp_audit,'total_seconds':time.time()-start,'device':'CPU, one torch thread'}
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
