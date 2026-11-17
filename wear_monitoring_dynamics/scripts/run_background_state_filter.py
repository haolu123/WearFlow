"""Positive-rate neural state dynamics + condition-dependent sensor observation.

Train-only affine observation and rank-two static random effects. Causal Kalman
assimilation, not supervised test adaptation or bidirectional smoothing.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from ..config import project_path,write_json,file_hash
from .run_sensor_mechanisms import context,OLD

ROOT=project_path('wear_monitoring_dynamics/outputs/background_state_filter_20261012')
KINDS=['sensor_filter','background_filter','separated_background_filter']
GRID=[dict(name='off',q=0.,r_scale=0.)]+[
    dict(name=f'q{q:g}_r{r:g}',q=q,r_scale=r) for q in [.001,.01,.1] for r in [1.,10.,100.]]


class PositiveRate(nn.Module):
    """dw/dtau=softplus(f(c,tau)), w(0)=softplus(g(c)); w is VB/train SD."""
    def __init__(self):
        super().__init__()
        self.initial=nn.Sequential(nn.Linear(4,8),nn.SiLU(),nn.Linear(8,1))
        self.rate=nn.Sequential(nn.Linear(5,16),nn.SiLU(),nn.Linear(16,16),nn.SiLU(),nn.Linear(16,1))
        with torch.no_grad():
            self.initial[-1].weight.zero_();self.initial[-1].bias.fill_(-5.)
            self.rate[-1].bias.fill_(-2.)

    def forward(self,condition,nodes,quad_weights,dt,cumulative):
        c=condition[:,None,:].expand(-1,nodes.shape[1],-1)
        x=torch.cat([c,nodes[:,:,None]],dim=-1)
        speed=torch.nn.functional.softplus(self.rate(x).squeeze(-1))
        increments=dt*(speed*quad_weights).sum(1)
        initial=torch.nn.functional.softplus(self.initial(condition)).ravel()
        return initial+cumulative@increments


def dynamics_inputs(c):
    table=c['table'];tau=table['time']/c['scalers']['time_scale'];dt=np.zeros(len(tau));cumulative=np.zeros((len(tau),len(tau)))
    for case in np.unique(table['case']):
        ids=np.flatnonzero(table['case']==case);ids=ids[np.argsort(table['run'][ids])]
        dt[ids]=np.diff(np.r_[0.,tau[ids]])
        assert (dt[ids]>=0).all()
        cumulative[np.ix_(ids,ids)]=np.tril(np.ones((len(ids),len(ids))))
    x,w=np.polynomial.legendre.leggauss(4);nodes=tau[:,None]-dt[:,None]+dt[:,None]*(x[None,:]+1)/2
    return (c['ct'][:,:4],torch.tensor(nodes,dtype=torch.float32),torch.tensor(w/2,dtype=torch.float32),
            torch.tensor(dt,dtype=torch.float32),torch.tensor(cumulative,dtype=torch.float32))


def train_dynamics(c,out,epochs):
    row=c['row'];scope=f'{row.fold}/{row.role}_{int(row.holdout) if pd.notna(row.holdout) else "all"}/seed_{int(row.seed)}'
    d=out/'dynamics'/scope;d.mkdir(parents=True,exist_ok=True)
    ident=dict(epochs=epochs,seed=int(row.seed),split=c['split'],source_run=row.run_dir,train_indices=c['train'].tolist(),
        equation='d(VB/sigma)/d(t/time_scale)=softplus(f(c,t/time_scale)); positive learned initial state',
        quadrature='4-point Gauss-Legendre on each observed interval',lr=.001,weight_decay=.0001)
    if (d/'complete.json').exists():
        assert json.loads((d/'identity.json').read_text())==ident
        return d
    torch.manual_seed(int(row.seed));model=PositiveRate();inputs=dynamics_inputs(c)
    scale=c['scalers']['wear'][1];target=torch.tensor(c['table']['VB']/scale,dtype=torch.float32);ids=c['train']
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    pred=np.empty((epochs,len(target)),np.float32);states=[];history=[];start=time.perf_counter()
    for epoch in range(1,epochs+1):
        optimizer.zero_grad(set_to_none=True);y=model(*inputs);loss=(y[ids]-target[ids]).square().mean()
        assert torch.isfinite(loss);loss.backward();optimizer.step()
        with torch.no_grad():p=model(*inputs).numpy()*scale
        pred[epoch-1]=p;states.append(torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy().copy())
        h=dict(epoch=epoch,loss=float(loss.detach()))
        for name,ix in c['evals'].items():
            if not len(ix):continue
            err=p[ix]-c['table']['VB'][ix];case=c['table']['case'][ix]
            h[name+'_mae']=float(np.mean([abs(err[case==k]).mean() for k in np.unique(case)]))
        history.append(h)
    assert np.isfinite(pred).all()
    for case in np.unique(c['table']['case']):
        ix=np.flatnonzero(c['table']['case']==case)
        assert np.min(np.diff(pred[:,ix],axis=1),initial=0)>-2e-6
    np.savez_compressed(d/'epoch_predictions.npz',predictions=pred,case=c['table']['case'],run=c['table']['run'],target=c['table']['VB'])
    np.savez_compressed(d/'epoch_weights.npz',weights=np.array(states))
    pd.DataFrame(history).to_csv(d/'history.csv',index=False);torch.save(model.state_dict(),d/'last.pt')
    check=PositiveRate();check.load_state_dict(torch.load(d/'last.pt',weights_only=True))
    with torch.no_grad():np.testing.assert_allclose(check(*inputs).numpy()*scale,pred[-1],atol=1e-7,rtol=1e-6)
    write_json(d/'identity.json',ident);write_json(d/'scalers.json',c['scalers'])
    write_json(d/'complete.json',dict(epochs=epochs,seconds=time.perf_counter()-start,parameters=sum(p.numel() for p in model.parameters()),reload_verified=True))
    return d


def fit_observation(c):
    """Only training labeled valid observations fit shared map/covariance/background."""
    ix=c['train'];cases=c['table']['case'][ix];w=c['table']['VB'][ix]/c['scalers']['wear'][1]
    cond=np.column_stack([np.ones(len(ix)),c['ct'][ix,:4].numpy()])
    design=np.column_stack([cond,w[:,None]*cond]);sensor=c['sensor'][ix,:18].numpy().astype(float)
    weight=np.array([1./(cases==k).sum() for k in cases]);ridge=.1
    beta=np.linalg.solve(design.T@(weight[:,None]*design)+ridge*np.eye(10),design.T@(weight[:,None]*sensor))
    error=sensor-design@beta
    means=np.array([error[cases==k].mean(0) for k in np.unique(cases)])
    beta[0]+=means.mean(0);error=sensor-design@beta
    means=np.array([error[cases==k].mean(0) for k in np.unique(cases)])
    within=np.zeros((18,18))
    for k in np.unique(cases):
        e=error[cases==k];e=e-e.mean(0);within+=e.T@e/len(e)
    within/=len(np.unique(cases))
    R=.5*within+.5*np.diag(np.diag(within))+.001*np.eye(18)
    L=np.linalg.cholesky(R);white=np.linalg.solve(L,means.T).T
    c_unique=np.array([cond[np.flatnonzero(cases==k)[0]] for k in np.unique(cases)])
    sensitivities=np.linalg.solve(L,(c_unique@beta[5:]).T)
    u,s,_=np.linalg.svd(sensitivities,full_matrices=False);rank=int((s>max(s[0]*1e-8,1e-10)).sum())
    projector=np.eye(18)-u[:,:rank]@u[:,:rank].T
    backgrounds={};overlap={}
    for name,samples in [('background_filter',white),('separated_background_filter',white@projector)]:
        cov=samples.T@samples/len(samples)
        eig,vec=np.linalg.eigh(cov);order=np.argsort(eig)[::-1][:2]
        Bwhite=vec[:,order]*np.sqrt(np.maximum(eig[order],0))
        backgrounds[name]=L@Bwhite
        U,sv,_=np.linalg.svd(Bwhite,full_matrices=False);rr=int((sv>1e-9).sum())
        overlap[name]=float(np.max(np.linalg.norm(U[:,:rr].T@sensitivities,axis=0)/np.maximum(np.linalg.norm(sensitivities,axis=0),1e-12))) if rr else 0.
    return dict(beta=beta,R=R,**backgrounds,train_cases=np.array(c['split']['train']),train_indices=np.array(ix),
        background_wear_overlap=np.array([overlap[k] for k in ['background_filter','separated_background_filter']]),
        observation_train_rmse=np.sqrt(np.mean(error**2,axis=0)),wear_sensitivity_rank=np.array(rank))


def filter_sequence(prior,dt,sensor,condition,valid,observation,kind,q,r_scale):
    """All arguments are deployable observations/prior; no labels, future smoothing.

    prior and returned state use VB/train-SD units. Fixed ODE increments propagate
    the current posterior. Background z is static with N(0,I) prior.
    """
    n=len(prior)
    if r_scale==0:
        return dict(wear=prior.copy(),sd=np.full(n,np.nan),background=np.zeros((n,2)),innovation=np.full(n,np.nan),correction=np.zeros(n))
    C=np.r_[1.,condition];a=C@observation['beta'][:5];b=C@observation['beta'][5:]
    B=np.empty((18,0)) if kind=='sensor_filter' else observation[kind]
    H=np.column_stack([b,B]);R=observation['R']*r_scale
    invRH=np.linalg.solve(R,H);information=H.T@invRH
    p=1+B.shape[1];m=np.zeros(p);m[0]=prior[0]
    P=np.diag([.25]+[1.]*(p-1));wear=[];sds=[];zs=[];innov=[];corrections=[]
    for i in range(n):
        if i:m[0]+=prior[i]-prior[i-1]
        P[0,0]+=q*max(float(dt[i]),0.)
        before=m[0]
        if valid[i]:
            residual=sensor[i]-a-H@m
            # Small state-space information update; correlated 18-D noise retained.
            post=np.linalg.solve(np.linalg.solve(P,np.eye(p))+information,np.eye(p))
            gain=post@invRH.T
            nis_cov=R+H@P@H.T
            nis=float(residual@np.linalg.solve(nis_cov,residual)/18)
            m=m+gain@residual;P=(post+post.T)/2
            m[0]=max(0.,m[0])  # Approximate nonnegative posterior mean projection.
        else:nis=np.nan
        assert np.isfinite(m).all() and np.min(np.linalg.eigvalsh(P))>-1e-9
        wear.append(m[0]);sds.append(np.sqrt(max(P[0,0],0)));z=np.zeros(2);z[:p-1]=m[1:];zs.append(z)
        innov.append(nis);corrections.append(m[0]-before)
    return dict(wear=np.array(wear),sd=np.array(sds),background=np.array(zs),innovation=np.array(innov),correction=np.array(corrections))


def evaluate_case(c,prior,observation,kind,hyper,case):
    t=c['table'];ix=np.flatnonzero(t['case']==case);ix=ix[np.argsort(t['run'][ix])]
    tau=t['time'][ix]/c['scalers']['time_scale'];dt=np.diff(np.r_[0.,tau]);scale=c['scalers']['wear'][1]
    out=filter_sequence(prior[ix]/scale,dt,c['sensor'][ix,:18].numpy(),c['ct'][ix[0],:4].numpy(),c['valid'][ix],
        observation,kind,hyper['q'],hyper['r_scale'])
    return ix,out


def train_all(out,epochs=500,smoke=False):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    snapshot=out/'runner_snapshot.py'
    if snapshot.exists():assert file_hash(snapshot)==file_hash(__file__)
    else:snapshot.write_bytes(Path(__file__).read_bytes())
    mf=pd.read_csv(OLD/'reused_sensor_manifest.csv')
    if smoke:mf=mf[(mf.fold=='fold_a')&(mf.role=='inner')].head(1)
    rows=[]
    write_json(out/'protocol.json',dict(epochs=epochs,filter_kinds=KINDS,grid=GRID,
        state='w=VB/train SD; static background z in R2; learned positive initial state, no true initial VB',
        dynamics='dw/dtau=softplus(f(condition,tau)); no dependence on w in this first model; 4-point positive quadrature per interval',
        observation='s=a(condition)+b(condition)*w+B*z+epsilon; ridge=.1, case-balanced supervised training labels only',
        covariance='R=.5 within-case residual covariance + .5 diagonal + .001 I; B from top2 training case-residual mean covariance directions',
        separated='project case residual directions orthogonal to training wear-sensitivity span in whitened observation space before PCA',
        filter='initial Pw=.25 in normalized wear variance, Pz=I; Qw=q*dt; Qz=0; invalid sensor skips update; wear posterior mean clipped at0',
        selection='primary fixed500; secondary prior epoch selected by inner prior MAE; q/R separately chosen on 7 inner cases per fold and family, off candidate included',
        limits='filter covariance is approximate and not calibrated uncertainty; posterior estimates can decrease despite nonnegative process drift; background offset only, not gain adaptation',
        label_policy='only training labels fit dynamics/observation; inner labels select hyperparameters; final filters accept no VB labels or future sensor',
        sources=['https://openreview.net/pdf?id=HyTqHL5xg'],source_note='General state-space motivation, not a reproduction of DVBF',source_sha256=file_hash(__file__)))
    for r in mf.itertuples():
        c=context(r);d=train_dynamics(c,out,epochs);obs=fit_observation(c);np.savez_compressed(d/'observation.npz',**obs)
        rows.append(dict(fold=r.fold,role=r.role,holdout=None if pd.isna(r.holdout) else int(r.holdout),seed=int(r.seed),run_dir=str(d),source_run=r.run_dir))
        pd.DataFrame(rows).to_csv(out/'study_manifest.csv',index=False)
        write_json(out/'status.json',dict(status='training_dynamics',completed=len(rows),planned=len(mf)))
        print(f'Dynamics {len(rows)}/{len(mf)} {r.fold} {r.role} holdout{r.holdout} seed{r.seed}',flush=True)
    write_json(out/'status.json',dict(status='dynamics_and_observation_completed',runs=len(rows),epochs=epochs))


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    out=ROOT.with_name(ROOT.name+'_smoke') if args.smoke else ROOT
    try:train_all(out,3 if args.smoke else 500,args.smoke)
    except Exception as e:write_json(out/'failure.json',dict(error=str(e)));raise


if __name__=='__main__':main()
