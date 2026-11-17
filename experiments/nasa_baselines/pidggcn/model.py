"""Chen 2025 PIDGGCN equations with explicit causal NASA adaptation.

Features are nodes following Section2.3 and Eq16; each layer recomputes
prefix cosine adjacency across feature trajectories. This is
not claimed to recover unspecified author code tensor layouts exactly.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import least_squares


def dynamic_adjacency(x):
    """Eq15--20; x=[available times,features], output=[features,features].

    Features are L2-normalized columns; zero logits remain positive edges.
    Caller must provide only the current prefix for causal inference.
    """
    # Smooth norm is required for second derivatives when dropout zeros an
    # entire feature column. clamp(norm,eps) still backprops sqrt(0) twice.
    normalized=x/torch.sqrt(x.square().sum(dim=0,keepdim=True)+1e-12)
    similarity=normalized.T @ normalized
    std=torch.sqrt(((similarity-similarity.mean())**2).mean()+1e-12)
    threshold=similarity.mean()+.5*std
    return torch.softmax(F.relu(similarity-threshold),dim=-1)


class DynamicGatedLayer(nn.Module):
    """Eq21--25 GRU gates applied to graph-aggregated chronological rows."""
    def __init__(self, inputs, hidden):
        super().__init__()
        self.hidden=hidden
        self.wz=nn.Linear(inputs,hidden); self.uz=nn.Linear(hidden,hidden,bias=False)
        self.wr=nn.Linear(inputs,hidden); self.ur=nn.Linear(hidden,hidden,bias=False)
        self.wc=nn.Linear(inputs,hidden); self.uc=nn.Linear(hidden,hidden,bias=False)

    def forward(self, x, causal=True):
        state=x.new_zeros(self.hidden)
        states=[];graphs=[]
        for i in range(len(x)):
            adjacency=dynamic_adjacency(x[:i+1] if causal else x)
            graphs.append(adjacency)
            message=adjacency @ x[i]
            z=torch.sigmoid(self.wz(message)+self.uz(state))
            r=torch.sigmoid(self.wr(message)+self.ur(state))
            candidate=torch.tanh(self.wc(message)+self.uc(r*state))
            state=(1-z)*state+z*candidate
            states.append(state)
        return torch.stack(states),torch.stack(graphs)


class PIDGGCN(nn.Module):
    """HMoTP Table2 widths; x=[12 current features,4 condition,time]."""
    def __init__(self, inputs=17, hidden=(64,64), dropout=.5):
        super().__init__()
        self.layers=nn.ModuleList([DynamicGatedLayer(inputs,hidden[0]),DynamicGatedLayer(hidden[0],hidden[1])])
        self.dropout=nn.Dropout(dropout)
        self.head=nn.Sequential(nn.Linear(hidden[-1],64),nn.ReLU(),nn.Dropout(dropout),
                                nn.Linear(64,32),nn.ReLU(),nn.Dropout(dropout),nn.Linear(32,1))

    def forward(self,x,causal=True,return_graphs=False):
        graphs=[]
        for layer in self.layers:
            x,graph=layer(x,causal);graphs.append(graph);x=self.dropout(x)
        y=self.head(x).squeeze(-1)
        return (y,graphs) if return_graphs else y


def fit_prior(time, wear, case, time_scale=None):
    """Train-label-only pooled nonlinear fit of Eq1, stable u=t/T units.

    Eq1 in physical units is equivalent after coefficient transformation;
    case-equal residual weighting is an explicit NASA adaptation.
    """
    time,wear,case=map(np.asarray,(time,wear,case))
    assert np.isfinite(time).all() and np.isfinite(wear).all()
    scale=max(float(time.max()),1.) if time_scale is None else float(time_scale)
    u=time/scale
    positive=(u>0)&(wear>0)
    assert positive.sum()>=3
    design=np.column_stack([np.ones(positive.sum()),u[positive],u[positive]**2])
    init=np.linalg.lstsq(design,np.log(wear[positive]/u[positive]),rcond=None)[0]
    init=np.clip(init,-29,29)
    counts={c:(case==c).sum() for c in np.unique(case)}
    weights=np.array([1/np.sqrt(counts[c]) for c in case])
    result=least_squares(lambda abc:weights*(u*np.exp(abc[0]+abc[1]*u+abc[2]*u*u)-wear),
                         init,bounds=(-30,30),max_nfev=10000)
    assert result.success
    return dict(abc=result.x.tolist(),time_scale=scale,success=bool(result.success),
                training_rmse_mm=float(np.sqrt(np.mean((u*np.exp(result.x[0]+result.x[1]*u+result.x[2]*u*u)-wear)**2))),
                bound_hit=bool((abs(result.x)>29.99).any()),n_labels=len(wear))


def fit_condition_priors(time,wear,case,condition,time_scale):
    """Fit training-only same-condition references; frozen sparse-data rule.

    Group keys are exact exported (DOC,feed,material). Fewer than4 positive
    labels or3 distinct positive times => pooled-training fallback. Failed or
    bound-hit fits also fall back; never consult validation/test labels.
    """
    pooled=fit_prior(time,wear,case,time_scale)
    groups={}
    for key in np.unique(condition,axis=0):
        mask=np.all(condition==key,axis=1)
        positive=mask&(time>0)&(wear>0)
        reason=None
        if positive.sum()<4 or len(np.unique(time[positive]))<3:
            reason='fewer_than_4_positive_labels_or_3_distinct_times'
        if reason is None:
            try:
                fitted=fit_prior(time[mask],wear[mask],case[mask],time_scale)
                if fitted['bound_hit']:reason='coefficient_bound_hit'
            except (ValueError,AssertionError,RuntimeError):reason='fit_failure'
        if reason is not None:fitted=dict(pooled)
        fitted=dict(fitted,condition=key.tolist(),group_labels=int(mask.sum()),
                    group_cases=np.unique(case[mask]).astype(int).tolist(),fallback_reason=reason)
        groups[tuple(key)]=fitted
    return pooled,groups


def prior_rate_u(u,abc,variant='paper_printed'):
    a,b,c=abc
    factor=1 if variant=='paper_printed' else 2
    assert variant in ('paper_printed','calculus_corrected')
    return (1+b*u+factor*c*u*u)*torch.exp(a+b*u+c*u*u)


def diagonal_time_derivative(prediction, time, indices=None, create_graph=True):
    """Own current-time partial derivative, not grad(sum(y)) across nodes."""
    if indices is None:indices=range(len(time))
    return torch.stack([torch.autograd.grad(prediction[i],time,retain_graph=True,create_graph=create_graph)[0][i] for i in indices])


def own_scalar_time_derivative(prediction, scalar_times, indices,create_graph=True):
    """Equivalent diagonal AD using independent scalar leaves, no detach."""
    return torch.stack([torch.autograd.grad(prediction[i],scalar_times[i],retain_graph=True,create_graph=create_graph)[0] for i in indices])


def physics_loss(prediction,target,derivative_u,u,abc,wear_std,variant='paper_printed'):
    data=F.mse_loss(prediction,target)
    reference=prior_rate_u(u,abc,variant)/wear_std
    physics=F.mse_loss(derivative_u,reference)
    monotonic=F.relu(-derivative_u).square().mean()
    return data,physics,monotonic


class HalfDecay:
    """Original5000 epoch schedule, optionally explicitly budget-rescaled.

    At100 epochs original schedule does not decay. Rescaled100 uses warmup50
    and checks every2, preserving fractions of5000/2500/100 source schedule.
    """
    def __init__(self,total_epochs=5000,rescale=False):
        self.warmup=total_epochs//2 if rescale else 2500
        self.interval=max(1,total_epochs//50) if rescale else 100
        self.data=1.;self.physics=1.
    def step(self,epoch,data,physics):
        if epoch>self.warmup and epoch%self.interval==0:
            if self.data*data>=2*self.physics*physics:self.data*=.5
            elif self.physics*physics>=2*self.data*data:self.physics*=.5
        return self.data,self.physics
