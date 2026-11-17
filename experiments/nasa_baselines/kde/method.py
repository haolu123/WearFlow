"""Kim et al. DOI 10.1007/s40684-025-00827-9: declared reconstruction.

See FULLTEXT_MAPPING.md for published equations versus engineering assumptions.
No fitting function consumes evaluation wear labels implicitly.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize, LinearConstraint, minimize_scalar
from scipy.stats import gaussian_kde
import torch
from torch import nn


def adaptive_labels(time, wear):
    """Remove downward observations relative to last retained anchor; interpolate.

    Exact deletion order/boundaries are not specified in Section 3.1.
    Our deterministic completion retains equal anchors and uses endpoint holds.
    Use only within a training trajectory; never alter evaluation targets.
    """
    t,y=np.asarray(time,float),np.asarray(wear,float)
    if len(t)!=len(y) or np.any(np.diff(t)<=0):raise ValueError('Sorted unique times required')
    keep=[]
    for i in np.flatnonzero(np.isfinite(y)):
        if not keep or y[i]>=y[keep[-1]]:keep.append(i)
    if not keep:raise ValueError('No labels')
    return np.interp(t,t[keep],y[keep]),np.asarray(keep,int)


def crop_windows(signal, feed, fs=250, window=300, hop=50):
    """Paper p883: 8:24 sec high feed, 16:32 sec low feed, 300/50."""
    x=np.asarray(signal)
    if x.ndim==1:x=x[:,None]
    if np.isclose(feed,.5):start,end=8,24
    elif np.isclose(feed,.25):start,end=16,32
    else:raise ValueError('Paper supports feed .5 or .25')
    if len(x)<end*fs:raise ValueError('Signal too short for paper crop; do not pad silently')
    x=x[start*fs:end*fs]
    return np.stack([x[i:i+window] for i in range(0,len(x)-window+1,hop)])


def trailing_average(values, width):
    """Causal completion of unspecified moving-average alignment/boundaries."""
    x=np.asarray(values,float);s=np.r_[0.,np.cumsum(x)];i=np.arange(len(x));lo=np.maximum(0,i-width+1)
    return (s[i+1]-s[lo])/(i-lo+1)


@dataclass(frozen=True)
class TransformerConfig:
    channels:int=6
    embedding:int=20  # inferred scale, not a published architecture detail
    heads:int=4
    layers:int=5      # p883 selected value
    feedforward:int=2048  # p883 selected value
    dropout:float=.1


class WearTransformer(nn.Module):
    def __init__(self,cfg=TransformerConfig()):
        super().__init__();self.cfg=cfg
        self.project=nn.Linear(cfg.channels,cfg.embedding)
        layer=nn.TransformerEncoderLayer(cfg.embedding,cfg.heads,cfg.feedforward,cfg.dropout,batch_first=True)
        self.encoder=nn.TransformerEncoder(layer,cfg.layers,enable_nested_tensor=False)
        self.readout=nn.Linear(cfg.embedding,1)
    def forward(self,x):
        if x.ndim!=3:raise ValueError('Expected batch, time-in-current-window, channels')
        z=self.project(x);pos=torch.arange(z.shape[1],device=z.device,dtype=z.dtype)[:,None]
        rates=torch.exp(torch.arange(0,self.cfg.embedding,2,device=z.device,dtype=z.dtype)*(-math.log(10000.)/self.cfg.embedding))
        pe=torch.zeros(z.shape[1],self.cfg.embedding,device=z.device,dtype=z.dtype)
        pe[:,0::2]=torch.sin(pos*rates);pe[:,1::2]=torch.cos(pos*rates[:pe[:,1::2].shape[1]])
        # Full attention is within an already available current raw window, not future cuts.
        return self.readout(self.encoder(z+pe).mean(dim=1))


class MonotoneCubic:
    """Eq7/8 no-intercept cubic; derivative constraints at observed fitting times.

    Constraints do NOT certify monotonicity outside or between sampled points.
    Internal time scaling changes conditioning only, not the cubic model family.
    """
    def fit(self,time,estimated_wear):
        t,y=np.asarray(time,float),np.asarray(estimated_wear,float)
        if len(t)!=len(y) or len(t)<3 or not np.isfinite(t).all() or not np.isfinite(y).all() or np.any(np.diff(t)<=0) or t[0]<0:raise ValueError('Need at least three sorted finite prefix points')
        self.scale=max(float(t[-1]),1.);u=t/self.scale
        A=np.column_stack([u**3,u**2,u]);D=np.column_stack([3*u*u,2*u,np.ones_like(u)])
        if np.linalg.matrix_rank(A)<3:raise ValueError('Prefix cubic design is rank deficient')
        # Nonnegative linear start; underdetermined early prefixes are excluded.
        start=np.array([0.,0.,max(float(u@y/max(u@u,1e-12)),0.)])
        fun=lambda c:.5*np.sum((A@c-y)**2)
        result=minimize(fun,start,jac=lambda c:A.T@(A@c-y),constraints=LinearConstraint(D,0,np.inf),method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
        if not result.success or (D@result.x).min() < -1e-8:raise RuntimeError(result.message)
        self.coefficients=result.x;self.end=float(t[-1]);self.fit_points=len(t)
        return self
    def predict(self,time):
        u=np.asarray(time,float)/self.scale;a,b,c=self.coefficients
        return ((a*u+b)*u+c)*u
    def derivative(self,time):
        u=np.asarray(time,float)/self.scale;a,b,c=self.coefficients
        return (3*a*u*u+2*b*u+c)/self.scale
    def crossing(self,threshold,after=None):
        # Paper predicts total operating/cut arrival time: an early crossing is an
        # error/negative remaining life, not grounds to selectively discard a member.
        after=0. if after is None else float(after)
        roots=np.roots([*self.coefficients,-float(threshold)])*self.scale
        valid=[float(z.real) for z in roots if abs(z.imag)<1e-7 and z.real>=after-1e-8 and self.derivative(z.real)>=-1e-9]
        return min(valid) if valid else np.nan


def kde_mode(samples,bandwidth='scott'):
    """Eq9/10 mode over threshold-crossing times, not VB.

    'literal_eq9' implements printed exp(-pi*z^2), sigma=sqrt(2*pi).
    Scott is an explicit bandwidth completion; paper does not specify h.
    """
    x=np.asarray(samples,float);x=x[np.isfinite(x)]
    if not len(x):return np.nan
    if len(x)==1 or np.ptp(x)<1e-12:return float(x[0])
    if bandwidth=='literal_eq9':
        sigma=np.sqrt(2*np.pi)
        density=lambda t:np.mean(np.exp(-.5*((np.atleast_1d(t)[:,None]-x)/sigma)**2),axis=1)/(sigma*np.sqrt(2*np.pi))
    else:
        kde=gaussian_kde(x,bw_method=bandwidth);density=lambda t:kde(np.atleast_1d(t))
    grid=np.linspace(x.min(),x.max(),4097);v=density(grid)
    candidates=[0,len(grid)-1]+list(np.flatnonzero((v[1:-1]>=v[:-2])&(v[1:-1]>=v[2:]))+1)
    modes=[]
    for i in candidates:
        lo,hi=grid[max(0,i-1)],grid[min(len(grid)-1,i+1)]
        opt=minimize_scalar(lambda t:-float(density(t)[0]),bounds=(lo,hi),method='bounded')
        modes.append((float(density(opt.x)[0]),float(opt.x)))
    return max(modes)[1]


def forecast_ensemble(prefix_time,member_vb,future_time,threshold=None,bandwidth='scott'):
    curves=[MonotoneCubic().fit(prefix_time,y) for y in np.asarray(member_vb)]
    forecasts=np.stack([c.predict(future_time) for c in curves])
    result={'member_future_vb':forecasts,'mean_future_vb':forecasts.mean(axis=0),'curves':curves}
    if threshold is not None:
        crossing=np.array([c.crossing(threshold) for c in curves])
        result.update(crossing_times=crossing,kde_crossing_time=kde_mode(crossing,bandwidth),valid_crossings=int(np.isfinite(crossing).sum()))
    return result
