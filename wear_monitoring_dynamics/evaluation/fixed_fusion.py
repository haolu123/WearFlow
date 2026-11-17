"""Causal fixed-gain filtering in physical VB units; no label access."""
import numpy as np

def rollout(times, observations, valid, alphas, monotone, initial, propagate):
    times=np.asarray(times,float);observations=np.asarray(observations,float);valid=np.asarray(valid,bool)
    alphas=np.asarray(alphas,float);monotone=np.asarray(monotone,bool);initial=np.asarray(initial,float)
    if observations.shape!=(len(times),len(initial)) or valid.shape!=observations.shape:raise ValueError('Shape mismatch')
    if np.any(np.diff(times)<=0) or np.any((alphas<0)|(alphas>1)):raise ValueError('Invalid time/gain')
    prior=np.empty_like(observations);post=np.empty_like(observations);prior[0]=post[0]=initial
    for i in range(1,len(times)):
        prior[i]=propagate(post[i-1],times[i]-times[i-1])
        gain=np.where(valid[i],alphas,0.)
        update=prior[i]+gain*(np.where(valid[i],observations[i],prior[i])-prior[i])
        post[i]=np.where(monotone,np.maximum(post[i-1],update),update)
    if not np.isfinite(post).all():raise FloatingPointError('Nonfinite fused state')
    return prior,post

