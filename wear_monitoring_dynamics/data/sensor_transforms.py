"""Serializable training-only feature transforms for controlled preprocessing studies."""
import numpy as np

def apply_sensor_transform(x,state):
    y=np.asarray(x,dtype=float).copy()
    if not state:return y
    idx=state['indices']
    if state['method']=='log_zscore':
        if np.any(y[:,idx]<0):raise ValueError('Log-designated features must be nonnegative')
        y[:,idx]=np.log1p(y[:,idx]/np.asarray(state['log_scales']))
    elif state['method']=='clip_zscore':
        y[:,idx]=np.clip(y[:,idx],state['lower'],state['upper'])
    return y

def fit_sensor_transform(x,config):
    method=config['method'];idx=config['indices']
    if method not in ['zscore','log_zscore','robust','clip_zscore']:raise ValueError(method)
    state=dict(method=method,indices=idx)
    if method=='log_zscore':
        scales=[]
        for j in idx:
            positive=x[:,j][x[:,j]>0];scales.append(max(float(np.median(positive)) if len(positive) else 1.,1e-12))
        state['log_scales']=scales
    if method=='clip_zscore':
        state['quantiles']=config.get('quantiles',[.01,.99])
        state['lower'],state['upper']=[a.tolist() for a in np.quantile(x[:,idx],state['quantiles'],axis=0)]
    values=apply_sensor_transform(x,state);center=values.mean(0);scale=np.maximum(values.std(0),1e-8)
    if method=='robust':
        center[idx]=np.median(values[:,idx],axis=0)
        scale[idx]=np.maximum(np.quantile(values[:,idx],.75,axis=0)-np.quantile(values[:,idx],.25,axis=0),1e-8)
    return state,(center.tolist(),scale.tolist())
