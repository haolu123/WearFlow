"""Raw waveform adapter: shared cases/quality, independent train-only raw scaler."""
from wearflow_paths import ROOT as _WF_ROOT
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from method import adaptive_labels,crop_windows,trailing_average
ROOT=Path(__file__).resolve().parent
CHANNELS=('smcAC','smcDC','vib_table','vib_spindle','AE_table','AE_spindle')

def build(fold):
    base=ROOT.parent/'data'/fold/'final';a=dict(np.load(base/'arrays.npz'));meta=json.loads((base/'metadata.json').read_text())
    records=loadmat(str(_WF_ROOT / 'mill/mill.mat'),squeeze_me=True,struct_as_record=False)['mill']
    bykey={(int(r.case),int(r.run)):r for r in records}
    # Restrict all channels by the shared valid-sensor record rule. No hidden new QA.
    ids=np.flatnonzero(a['sensor_valid'] & np.isin(a['case'],meta['split']['train']+meta['split']['test']))
    ids=sorted(ids,key=lambda i:(a['case'][i],a['run'][i]))
    windows=[];crops=[];keys=[]
    for i in ids:
        r=bykey[(int(a['case'][i]),int(a['run'][i]))]
        signal=np.column_stack([getattr(r,k) for k in CHANNELS])
        w=crop_windows(signal,r.feed);assert w.shape==(75,300,6) and np.isfinite(w).all()
        windows.append(w);keys.append(i)
        if r.case in meta['split']['train']:
            begin=2000 if np.isclose(r.feed,.5) else 4000
            crops.append(signal[begin:begin+4000])
    raw=np.concatenate(crops);mean=raw.mean(0);std=raw.std(0);std=np.maximum(std,1e-8)
    x=((np.stack(windows)-mean)/std).astype('float32');keys=np.asarray(keys)
    train_record=np.isin(a['case'][keys],meta['split']['train'])
    # Label correction can use the complete TRAIN case offline; test targets untouched.
    labels=np.full((len(keys),75),np.nan,dtype='float32');anchor_log={}
    for case in meta['split']['train']:
        allidx=np.flatnonzero(a['case']==case);allidx=allidx[np.argsort(a['run'][allidx])]
        # Existing exclusion remains: only exported pool defines available label anchors.
        original=np.where(a['train_mask'][allidx],a['y_mm'][allidx],np.nan)
        corrected,keep=adaptive_labels(a['run'][allidx],original)
        anchor_log[str(case)]={'run':a['run'][allidx].tolist(),'original':original.tolist(),'corrected':corrected.tolist(),'retained_runs':a['run'][allidx][keep].tolist()}
        for j,i in enumerate(keys):
            if a['case'][i]!=case:continue
            # Raw crop endpoints map monotonically onto the preceding unit cut interval.
            # 300,350,...,4000 /4000; final window label equals current corrected label.
            q=a['run'][i]-1+np.arange(300,4001,50)/4000
            labels[j]=np.interp(q,a['run'][allidx],corrected)
    assert np.isfinite(labels[train_record]).all()
    return dict(x=x,window_y=labels,record_indices=keys,train_record=train_record,a=a,meta=meta,mean=mean,std=std,anchors=anchor_log)

def cut_predictions(window_pred,data):
    values=np.asarray(window_pred).reshape(-1,75);out=np.empty(len(values))
    case=data['a']['case'][data['record_indices']]
    for c in np.unique(case):
        ids=np.flatnonzero(case==c)
        smoothed=trailing_average(values[ids].reshape(-1),150).reshape(-1,75)
        out[ids]=smoothed[:,-1]
    return out
