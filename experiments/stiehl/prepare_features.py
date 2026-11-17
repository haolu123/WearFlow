"""Label-independent force features; run with the base Anaconda h5py environment."""
from pathlib import Path
import hashlib
import json
import time
import h5py
import numpy as np
import pandas as pd
from scipy import signal, stats

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / 'external_datasets/stiehl_milling_2023_v3'
BANDS = [(0,250),(250,500),(500,1000),(1000,2000),(2000,5000),(5000,10000)]
NAMES = ['mean','std','rms','ptp','median','iqr','skew','kurtosis','crest'] + [f'power_{a}_{b}' for a,b in BANDS] + ['centroid','entropy']

def extract(path):
    with h5py.File(path, 'r') as f:
        t = f['signals_sensor/time_sensor'][()].ravel()
        x = np.stack([f[f'signals_sensor/force_sensor_{a}'][()].ravel() for a in 'xyz'])
    if not np.isfinite(x).all() or not np.isfinite(t).all() or not (np.diff(t)>0).all():
        raise ValueError('nonfinite_or_invalid_clock')
    # Adjacent differences of float32 timestamps have quantization bias.
    # The full recording span gives a stable sample-rate estimate.
    fs = (len(t)-1) / (float(t[-1])-float(t[0]))
    x = x[:,int(.2*x.shape[1]):int(.8*x.shape[1])]
    if x.shape[1] < 2048 or np.all(np.std(x,axis=1)<1e-12):
        raise ValueError('short_or_flat')
    features=[]
    for v in x:
        rms=np.sqrt(np.mean(v*v))
        frequency, psd=signal.welch(v,fs=fs,window='hann',nperseg=2048,noverlap=1024,detrend='constant')
        df=frequency[1]-frequency[0]
        prob=psd/max(psd.sum(),1e-30)
        features.extend([v.mean(),v.std(),rms,np.ptp(v),np.median(v),stats.iqr(v),stats.skew(v) if v.std()>1e-12 else 0,stats.kurtosis(v) if v.std()>1e-12 else 0,np.max(np.abs(v))/max(rms,1e-12)])
        features.extend([psd[(frequency>=a)&(frequency<b)].sum()*df for a,b in BANDS])
        features.extend([(frequency*prob).sum(),-(prob*np.log(prob+1e-30)).sum()/np.log(len(prob))])
    values=np.asarray(features)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite_features')
    return values,fs

def main():
    start=time.time()
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if (ROOT/'features.npz').exists():
        audit=json.loads((ROOT/'feature_audit.json').read_text())
        if audit['source_sha256'] != source_hash:
            raise RuntimeError('Feature implementation changed: use a new output directory.')
        print('Verified existing feature archive; no extraction repeated.',flush=True)
        return
    index=pd.read_csv(DATA/'filelist.csv').sort_values(['tool','run']).reset_index(drop=True)
    files={p.name:p for p in (DATA/'data').rglob('*.h5')}
    values=np.full((len(index),51),np.nan)
    quality=[]
    for i,row in index.iterrows():
        try:
            values[i],fs=extract(files[row.filename]); reason='ok'
        except ValueError as exc:
            fs=np.nan;reason=str(exc)
        quality.append({'filename':row.filename,'valid':reason=='ok','reason':reason,'fs':fs})
        if (i+1)%250==0:
            print(f'{i+1}/{len(index)} files; {time.time()-start:.1f}s',flush=True)
    names=[f'force_{a}_{n}' for a in 'xyz' for n in NAMES]
    np.savez_compressed(ROOT/'features.npz',values=values,names=np.asarray(names),filenames=index.filename.to_numpy(dtype=str))
    index.to_csv(ROOT/'records.csv',index=False)
    pd.DataFrame(quality).to_csv(ROOT/'sensor_quality.csv',index=False)
    audit={'source_sha256':source_hash,'archive_sha256':hashlib.sha256((ROOT/'features.npz').read_bytes()).hexdigest(),'records':len(index),'features':len(names),'invalid':sum(not q['valid'] for q in quality),'seconds':time.time()-start,'label_used_in_extraction':False,'segment':'central 60 percent, not a verified contact-only mask','measurement_mask':'not supplied; all labels treated as published potentially interpolated labels'}
    (ROOT/'feature_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit),flush=True)

if __name__=='__main__':
    main()
