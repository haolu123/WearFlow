"""Tool-disjoint preprocessing and strictly causal published-label targets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parent
HORIZONS=(240,480,720)

def records():
    raw=pd.read_csv(ROOT/'records.csv')
    archive=np.load(ROOT/'features.npz')
    assert np.array_equal(raw.filename.to_numpy(str),archive['filenames'])
    raw['feature_row']=np.arange(len(raw))
    raw['previous_feature_row']=raw.groupby('tool').feature_row.shift(1)
    raw['previous_C']=raw.groupby('tool').cumulated_tool_contact_time.shift(1)
    raw=raw.dropna(subset=['previous_feature_row']).reset_index(drop=True)
    raw['previous_feature_row']=raw.previous_feature_row.astype(int)
    raw['VB_mm']=raw.wear/1000
    assert (raw.previous_C<raw.cumulated_tool_contact_time).all()
    x=archive['values'][raw.previous_feature_row.to_numpy()]
    return raw,x,archive['names']

def pairs(frame):
    rows=[]
    for tool,g in frame.groupby('tool'):
        t=g.cumulated_tool_contact_time.to_numpy()
        ix=g.index.to_numpy()
        for h in HORIZONS:
            js=np.searchsorted(t,t+h)
            for i,j in enumerate(js):
                if j<len(t) and t[j]-t[i]-h<=30:
                    rows.append((ix[i],ix[j],h,t[j]-t[i],tool))
    return pd.DataFrame(rows,columns=['origin','target','requested_seconds','actual_seconds','tool'])

def fit(frame,x,names,train_tools):
    valid=np.isfinite(x).all(axis=1)
    mask=frame.tool.isin(train_tools).to_numpy()&valid
    scores=[]
    for k in range(x.shape[1]):
        tool_scores=[]
        for tool in train_tools:
            m=mask&(frame.tool.to_numpy()==tool)
            if np.ptp(x[m,k])==0: tool_scores.append(0.)
            else: tool_scores.append(abs(spearmanr(x[m,k],frame.VB_mm.to_numpy()[m]).statistic))
        scores.append(float(np.mean(tool_scores)))
    selected=sorted(range(len(names)),key=lambda k:(-scores[k],names[k]))[:12]
    low,high=np.percentile(x[mask][:,selected],[1,99],axis=0)
    clipped=np.clip(x[mask][:,selected],low,high)
    mean=clipped.mean(axis=0); std=clipped.std(axis=0)
    std=np.where(std<1e-8,1.,std)
    train_y=frame.loc[frame.tool.isin(train_tools),'VB_mm'].to_numpy()
    return {'train_tools':list(train_tools),'selected':selected,'names':names[selected].tolist(),'all_scores':dict(zip(names.tolist(),scores)),'low':low.tolist(),'high':high.tolist(),'mean':mean.tolist(),'std':std.tolist(),'wear_mean':float(train_y.mean()),'wear_std':float(train_y.std()),'valid_train_records':int(mask.sum())}

def transform(frame,x,scaler):
    valid=np.isfinite(x).all(axis=1)
    z=(np.clip(x[:,scaler['selected']],scaler['low'],scaler['high'])-scaler['mean'])/scaler['std']
    z[~valid]=0
    return np.column_stack([z,frame.cumulated_tool_contact_time.to_numpy()/1000]).astype('float32'),valid

def freeze_splits():
    frame,x,names=records()
    plan=[]
    folder=ROOT/'splits'; folder.mkdir(exist_ok=True)
    for test in range(1,10):
        group=3*((test-1)//3)
        val=group+(test-group-2)%3+1
        for stage in ['inner','final']:
            train=[t for t in range(1,10) if t!=test and (stage=='final' or t!=val)]
            scaler=fit(frame,x,names,train)
            record={'split':f'test{test}_{stage}','test_tool':test,'validation_tool':val if stage=='inner' else None,**scaler}
            (folder/f'test{test}_{stage}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            plan.append({'split':record['split'],'test_tool':test,'validation_tool':record['validation_tool'],'train_tools':','.join(map(str,train))})
    pd.DataFrame(plan).to_csv(ROOT/'split_manifest.csv',index=False)
    frame.to_csv(ROOT/'causal_targets.csv',index=False)
    pp=pairs(frame);pp.to_csv(ROOT/'forecast_pairs.csv',index=False)
    pp.groupby(['tool','requested_seconds']).agg(count=('origin','size'),min_dt=('actual_seconds','min'),max_dt=('actual_seconds','max')).to_csv(ROOT/'forecast_coverage.csv')
    audit={'target_count':len(frame),'first_records_excluded':9,'pair_count':len(pp),'horizons_seconds':HORIZONS,'all_previous_C_strictly_lower':True,'splits':len(plan),'test_performance_accessed':False,'invalid_previous_sensor_count':int((~np.isfinite(x).all(axis=1)).sum())}
    (ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit))

if __name__=='__main__':
    freeze_splits()
