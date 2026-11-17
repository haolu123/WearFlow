"""Recalibrate input quality rules and verify physical-unit RK4 conversion."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..config import project_path,write_json
from ..data.sensor_quality import raw_quality_metrics,fit_rules,assess
from ..training.recursive_adaptation import prepare,layout
from .audit_current12_recursive_joint import ROOT

def main():
    start=time.perf_counter();torch.set_num_threads(1)
    raw=np.load(project_path('mill_numpy/mill.npy'),allow_pickle=True).tolist();qm=raw_quality_metrics(raw)
    m=pd.read_csv(ROOT/'study_manifest.csv');seen=set();rows=[]
    for r in m[m.name=='single'].itertuples():
        task=json.loads((Path(r.run_dir)/'identity.json').read_text())['task']
        model,b,t,parent=prepare(task,'cpu');sp=parent['split'];key=(r.fold,tuple(sp['train']))
        if key in seen:continue
        seen.add(key);quality=parent['config']['data']['sensor_quality'];rules=fit_rules(qm,sp['train']);assert rules==quality['rules']
        decisions,_=assess(raw,qm,rules);invalid=decisions[~decisions.sensor_valid][['case','run']].astype(int).values.tolist();assert invalid==quality['invalid_records']
        st=layout(t,sp['outer_test_reserved'],'cpu');a,z=st['pairs'][:20,0],st['pairs'][:20,1]
        sc=parent['scalers'];mean,std=sc['wear'];scale=sc['time_scale'];c=b['condition'][a];dt=torch.tensor((t['time'][z]-t['time'][a])[:,None],dtype=torch.float32)
        with torch.no_grad():
            v=model.estimate(b,a)*std+mean;ref=model.dynamics.propagate((v-mean)/std,c,dt/scale)*std+mean
            y=v.clone();step=dt/model.dynamics.rk4_steps
            def rate(x):return std/scale*model.dynamics.rhs((x-mean)/std,c)
            for _ in range(model.dynamics.rk4_steps):
                k1=rate(y);k2=rate(y+step*k1/2);k3=rate(y+step*k2/2);k4=rate(y+step*k3);y=y+step*(k1+2*k2+2*k3+k4)/6
            torch.testing.assert_close(y,ref,rtol=2e-5,atol=3e-6)
        rows.append(dict(fold=r.fold,train_cases=sp['train'],invalid_records=invalid,physical_rk4_max_error_mm=float(abs(y-ref).max())))
    assert len(rows)==16
    write_json(ROOT/'input_quality_units_audit.json',dict(status='passed',splits=rows,seconds=time.perf_counter()-start));print('QUALITY AND UNITS PASSED',flush=True)
if __name__=='__main__':main()

