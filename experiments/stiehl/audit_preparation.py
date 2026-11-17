"""Independent checks of split isolation, alignment, units and evaluation timing."""
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
from data_protocol import ROOT, records, fit, transform, pairs
from models import WearModel

def main():
    torch.set_num_threads(1)
    frame,x,names=records(); all_pairs=pairs(frame)
    saved=pd.read_csv(ROOT/'forecast_pairs.csv')
    pd.testing.assert_frame_equal(saved,all_pairs)
    assert (frame.groupby('tool').run.min()>1).all()
    assert np.all(frame.tool.to_numpy()[saved.origin]==frame.tool.to_numpy()[saved.target])
    assert np.all(frame.cumulated_tool_contact_time.to_numpy()[saved.target]-frame.cumulated_tool_contact_time.to_numpy()[saved.origin]==saved.actual_seconds)
    assert (saved.actual_seconds>=saved.requested_seconds).all()
    assert (saved.actual_seconds<=saved.requested_seconds+30).all()
    rows=[]
    for path in sorted((ROOT/'splits').glob('*.json')):
        s=json.loads(path.read_text())
        excluded=set(range(1,10))-set(s['train_tools'])
        assert s['test_tool'] in excluded
        if s['validation_tool'] is not None:assert s['validation_tool'] in excluded
        perturbed=frame.copy()
        perturbed.loc[perturbed.tool.isin(excluded),'VB_mm']=1000
        repeat=fit(perturbed,x,names,s['train_tools'])
        for key in repeat:assert repeat[key]==s[key],key
        rows.append({'split':s['split'],'excluded_label_perturbation_invariant':True,'train_n':s['valid_train_records']})
    s=json.loads((ROOT/'splits/test9_inner.json').read_text())
    transformed,_=transform(frame,x,s)
    dev=frame.tool!=9
    devframe=frame[dev].reset_index(drop=True)
    xx=torch.tensor(transformed[dev]);y=torch.tensor((devframe.VB_mm.to_numpy()-s['wear_mean'])/s['wear_std'],dtype=torch.float32)
    pp=pairs(devframe)
    dt=torch.tensor(pp.actual_seconds.to_numpy()/1000,dtype=torch.float32)
    model=WearModel();model.load_state_dict(torch.load(ROOT/'smoke/integral.pt',weights_only=False)['model'])
    timing=[]
    for _ in range(3):
        t=time.perf_counter()
        with torch.no_grad():
            current=model(xx)
            future=model.propagate(current[pp.origin.to_numpy()],dt)
            oracle=model.propagate(y[pp.origin.to_numpy()],dt)
        assert torch.isfinite(future).all() and torch.isfinite(oracle).all()
        timing.append(time.perf_counter()-t)
    probe=xx[:50].clone().requires_grad_(True)
    derivative=torch.autograd.grad(model(probe).sum(),probe)[0][:,-1]
    plus=probe.detach().clone();minus=plus.clone();plus[:,-1]+=.001;minus[:,-1]-=.001
    finite_difference=(model(plus)-model(minus))/.002
    assert torch.max(torch.abs(finite_difference-derivative))<1e-4
    # z -> mm conversion and tau -> seconds conversion are both necessary.
    z=y[:50]; dt_seconds=torch.ones(50)
    with torch.no_grad():
        mm_increment=(model.propagate(z,dt_seconds/1000)-z)*s['wear_std']
        rate_mm_s=model.rhs(z)*s['wear_std']/1000
    assert torch.max(torch.abs(mm_increment-rate_mm_s))<1e-6
    report={'passed':True,'split_checks':rows,'same_tool_forward_pairs':True,'previous_run_only':True,'time_origin_preserved':True,'excluded_label_perturbation_invariant':True,'derivative_finite_difference_max':float((finite_difference-derivative).abs().max().detach()),'mm_seconds_conversion_passed':True,'eval_seconds_development_all_current_future_oracle':timing,'outer_test_performance_computed':False}
    (ROOT/'preparation_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    # Freeze identities now, before formal test predictions exist.
    manifest=[]
    for row in rows:
        for seed in ([0] if row['split'].endswith('inner') else range(5)):
            for method in ['time','current','local','integral']:
                manifest.append({'split':row['split'],'seed':seed,'method':method,'epochs':100,'status':'planned'})
    pd.DataFrame(manifest).to_csv(ROOT/'study_manifest.csv',index=False)
    pd.DataFrame([{'split':r['split'],'method':'gpr','random_state':0,'max_train_records_per_tool':128,'status':'planned'} for r in rows]).to_csv(ROOT/'gpr_manifest.csv',index=False)
    protocol={'evaluation':'published potentially interpolated wear labels','alignment':'previous completed run sensor -> next start VB','horizons_seconds':[240,480,720],'max_matching_lag_seconds':30,'time_scale_seconds':1000,'wear_units':'mm','rank_top_k':12,'monitor_hidden':[64,32],'rate_hidden':[32,32],'condition_inputs':[],'loss_reduction':'tool equal','auxiliary_weight':1,'epochs':100,'learning_rate':.001,'weight_decay':.0001,'rk4_steps':20,'seeds':[0,1,2,3,4],'training':'cold start, no rate pretraining','outer_split':'leave one tool out','inner_split':'previous tool in same machine cyclic; one fixed diagnostic split','checkpoint_selection':'last epoch','statistics':'average seeds within each tool before paired tool bootstrap'}
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'prepare_features.py',ROOT/'data_protocol.py',ROOT/'models.py',ROOT/'README.md']}
    (ROOT/'protocol_identity.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
