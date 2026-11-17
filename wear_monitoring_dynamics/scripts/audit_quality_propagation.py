"""Verify matched Phase 3 targets and append horizon details without retraining."""
import json
import numpy as np
import pandas as pd
from .audit_history_quality import ROOT, OLD, markdown
from ..config import write_json

def main():
    counts={}
    for task in ['forecast','recursive']:
        frames=[]
        for root in [OLD,ROOT]:
            p=pd.read_csv(root/f'phase3_{task}_predictions.csv.gz')
            selected=json.loads((root/'history_selection.json').read_text())
            p['method']=[('selected_history' if method==selected[fold] else method) for method,fold in zip(p.method,p.fold)]
            frames.append(p)
        keys=['fold','seed','case','method','sigma','replicate']
        keys+=['origin_run','target_run','horizon'] if task=='forecast' else ['start_kind','run','initial_run']
        joined=frames[0].merge(frames[1],on=keys,suffixes=('_old','_new'),validate='one_to_one')
        assert len(joined)==len(frames[0])==len(frames[1])
        np.testing.assert_allclose(joined.target_vb_mm_old,joined.target_vb_mm_new,equal_nan=True,atol=0,rtol=0)
        assert np.isfinite(joined.predicted_vb_mm_new).all()
        unchanged=joined.method.isin(['oracle','noisy','ct'])
        np.testing.assert_allclose(joined.loc[unchanged,'predicted_vb_mm_old'],joined.loc[unchanged,'predicted_vb_mm_new'],atol=1e-7,rtol=0)
        if task=='forecast':assert (joined.target_run>joined.origin_run).all()
        else:
            assert (joined.is_initial_old==joined.is_initial_new).all()
            assert (joined.time_old==joined.time_new).all()
            a=frames[1];b=a[(a.case==12)&(a.fold=='fold_a')&(a.start_kind=='first_record')&a.is_initial]
            ct=b[b.method=='ct'].set_index('seed').predicted_vb_mm
            for method in ['mlp_L1','selected_history']:
                np.testing.assert_allclose(b[b.method==method].set_index('seed').predicted_vb_mm.sort_index(),ct.sort_index(),atol=1e-6,rtol=0)
        counts[task]=len(joined)
    horizons=[]
    for root,label in [(OLD,'old'),(ROOT,'new')]:
        p=pd.read_csv(root/'phase3_forecast_horizons.csv');sel=json.loads((root/'history_selection.json').read_text())
        p['method']=[('selected_history' if method==sel[fold] else method) for method,fold in zip(p.method,p.fold)]
        horizons.append(p.rename(columns={'mae':'mae_'+label}))
    comparison=horizons[0].merge(horizons[1],on=['fold','method','sigma','horizon'],validate='one_to_one')
    comparison.to_csv(ROOT/'phase3_horizon_before_after.csv',index=False)
    p=ROOT/'RESULTS.md';body=p.read_text(encoding='utf-8').split('\n## 逐horizon补充审计')[0]
    body+='\n## 逐horizon补充审计\n\n'+markdown(comparison)+'\n\n'
    body+=f'逐点核查修复前后 {counts["forecast"]} 条未来预测、{counts["recursive"]} 条递推记录，起终点和真实标签一致；oracle/noisy/ct预测未改变。case12首记录的sensor模型初始化与同seed的ct后备一致。完整记录和逐case指标保留在本目录。\n'
    p.write_text(body,encoding='utf-8');write_json(ROOT/'propagation_audit.json',dict(passed=True,matched_rows=counts))
    print(counts)

if __name__=='__main__':main()
