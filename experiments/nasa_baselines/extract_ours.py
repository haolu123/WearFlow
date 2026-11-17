"""Extract matched single-shot reference predictions; never mix fused128 with raw138."""
from wearflow_paths import ROOT as _WF_ROOT
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent
PARENT=Path(str(_WF_ROOT / 'wear_monitoring_dynamics/outputs/current12_abcd_from100_20261019'))
def main():
    tasks=json.loads((PARENT/"planned_tasks.json").read_text())
    rows=[]
    names={"A":"independent_monitor100_ode500","B":"current_only_extra100","C":"frozen_ode_future_extra100","D":"joint_future_extra100"}
    for task in tasks:
        p=PARENT/"runs"/task["id"]/"epoch_predictions.csv.gz"
        f=pd.read_csv(p)
        epochs=[100]+([0] if task["group"]=="B" else [])
        for epoch in epochs:
            group=task["group"] if epoch else "A"
            q=f[(f.epoch==epoch)&(f.split=="test_diagnostic")].copy()
            assert len(q)>0
            q["method"]=names[group];q["fold"]=task["fold"];q["seed"]=task["seed"];q["budget"]=("monitor100_ode500" if group=="A" else "monitor100_ode500_plus100")
            rows.append(q)
    f=pd.concat(rows,ignore_index=True).rename(columns={"target_vb_mm":"target_mm","predicted_vb_mm":"pred_mm"})
    current=f[f.task=="monitoring"].copy();current["run"]=current.target_run
    future=f[f.task=="forecasting"].copy()
    for (name,seed),g in current.groupby(["method","seed"]):
        assert len(g)==138 and not g.duplicated(["case","run"]).any()
    for (name,seed),g in future.groupby(["method","seed"]):
        assert not g.duplicated(["case","origin_run","target_run","horizon"]).any()
    reference=pd.read_csv(ROOT/"references/future_predictions.csv")
    k=["fold","seed","case","origin_run","target_run","horizon"]
    for name,g in future.groupby("method"):
        m=g.merge(reference[k+["dt","target_mm"]],on=k,validate="one_to_one",suffixes=("","_reference"))
        assert len(m)==len(g)==len(reference)
        np.testing.assert_allclose(m.target_mm,m.target_mm_reference,rtol=0,atol=1e-12)
    current.to_csv(ROOT/"references/ours_current_predictions.csv",index=False)
    future.to_csv(ROOT/"references/ours_future_predictions.csv",index=False)
    (ROOT/"references/ours_reuse_audit.json").write_text(json.dumps(dict(status="matched",current_rows=len(current),future_rows=len(future),target_keys_equal=True,source="single-shot all labeled origins; NOT fused128 protocol"),indent=2))
    print({"current":len(current),"future":len(future)})
if __name__=="__main__":main()

