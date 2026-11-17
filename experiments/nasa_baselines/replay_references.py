"""Replay existing epoch100 CT/sensor monitors; no training and no epoch selection."""
from wearflow_paths import ROOT as _WF_ROOT
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parent
PROJECT=Path(str(_WF_ROOT / ''));sys.path.insert(0,str(PROJECT))
from wear_monitoring_dynamics.config import project_path,file_hash
from wear_monitoring_dynamics.models.registry import build_model
from wear_monitoring_dynamics.data.records import load_table
from wear_monitoring_dynamics.data.feature_archive import attach_features
from wear_monitoring_dynamics.data.sensor_quality import apply_quality
from wear_monitoring_dynamics.data.monitor_batch import make_monitor_batch

def main():
    torch.set_num_threads(1); started=time.perf_counter()
    m=pd.read_csv(PROJECT/"wear_monitoring_dynamics/outputs/current12_fusion_20261013/study_manifest.csv")
    m=m[(m.type=="monitor")&(m.role=="final")]
    result=[];forecasts=[];proof=[]
    for r in m.itertuples():
        spath=Path(r.run_dir)/"epochs/epoch_0100.pt"
        last=torch.load(Path(r.run_dir)/"last.pt",map_location="cpu",weights_only=True)
        cfg=last["config"];sc=last["scalers"];split=last["split"]
        table,signature=load_table(project_path(cfg["data"]["path"]),project_path(cfg["data"]["cache"]))
        table=apply_quality(attach_features(table,cfg["data"]["feature_archive"]),cfg["data"]["sensor_quality"])
        b=make_monitor_batch(cfg,table,sc,"cpu")
        with np.load(ROOT/"data"/r.fold/"final/arrays.npz") as a: data={k:a[k] for k in a.files}
        np.testing.assert_array_equal(table["case"],data["case"])
        np.testing.assert_array_equal(table["run"],data["run"])
        for method,cp in [("CT100",Path(cfg["monitor"]["fallback_checkpoint"]).parent/"epochs/epoch_0100.pt"),("sensor12_100",spath)]:
            state=torch.load(cp,map_location="cpu",weights_only=True)
            assert state["epoch"]==100
            parent=torch.load(cp.parent.parent/"last.pt",map_location="cpu",weights_only=True)
            assert parent["split"]==split
            own=parent["config"];ownsc=parent["scalers"]; ownb=make_monitor_batch(own,table,ownsc,"cpu")
            net=build_model(own);net.load_state_dict(state["model"]);net.eval()
            with torch.no_grad():
                pred=net.estimate(ownb,np.arange(len(table["case"]))).numpy().ravel()*ownsc["wear"][1]+ownsc["wear"][0]
            assert np.isfinite(pred).all()
            # Any sensor fallback is the historical frozen CT500; audit and disclose.
            for i in np.flatnonzero(data["test_mask"]):
                result.append(dict(method=method,fold=r.fold,seed=int(r.seed),case=int(data["case"][i]),run=int(data["run"][i]),target_mm=float(data["y_mm"][i]),pred_mm=float(pred[i]),sensor_valid=bool(data["sensor_valid"][i])))
            if method=="sensor12_100":
                np.savez_compressed(ROOT/"references"/f"{r.fold}_seed{r.seed}_monitor.npz",case=data["case"],run=data["run"],time=data["time"],pred_mm=pred)
                for i,j,h in data["test_pairs"]:
                    forecasts.append(dict(method="persistence_sensor100",fold=r.fold,seed=int(r.seed),case=int(data["case"][i]),origin_run=int(data["run"][i]),target_run=int(data["run"][j]),horizon=int(h),dt=float(data["time"][j]-data["time"][i]),target_mm=float(data["y_mm"][j]),pred_mm=float(pred[i])))
            proof.append(dict(method=method,fold=r.fold,seed=int(r.seed),checkpoint=str(cp),sha256=file_hash(cp),epoch=100,source_signature=signature,outer_invalid_labeled=int((data["test_mask"]&~data["sensor_valid"]).sum())))
    df=pd.DataFrame(result);future=pd.DataFrame(forecasts)
    df["ae"]=(df.pred_mm-df.target_mm).abs();future["ae"]=(future.pred_mm-future.target_mm).abs()
    for (method,seed),g in df.groupby(["method","seed"]):assert len(g)==138 and not g.duplicated(["case","run"]).any()
    df.to_csv(ROOT/"references/current_predictions.csv",index=False)
    future.to_csv(ROOT/"references/future_predictions.csv",index=False)
    current=df.groupby(["method","fold","seed","case"]).ae.mean().reset_index()
    summary=current.groupby("method").ae.mean().to_dict()
    (ROOT/"references/audit.json").write_text(json.dumps(dict(status="replayed",training=False,records=len(df),models=proof,elapsed_seconds=time.perf_counter()-started,current_case_macro_mae=summary),indent=2),encoding="utf-8")
    print(json.dumps(summary))
if __name__=="__main__":
    (ROOT/"references").mkdir(exist_ok=True)
    main()

