"""Export immutable, split-specific NASA inputs from the current thread's protocol.
Run with CondWear-ODE Python. No changes to historical project artifacts.
"""
from wearflow_paths import ROOT as _WF_ROOT
import json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT=Path(str(_WF_ROOT / ''))
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT))
from wear_monitoring_dynamics.config import project_path,file_hash
from wear_monitoring_dynamics.data.records import load_table
from wear_monitoring_dynamics.data.feature_archive import attach_features
from wear_monitoring_dynamics.data.sensor_quality import apply_quality
from wear_monitoring_dynamics.data.preprocessing import fit_scalers,transform,forecasting_pairs
from wear_monitoring_dynamics.scripts.run_psd_sweep import rank

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")

def equal(a,b):
    if isinstance(a,dict):
        assert set(a)==set(b)
        for k in a: equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,(int,float)):
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
    else: assert a==b

def main():
    source=PROJECT/"wear_monitoring_dynamics/outputs/current12_fusion_20261013"
    mf=pd.read_csv(source/"study_manifest.csv")
    mf=mf[mf.type=="monitor"]; assert len(mf)==24
    snapshots=[]; checked={}; outer_keys=set()
    for row in mf.itertuples():
        parent=Path(row.run_dir)/"last.pt"
        s=torch.load(parent,map_location="cpu",weights_only=True)
        cfg=s["config"]; split=s["split"]; sc=s["scalers"]
        key=row.fold+("/inner_"+str(int(row.holdout)) if row.role=="inner" else "/final")
        if key in checked:
            assert checked[key]==json.dumps({"split":split,"scalers":sc},sort_keys=True)
            continue
        checked[key]=json.dumps({"split":split,"scalers":sc},sort_keys=True)
        table,signature=load_table(project_path(cfg["data"]["path"]),project_path(cfg["data"]["cache"]))
        assert signature==s["data_signature"]
        table=apply_quality(attach_features(table,cfg["data"]["feature_archive"]),cfg["data"]["sensor_quality"])
        assert cfg["data"]["sensor_quality"]["rules"]["train_cases"]==split["train"]
        fresh=fit_scalers(table,split["train"],cfg["data"]["sensor_preprocessing"])
        equal(fresh,sc)
        names=cfg["data"]["feature_archive"]["names"]
        candidates=[i for i,n in enumerate(names) if n.startswith(("smcAC_","smcDC_"))]
        assert len(candidates)==50
        scores,_=rank(table,split["train"])
        selected=sorted(candidates,key=lambda i:(-scores[i],names[i]))[:12]
        assert selected==cfg["monitor"]["selected_indices"]
        z=transform(table,sc)
        x=np.column_stack([z["sensor"][:,selected],z["condition"],z["time"]])
        assert x.shape[1]==17 and np.isfinite(x).all()
        labeled=np.isfinite(table["VB"])
        arrays=dict(X=x,X_ct=np.column_stack([z["condition"],z["time"]]),
            sensor_raw=table["sensor"][:,selected],sensor_z=z["sensor"][:,selected],
            case=table["case"].astype(int),run=table["run"].astype(int),time=table["time"],
            condition_raw=np.column_stack([table[k] for k in ["DOC","feed","material"]]),
            y_mm=table["VB"],y_z=z["target"].ravel(),sensor_valid=table["sensor_valid"])
        for label in ["train","validation","test"]:
            arrays[label+"_mask"]=np.isin(table["case"],split[label])&labeled
            arrays[label+"_fit_mask"]=arrays[label+"_mask"]&table["sensor_valid"]
            arrays[label+"_pairs"]=forecasting_pairs(table,split[label],[1,2,3])
        d=ROOT/"data"/key;d.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(d/"arrays.npz",**arrays)
        info=dict(fold=row.fold,role=row.role,split=split,scalers=sc,
            selected_features=[names[i] for i in selected],selected_indices=selected,
            parent=str(parent),parent_sha256=file_hash(parent),
            feature_archive=cfg["data"]["feature_archive"],data_signature=signature,
            quality=cfg["data"]["sensor_quality"],
            audit=dict(training_only_ranking_recomputed=True,scalers_recomputed=True,
                       quality_training_cases_checked=True,quality_thresholds_refit=False),
            counts={k:int(arrays[k+"_mask"].sum()) for k in ["train","validation","test"]})
        write(d/"metadata.json",info)
        if row.role=="final":
            outer_keys.update(zip(arrays["case"][arrays["test_mask"]],arrays["run"][arrays["test_mask"]]))
        snapshots.append(dict(key=key,directory=str(d),arrays_sha256=file_hash(d/"arrays.npz"),**info["counts"]))
    assert len(snapshots)==16 and len(outer_keys)==138
    write(ROOT/"data_manifest.json",dict(parent_study=str(source),splits=snapshots,
        unique_outer_labels=len(outer_keys),candidate_count=50,selected_count=12,
        status="input_export_verified",not_full_model_audit=True))
    print(json.dumps({"splits":len(snapshots),"outer_labels":len(outer_keys),"ranking_and_scalers":"verified"}))
if __name__=="__main__":main()

