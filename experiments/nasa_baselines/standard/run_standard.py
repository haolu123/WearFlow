"""Standard GPR/SVR baselines; no claim to reproduce physics-informed GPR."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key]="1"
import argparse, hashlib, itertools, json, pickle, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern, WhiteKernel
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

BASE=Path(__file__).resolve().parent
SUITE=BASE.parent
def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def configs():
    out=[dict(method="gpr",id="gpr_"+k,kernel=k) for k in ("rbf","matern15","matern25")]
    for c,e,g in itertools.product((.1,1.,10.,100.),(.01,.1),("scale",.01)):
        out.append(dict(method="svr",id=f"svr_c{c}_eps{e}_g{g}",C=c,epsilon=e,gamma=g))
    return out
def build(cfg):
    if cfg["method"]=="svr":
        return SVR(kernel="rbf",C=cfg["C"],epsilon=cfg["epsilon"],gamma=cfg["gamma"])
    k=RBF(1.,(1e-2,1e3)) if cfg["kernel"]=="rbf" else Matern(1.,(1e-2,1e3),nu=1.5 if cfg["kernel"]=="matern15" else 2.5)
    kernel=ConstantKernel(1.,(1e-3,1e3))*k+WhiteKernel(.01,(1e-6,1.))
    return GaussianProcessRegressor(kernel=kernel,alpha=1e-10,normalize_y=False,n_restarts_optimizer=0,random_state=0)
def load_split(entry):
    path=Path(entry["directory"])
    assert digest(path/"arrays.npz")==entry["arrays_sha256"],entry["key"]
    with np.load(path/"arrays.npz",allow_pickle=False) as z:
        d={k:z[k].copy() for k in z.files}
    m=json.loads((path/"metadata.json").read_text(encoding="utf-8"))
    assert d["X"].shape[1]==17
    fit=d["train_fit_mask"]
    assert np.array_equal(fit,d["train_mask"] & d["sensor_valid"])
    assert np.isfinite(d["X"][fit]).all() and np.isfinite(d["y_z"][fit]).all()
    mean,std=m["scalers"]["wear"]
    assert std>0
    labelled=np.isfinite(d["y_z"])
    np.testing.assert_allclose(d["y_z"][labelled]*std+mean,d["y_mm"][labelled],atol=1e-12)
    for part in ("validation","test"):
        assert not np.any(fit & d[part+"_mask"])
        assert set(d["case"][fit]).isdisjoint(set(d["case"][d[part+"_mask"]]))
    return d,m
def fit(cfg,d):
    model=build(cfg);begin=time.perf_counter()
    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter("always")
        model.fit(d["X"][d["train_fit_mask"]],d["y_z"][d["train_fit_mask"]])
    return model,time.perf_counter()-begin,[str(w.message) for w in warn]
def predict_mm(model,d,m,mask):
    mean,std=m["scalers"]["wear"]
    return model.predict(d["X"][mask])*std+mean
def records(d,mask,pred,fold,cfg,split):
    return pd.DataFrame(dict(method=cfg["method"],config_id=cfg["id"],fold=fold,split=split,
        case=d["case"][mask],run=d["run"][mask],time=d["time"][mask],
        target_mm=d["y_mm"][mask],pred_mm=pred,sensor_valid=d["sensor_valid"][mask],scope="valid_only"))
def macro(frame):
    return float(frame.assign(ae=abs(frame.pred_mm-frame.target_mm)).groupby("case").ae.mean().mean())
def synthetic_check():
    x=np.linspace(-1,1,20)[:,None]
    d=dict(X=np.repeat(x,17,axis=1),y_z=.25*x[:,0]+.1,train_fit_mask=np.arange(20)<16)
    for cfg in (configs()[0],configs()[3]):
        mdl,_,_=fit(cfg,d)
        assert np.isfinite(mdl.predict(d["X"])).all()
        np.testing.assert_allclose(mdl.predict(d["X"]),pickle.loads(pickle.dumps(mdl)).predict(d["X"]),rtol=0,atol=0)
    return {"finite_predictions":True,"pickle_exact":True}
def run(smoke):
    out=BASE/("smoke" if smoke else "results")
    out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    dump(out/"status.json",{"status":"validating","smoke":smoke})
    syn=synthetic_check()
    manifest=json.loads((SUITE/"data_manifest.json").read_text(encoding="utf-8"))
    entries=manifest["splits"];cfgs=configs()
    if smoke:
        entries=[next(e for e in entries if "/inner_" in e["key"])]
        cfgs=[cfgs[0],cfgs[3]]
    dump(out/"protocol.json",{"scope":"valid_only","data_manifest_sha256":digest(SUITE/"data_manifest.json"),
        "code_sha256":digest(Path(__file__)),"sklearn":sklearn.__version__,"configs":cfgs,
        "selection":"mean of inner held-out-case MAE in mm; deterministic ties resolved by config id",
        "fit_target":"y_z; GPR normalize_y=False","seed_repeats":1,
        "gpr":"training LML only; zero restarts; isotropic length scale; jitter alpha=1e-10",
        "threads":1})
    inner=[];inner_predictions=[];warning_rows=[];split_audits=[]
    for entry in entries:
        if "/inner_" not in entry["key"]:continue
        d,m=load_split(entry);mask=d["validation_mask"] & d["sensor_valid"];fold=entry["key"].split("/")[0]
        assert mask.any()
        split_audits.append(dict(split=entry["key"],fit=int(d["train_fit_mask"].sum()),
            validation_valid=int(mask.sum()),validation_total=int(d["validation_mask"].sum()),
            validation_cases=np.unique(d["case"][mask]).tolist(),arrays_sha256=entry["arrays_sha256"]))
        for cfg in cfgs:
            model,secs,warns=fit(cfg,d)
            pred=predict_mm(model,d,m,mask)
            frame=records(d,mask,pred,fold,cfg,entry["key"])
            inner_predictions.append(frame)
            inner.append(dict(fold=fold,split=entry["key"],method=cfg["method"],config_id=cfg["id"],
                mae_mm=macro(frame),n=int(mask.sum()),n_cases=int(frame.case.nunique()),fit_seconds=secs,
                learned_kernel=str(model.kernel_) if cfg["method"]=="gpr" else "rbf"))
            warning_rows.extend(dict(split=entry["key"],config_id=cfg["id"],message=w) for w in warns)
        dump(out/"status.json",{"status":"inner_fitting","completed_fits":len(inner),"smoke":smoke})
    inner=pd.DataFrame(inner);inner.to_csv(out/"inner_scores.csv",index=False)
    pd.concat(inner_predictions,ignore_index=True).to_csv(out/"inner_predictions.csv.gz",index=False)
    scores=inner.groupby(["fold","method","config_id"],as_index=False).agg(inner_case_mae_mm=("mae_mm","mean"),inner_splits=("split","nunique"))
    scores.to_csv(out/"candidate_summary.csv",index=False)
    if smoke:
        dump(out/"audit.json",dict(synthetic=syn,split_audits=split_audits))
        dump(out/"warnings.json",warning_rows)
        dump(out/"status.json",dict(status="smoke_passed",fits=len(inner),elapsed_seconds=time.perf_counter()-start))
        return
    assert len(inner)==266 and len(split_audits)==14
    selected={};outer=[];all_records=[];reload_audits=[];final_times=[]
    for fold in ("fold_a","fold_b"):
        selected[fold]={}
        for method in ("gpr","svr"):
            cand=scores[(scores.fold==fold)&(scores.method==method)].sort_values(["inner_case_mae_mm","config_id"])
            assert (cand.inner_splits==7).all()
            winner=cand.iloc[0];cfg=next(c for c in cfgs if c["id"]==winner.config_id)
            selected[fold][method]={"config":cfg,"inner_case_mae_mm":float(winner.inner_case_mae_mm)}
    dump(out/"selected.json",selected) # Persist validation-only choice before any final test evaluation.
    for fold in ("fold_a","fold_b"):
        entry=next(e for e in entries if e["key"]==fold+"/final")
        d,m=load_split(entry);test=d["test_mask"]&d["sensor_valid"]
        for method in ("gpr","svr"):
            cfg=selected[fold][method]["config"];model,secs,warns=fit(cfg,d)
            md=out/"models"/fold/method;md.mkdir(parents=True,exist_ok=True)
            (md/"model.pkl").write_bytes(pickle.dumps(model,protocol=pickle.HIGHEST_PROTOCOL))
            dump(md/"metadata.json",dict(config=cfg,wear_scaler=m["scalers"]["wear"],data_key=entry["key"],
                arrays_sha256=entry["arrays_sha256"],feature_count=17,
                learned_kernel=str(model.kernel_) if method=="gpr" else "rbf",sklearn=sklearn.__version__))
            loaded=pickle.loads((md/"model.pkl").read_bytes())
            valid=d["sensor_valid"]
            pred=predict_mm(model,d,m,valid);pred2=predict_mm(loaded,d,m,valid)
            np.testing.assert_allclose(pred,pred2,rtol=0,atol=0)
            reload_audits.append(dict(fold=fold,method=method,max_abs_reload_error=float(np.max(abs(pred-pred2))),
                n_test_valid=int(test.sum()),n_test_all=int(d["test_mask"].sum()),
                excluded_test_keys=[dict(case=int(d["case"][i]),run=int(d["run"][i])) for i in np.flatnonzero(d["test_mask"] & ~d["sensor_valid"])]))
            allf=records(d,valid,pred,fold,cfg,entry["key"])
            allf["is_test"]=d["test_mask"][valid];all_records.append(allf)
            outer.append(records(d,test,predict_mm(model,d,m,test),fold,cfg,entry["key"]))
            final_times.append(dict(fold=fold,method=method,fit_seconds=secs))
            warning_rows.extend(dict(split=entry["key"],config_id=cfg["id"],message=w) for w in warns)
    outer=pd.concat(outer,ignore_index=True)
    assert not outer.duplicated(["method","fold","case","run"]).any()
    outer.to_csv(out/"outer_predictions_valid_only.csv",index=False)
    pd.concat(all_records,ignore_index=True).to_csv(out/"all_record_predictions_valid_only.csv",index=False)
    outer["ae"]=abs(outer.pred_mm-outer.target_mm);outer["se"]=(outer.pred_mm-outer.target_mm)**2
    cm=outer.groupby(["method","fold","case"],as_index=False).agg(mae_mm=("ae","mean"),mse_mm2=("se","mean"),n=("ae","size"))
    cm.to_csv(out/"case_metrics_valid_only.csv",index=False)
    summary=[]
    for method in ("gpr","svr"):
        for subset in ("overall","fold_a","fold_b","fold_a_in","fold_a_out"):
            f=outer[outer.method==method]
            if subset!="overall":f=f[f.fold==("fold_b" if subset=="fold_b" else "fold_a")]
            if subset=="fold_a_in":f=f[f.target_mm<=.74]
            if subset=="fold_a_out":f=f[f.target_mm>.74]
            summary.append(dict(method=method,subset=subset,scope="valid_only",n=len(f),cases=int(f.case.nunique()),
                case_mae_mm=macro(f),case_rmse_mm=float(np.sqrt(f.groupby("case").se.mean()).mean()),
                point_mae_mm=float(f.ae.mean())))
    pd.DataFrame(summary).to_csv(out/"summary_valid_only.csv",index=False)
    dump(out/"warnings.json",warning_rows)
    dump(out/"audit.json",dict(synthetic=syn,split_audits=split_audits,reload=reload_audits,
        selection_uses_outer_test=False,deterministic=True,final_fit_times=final_times,
        expected_inner_fits=266,actual_inner_fits=len(inner),final_fits=4,
        valid_only=True,no_fallback_applied=True))
    dump(out/"status.json",dict(status="completed_valid_only",inner_fits=len(inner),final_fits=4,
        elapsed_seconds=time.perf_counter()-start,total_fit_seconds=float(inner.fit_seconds.sum()+sum(x["fit_seconds"] for x in final_times)),
        outer_rows=len(outer)))
    print(pd.DataFrame(summary).to_string(index=False))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--smoke",action="store_true");args=ap.parse_args()
    with threadpool_limits(limits=1):
        try:run(args.smoke)
        except Exception as e:
            dump(BASE/("smoke" if args.smoke else "results")/"status.json",dict(status="failed",error=repr(e)))
            raise

