"""Full source-range PI-GPR grid, train-only preprocessing and inner selection.

Spectral evaluation is algebraically equivalent to exact Cholesky GP:
K=sf^2 K0, A=sf^2 K0+(sn^2+jitter)I. Twenty direct fits validate it.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import csv, hashlib,json,pickle,time
from pathlib import Path
from dataclasses import replace
import numpy as np
from threadpoolctl import threadpool_limits
from sun_pigpr import SunPIGPR,PIGPRConfig,paper_grid,matern32

BASE=Path(__file__).resolve().parent
ROOT=BASE.parent
OUT=BASE/"formal"
GRID=list(paper_grid())
BASECFG=PIGPRConfig()

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(path)
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)
def read_split(fold,split,role):
    directory=ROOT/"data"/fold/split
    z=np.load(directory/"arrays.npz")
    meta=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    tr=np.isin(z["case"],meta["split"]["train"]) & z["sensor_valid"]
    ev=np.isin(z["case"],meta["split"][role]) & z["sensor_valid"]
    return z,meta,tr,ev

def fitted_arrays(fold,split,role):
    z,meta,tr,ev=read_split(fold,split,role)
    m=SunPIGPR(BASECFG).fit(z["sensor_z"][tr],z["y_mm"][tr],z["time"][tr],z["case"][tr],z["X_ct"][tr,:4])
    X=z["sensor_z"][ev];tt=z["time"][ev];cc=z["case"][ev];cond=z["X_ct"][ev,:4]
    hi=m._hi(X,tt,cc)
    Z=(m._design(X,tt,cond,hi)-m.x_location_)/m.x_scale_
    physical=m.mean_coef_[0]*hi+m.mean_coef_[1]
    hi_tr=m._hi(z["sensor_z"][tr],z["time"][tr],z["case"][tr])
    labels=np.isfinite(z["y_mm"][tr])
    r=(z["y_mm"][tr][labels]-(m.mean_coef_[0]*hi_tr[labels]+m.mean_coef_[1]))/m.y_scale_
    return z,meta,tr,ev,m,Z,physical,r

def spectral_grid(m,Z,physical,residual,subset=None):
    grid=GRID if subset is None else subset
    pred=np.empty((len(grid),len(Z)),float)
    cache={}
    for i,c in enumerate(grid):
        ell=c["length_scale"]
        if ell not in cache:
            K0=matern32(m.X_train_,m.X_train_,ell,1.)
            vals,U=np.linalg.eigh(K0)
            if vals.min() < -1e-8:raise AssertionError("Kernel not PSD")
            cache[ell]=(vals,U.T@residual,matern32(Z,m.X_train_,ell,1.)@U)
        vals,Ur,KU=cache[ell]
        sf2=c["signal_std"]**2
        weights=sf2*Ur/(sf2*vals+c["noise_std"]**2+BASECFG.jitter)
        pred[i]=physical+m.y_scale_*(KU@weights)
    return pred

def benchmark():
    z,meta,tr,ev,m,Z,physical,r=fitted_arrays("fold_a","inner_1","validation")
    ids=np.linspace(0,len(GRID)-1,20).astype(int)
    subset=[GRID[i] for i in ids]
    t=time.perf_counter();fast=spectral_grid(m,Z,physical,r,subset);spectral_sec=time.perf_counter()-t
    direct=[]
    t=time.perf_counter()
    for c in subset:
        model=SunPIGPR(replace(BASECFG,**c)).fit(z["sensor_z"][tr],z["y_mm"][tr],z["time"][tr],z["case"][tr],z["X_ct"][tr,:4])
        direct.append(model.predict(z["sensor_z"][ev],z["time"][ev],z["case"][ev],z["X_ct"][ev,:4]))
    direct_sec=time.perf_counter()-t
    error=float(np.max(np.abs(fast-np.array(direct))))
    np.testing.assert_allclose(fast,direct,atol=1e-10,rtol=1e-10)
    report={"candidates":20,"direct_seconds":direct_sec,"spectral_seconds":spectral_sec,
            "max_abs_difference_mm":error,"estimated_direct_47250_seconds":direct_sec/20*47250,
            "outer_test_evaluated":False}
    write_json(OUT/"benchmark.json",report);print("BENCHMARK",json.dumps(report),flush=True)

def main():
    OUT.mkdir(exist_ok=True)
    identity={"model_sha":sha(BASE/"sun_pigpr.py"),"runner_sha":sha(Path(__file__)),
              "data_manifest_sha":sha(ROOT/"data_manifest.json"),"n_grid":len(GRID),
              "protocol":"causal-w7-sigma0.5; normalized GP target/input; inner case mean; source numeric hyperparameter ranges"}
    ip=OUT/"identity.json"
    if ip.exists() and json.loads(ip.read_text())!=identity:
        raise RuntimeError("Identity changed: do not overwrite/reuse stale runs")
    write_json(ip,identity)
    write_csv(OUT/"candidate_grid.csv",[{"candidate_id":i,**c} for i,c in enumerate(GRID)])
    benchmark()
    started=time.perf_counter()
    selected={}
    splits_done=0
    # All inner selection is completed before reading any final/test arrays.
    for fold in ["fold_a","fold_b"]:
        folders=sorted((ROOT/"data"/fold).glob("inner_*"),key=lambda p:int(p.name.split("_")[-1]))
        scores=[]
        for directory in folders:
            name=directory.name;dest=OUT/fold/name;dest.mkdir(parents=True,exist_ok=True)
            completed=dest/"completed.json"
            if completed.exists():
                data=np.load(dest/"validation_predictions.npz");scores.append(data["mae"])
            else:
                start=time.perf_counter()
                z,meta,tr,ev,m,Z,physical,r=fitted_arrays(fold,name,"validation")
                pred=spectral_grid(m,Z,physical,r)
                target=z["y_mm"][ev];label=np.isfinite(target)
                assert len(np.unique(z["case"][ev]))==1
                mae=np.abs(pred[:,label]-target[label]).mean(axis=1)
                np.savez_compressed(dest/"validation_predictions.npz",candidate_id=np.arange(len(GRID)),
                    pred_mm=pred,target_mm=target,case=z["case"][ev],run=z["run"][ev],
                    time=z["time"][ev],label_mask=label,mae=mae)
                write_csv(dest/"scores.csv",[{"candidate_id":i,"validation_case":meta["split"]["validation"][0],
                    "mae_mm":float(v),"n_labels":int(label.sum())} for i,v in enumerate(mae)])
                write_json(dest/"train_state.json",m.audit_state())
                write_json(completed,{"seconds":time.perf_counter()-start,"train_cases":meta["split"]["train"],
                    "validation_case":meta["split"]["validation"],"n_candidates":len(GRID),"source_hash":sha(directory/"arrays.npz")})
                scores.append(mae)
            splits_done+=1
            write_json(OUT/"status.json",{"stage":"inner_grid","completed_inner":splits_done,"planned_inner":14})
            print(f"INNER {fold}/{name} complete",flush=True)
        score=np.mean(np.stack(scores),axis=0)
        best=int(np.argmin(score))
        selected[fold]={"candidate_id":best,"parameters":GRID[best],"inner_case_macro_mae_mm":float(score[best]),
                        "n_inner_cases":len(folders),"tie_break":"first candidate_id"}
        write_csv(OUT/fold/"inner_mean_scores.csv",[{"candidate_id":i,"case_macro_mae_mm":float(v)} for i,v in enumerate(score)])
    write_json(OUT/"selected_parameters.json",selected)
    all_rows=[];final_audits=[]
    for fold in ["fold_a","fold_b"]:
        start=time.perf_counter()
        z,meta,tr,ev=read_split(fold,"final","test")
        config=replace(BASECFG,**selected[fold]["parameters"])
        model=SunPIGPR(config).fit(z["sensor_z"][tr],z["y_mm"][tr],z["time"][tr],z["case"][tr],z["X_ct"][tr,:4])
        components=model.predict_components(z["sensor_z"][ev],z["time"][ev],z["case"][ev],z["X_ct"][ev,:4])
        blob=pickle.dumps(model)
        (OUT/fold/"final_model.pkl").write_bytes(blob)
        loaded=pickle.loads(blob)
        np.testing.assert_array_equal(components["mean"],loaded.predict(z["sensor_z"][ev],z["time"][ev],z["case"][ev],z["X_ct"][ev,:4]))
        write_json(OUT/fold/"final_state.json",model.audit_state())
        idx=np.flatnonzero(ev)
        rows=[]
        for j,i in enumerate(idx):
            if not np.isfinite(z["y_mm"][i]):continue
            rows.append({"method":"sun_pigpr_causal_nasa","fold":fold,"case":int(z["case"][i]),"run":int(z["run"][i]),
                "time":float(z["time"][i]),"target_mm":float(z["y_mm"][i]),"pred_mm":float(components["mean"][j]),
                "physical_mean_mm":float(components["physical_mean"][j]),"gp_residual_mm":float(components["gp_residual"][j]),
                "latent_std_mm":float(components["latent_std"][j]),"scope":"valid_sensor_only"})
        write_csv(OUT/fold/"final_predictions.csv",rows);all_rows+=rows
        final_audits.append({"fold":fold,"seconds":time.perf_counter()-start,"train_cases":meta["split"]["train"],
            "test_cases":meta["split"]["test"],"n_test_labels":len(rows),"reload_exact":True,"selected_before_test_read":True})
    write_csv(OUT/"final_predictions.csv",all_rows)
    cases=[]
    for fold in ["fold_a","fold_b"]:
        for c in sorted(set(r["case"] for r in all_rows if r["fold"]==fold)):
            rr=[r for r in all_rows if r["fold"]==fold and r["case"]==c]
            cases.append({"fold":fold,"case":c,"n":len(rr),"mae_mm":float(np.mean([abs(r["pred_mm"]-r["target_mm"]) for r in rr]))})
    write_csv(OUT/"case_metrics.csv",cases)
    summaries=[]
    for f in ["overall","fold_a","fold_b"]:
        cs=[r for r in cases if f=="overall" or r["fold"]==f]
        summaries.append({"subset":f,"case_macro_mae_mm":float(np.mean([r["mae_mm"] for r in cs])),
                         "n_cases":len(cs),"n_labels":sum(r["n"] for r in cs)})
    write_csv(OUT/"summary.csv",summaries)
    inner_seconds=sum(json.loads(p.read_text())["seconds"] for p in OUT.glob("fold_*/inner_*/completed.json"))
    write_json(OUT/"audit.json",{"final_checks":final_audits,"n_candidates":len(GRID),"n_inner_splits":splits_done,
        "n_grid_evaluations":splits_done*len(GRID),"unique_final_fits":2,"stochastic_seeds":0,
        "total_inner_seconds":inner_seconds,"current_invocation_seconds":time.perf_counter()-started,
        "test_labels":len(all_rows),"scope":"valid_sensor_only; shared CT fallback to be merged by coordinator"})
    write_json(OUT/"status.json",{"stage":"complete","completed_inner":14,"completed_final":2,"test_labels":len(all_rows)})
    print("SUMMARY",json.dumps(summaries),flush=True)

if __name__=="__main__":
    with threadpool_limits(limits=1):main()

