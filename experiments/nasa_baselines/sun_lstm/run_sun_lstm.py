from __future__ import annotations
import os
for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ[k]="1"
import argparse,csv,gzip,hashlib,json,time,math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import SunLSTM,logcosh_loss,numpy_forward
BASE=Path(__file__).resolve().parent;SUITE=BASE.parent
def write(path,d):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(key):
    man=json.loads((SUITE/"data_manifest.json").read_text());e=next(e for e in man["splits"] if e["key"]==key)
    p=Path(e["directory"]);assert sha(p/"arrays.npz")==e["arrays_sha256"]
    with np.load(p/"arrays.npz",allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
    m=json.loads((p/"metadata.json").read_text());return d,m,e
def windows(d,mask,full):
    out=[]
    for case in np.unique(d["case"][mask]):
        idx=np.flatnonzero(mask & (d["case"]==case));idx=idx[np.argsort(d["time"][idx],kind="stable")]
        for k in range(1,len(idx)-1):
            targets=idx[k+1:k+4]
            if full and len(targets)<3:continue
            history=idx[k-1:k+1]
            assert np.all(d["time"][targets]>d["time"][history[-1]])
            out.append((history,targets))
    return out
def history_x(win,series):return np.stack([series[h] for h,t in win])[:,:,None]
def audit():
    torch.manual_seed(17);model=SunLSTM().double()
    x=torch.tensor([[[.1],[.2]],[[-.3],[.7]]],dtype=torch.float64,requires_grad=True)
    y=model(x);ref=numpy_forward(x.detach().numpy(),model)
    forward=float(np.max(abs(y.detach().numpy()-ref)));assert forward<1e-12
    y.sum().backward();analytic=float(x.grad[0,0,0]);eps=1e-6
    xp=x.detach().clone();xm=x.detach().clone();xp[0,0,0]+=eps;xm[0,0,0]-=eps
    finite=float(((model(xp).sum()-model(xm).sum())/(2*eps)).detach());assert abs(analytic-finite)<1e-8
    # Endpoint selection uses only two past labelled records. Perturb all future
    # sensor-monitor estimates after a chosen origin; constructed input is identical.
    d=dict(case=np.ones(7,dtype=int),time=np.arange(7,dtype=float))
    win=windows(d,np.ones(7,dtype=bool),False);series=np.arange(7,dtype=float)
    changed=series.copy();changed[win[0][0][-1]+1:]+=100
    np.testing.assert_array_equal(history_x(win[:1],series),history_x(win[:1],changed))
    loss=logcosh_loss(torch.tensor([1e4,-1e4]),torch.zeros(2));assert torch.isfinite(loss)
    return dict(numpy_forward_max_error=forward,input_gradient_error=abs(analytic-finite),
        future_monitor_perturbation_pass=True,stable_logcosh_pass=True,
        note="NumPy gate-equation equivalence, not an executed original Keras runtime")
def eval_rows(model,win,d,m,series,fold,seed,epoch,part,initial):
    mean,std=m["scalers"]["wear"]
    x=torch.tensor(history_x(win,series),dtype=torch.float32)
    with torch.no_grad():p=model(x).numpy()*std+mean
    rows=[]
    for j,(hist,targets) in enumerate(win):
        i=int(hist[-1])
        for h,t in enumerate(targets,1):
            rows.append(dict(method="sun2020_lstm_module",fold=fold,seed=seed,epoch=epoch,partition=part,
                initialization=initial,case=int(d["case"][i]),origin_run=int(d["run"][i]),
                target_run=int(d["run"][t]),horizon=h,dt=float(d["time"][t]-d["time"][i]),
                target_mm=float(d["y_mm"][t]),pred_mm=float(p[j,h-1])))
    return rows
def metrics(rows):
    f=pd.DataFrame(rows);f["ae"]=abs(f.pred_mm-f.target_mm)
    out=[]
    for (part,init,h),g in f.groupby(["partition","initialization","horizon"]):
        out.append(dict(partition=part,initialization=init,horizon=int(h),n=len(g),cases=int(g.case.nunique()),
            case_mae_mm=float(g.groupby("case").ae.mean().mean())))
    return out
def one(fold,seed,epochs,smoke):
    key=fold+"/inner_1" if smoke else fold+"/final"
    d,m,e=load(key);mean,std=m["scalers"]["wear"]
    root=BASE/("smoke" if smoke else "results")/"runs"/fold/f"seed{seed}";root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);rng=np.random.default_rng(seed);model=SunLSTM()
    train=windows(d,d["train_mask"],True);assert train
    train_eval=windows(d,d["train_mask"],False)
    part="validation" if smoke else "test";ev=windows(d,d[part+"_mask"],False);assert ev
    assert set(d["case"][d["train_mask"]]).isdisjoint(set(d["case"][d[part+"_mask"]]))
    refp=SUITE/"references"/f"{fold}_seed{seed}_monitor.npz"
    monitor=None
    if not smoke:
        with np.load(refp) as r:
            for k in ("case","run","time"):np.testing.assert_array_equal(r[k],d[k])
            monitor=(r["pred_mm"].astype(float)-mean)/std
        assert np.isfinite(monitor).all()
    X=torch.tensor(history_x(train,d["y_z"]),dtype=torch.float32)
    Y=torch.tensor(np.stack([d["y_z"][t] for h,t in train]),dtype=torch.float32)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,eps=1e-7)
    identity=dict(fold=fold,seed=seed,epochs=epochs,history=2,horizon=3,train_windows=len(train),
        arrays_sha256=e["arrays_sha256"],monitor_sha256=None if smoke else sha(refp),
        model_code_sha256=sha(BASE/"model.py"),runner_sha256=sha(Path(__file__)),lr=.001,
        batch_size=16,loss="logcosh on train-split y_z",parameter_count=sum(p.numel() for p in model.parameters()))
    write(root/"identity.json",identity);torch.save(model.state_dict(),root/"epoch0.pt")
    begin=time.perf_counter();history=[];finalrows=[];grad=None
    with gzip.open(root/"epoch_predictions.csv.gz","wt",newline="",encoding="utf-8") as f:
        writer=None
        for epoch in range(epochs+1):
            lossval=None
            if epoch:
                model.train();order=rng.permutation(len(X));num=0.;den=0
                for off in range(0,len(order),16):
                    idx=order[off:off+16];optimizer.zero_grad();loss=logcosh_loss(model(X[idx]),Y[idx]);loss.backward()
                    if epoch==1 and grad is None:grad=float(torch.sqrt(sum(p.grad.square().sum() for p in model.parameters() if p.grad is not None)))
                    optimizer.step();num+=float(loss.detach())*len(idx);den+=len(idx)
                lossval=num/den
            model.eval();rows=[]
            for name,win in (("train",train_eval),(part,ev)):
                rows+=eval_rows(model,win,d,m,d["y_z"],fold,seed,epoch,name,"oracle")
                if monitor is not None:rows+=eval_rows(model,win,d,m,monitor,fold,seed,epoch,name,"estimated")
            if writer is None:writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
            writer.writerows(rows)
            for v in metrics(rows):history.append(dict(epoch=epoch,train_loss=lossval,**v))
            if epoch==epochs:finalrows=rows
    torch.save(dict(state_dict=model.state_dict(),wear_scaler=[mean,std],identity=identity),root/"last.pt")
    loaded=SunLSTM();loaded.load_state_dict(torch.load(root/"last.pt",weights_only=False)["state_dict"]);loaded.eval()
    with torch.no_grad():err=float((model(X)-loaded(X)).abs().max())
    assert err==0 and grad>0
    pd.DataFrame(history).to_csv(root/"history.csv",index=False)
    pd.DataFrame(finalrows).to_csv(root/"final_predictions.csv",index=False)
    elapsed=time.perf_counter()-begin
    write(root/"audit.json",dict(reload_max_error=err,first_gradient_norm=grad,elapsed_seconds=elapsed,
        epochs=epochs,train_windows=len(train),evaluation_keys_unique=not pd.DataFrame(finalrows).duplicated(["partition","initialization","case","origin_run","target_run","horizon"]).any(),
        monitor_used_only_after_training=not smoke,train_input="measured VB history only",future_sensor_input=False))
    return dict(fold=fold,seed=seed,run_dir=str(root),elapsed_seconds=elapsed,epochs=epochs)
def run(smoke):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    out=BASE/("smoke" if smoke else "results");out.mkdir(parents=True,exist_ok=True)
    write(out/"status.json",dict(status="auditing"))
    write(out/"equivalence_audit.json",audit())
    tasks=[("fold_a",0)] if smoke else [(f,s) for f in ("fold_a","fold_b") for s in range(5)]
    manifest=[];start=time.perf_counter()
    for fold,seed in tasks:
        manifest.append(one(fold,seed,3 if smoke else 100,smoke))
        write(out/"manifest.json",manifest);write(out/"status.json",dict(status="running",completed=len(manifest),planned=len(tasks)))
    fs=pd.concat([pd.read_csv(Path(t["run_dir"])/"final_predictions.csv") for t in manifest],ignore_index=True)
    fs.to_csv(out/"final_predictions.csv",index=False)
    if not smoke:
        test=fs[(fs.partition=="test")&(fs.initialization=="estimated")]
        test.to_csv(out/"future_predictions.csv",index=False)
        fs["ae"]=abs(fs.pred_mm-fs.target_mm);fs["se"]=(fs.pred_mm-fs.target_mm)**2
        cm=fs.groupby(["partition","initialization","fold","seed","case","horizon"],as_index=False).agg(mae_mm=("ae","mean"),n=("ae","size"))
        cm.to_csv(out/"case_seed_metrics.csv",index=False)
        summary=[]
        for init in ("estimated","oracle"):
            for subset in ("overall","fold_a","fold_b","fold_a_in","fold_a_out"):
                f=fs[(fs.partition=="test")&(fs.initialization==init)]
                if subset!="overall":f=f[f.fold==("fold_b" if subset=="fold_b" else "fold_a")]
                if subset=="fold_a_in":f=f[f.target_mm<=.74]
                if subset=="fold_a_out":f=f[f.target_mm>.74]
                for h,g in f.groupby("horizon"):
                    case_seed=g.groupby(["fold","case","seed"]).ae.mean()
                    summary.append(dict(initialization=init,subset=subset,horizon=int(h),
                        case_mae_mm=float(case_seed.groupby(["fold","case"]).mean().mean()),
                        n_per_seed=len(g)//5,cases=int(g.case.nunique())))
        pd.DataFrame(summary).to_csv(out/"summary.csv",index=False)
    write(out/"status.json",dict(status="smoke_passed" if smoke else "completed",runs=len(manifest),epochs=3 if smoke else 100,
        elapsed_seconds=time.perf_counter()-start,run_elapsed_seconds=sum(t["elapsed_seconds"] for t in manifest)))
    print(json.dumps(json.loads((out/"status.json").read_text()),indent=2))
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--smoke",action="store_true");a=ap.parse_args()
    try:run(a.smoke)
    except Exception as e:
        write(BASE/("smoke" if a.smoke else "results")/"status.json",dict(status="failed",error=repr(e)));raise

