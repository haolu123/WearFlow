"""Explain legacy/new sensor differences by replaying the original RNG consumption."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..config import write_json
from ..models.registry import build_model
from .run_sensor_mechanisms import context, fit, predict_raw, Net, ROOT, OLD


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    out=ROOT/'initialization_audit';out.mkdir(exist_ok=True)
    mf=pd.read_csv(OLD/'reused_sensor_manifest.csv');rows=[]
    for r in mf[mf.role=='final'].itertuples():
        c=context(r);cfg=c['cfg'];torch.manual_seed(int(r.seed))
        # Legacy make_monitor_batch constructs a random fallback CT before loading
        # its checkpoint. It therefore consumes RNG before creating the predictor.
        fallback=torch.load(cfg['monitor']['fallback_checkpoint'],map_location='cpu',weights_only=True)
        build_model(fallback['config'])
        model=Net(23,[64,32])
        init={k:v.clone() for k,v in model.state_dict().items()}
        fit(model,c['sensor'],c['y'],c['train'],500)
        pred=predict_raw(model,c['sensor'],c)
        old=torch.load(Path(r.run_dir)/'last.pt',map_location='cpu',weights_only=True)
        oldmodel=Net(23,[64,32]);oldmodel.load_state_dict({k:v for k,v in old['model'].items() if k.startswith('net.')})
        oldpred=predict_raw(oldmodel,c['sensor'],c)
        ids=c['evals']['test'];target=c['table']['VB'];cases=c['table']['case']
        for case in np.unique(cases[ids]):
            ix=ids[cases[ids]==case]
            rows.append(dict(fold=r.fold,seed=int(r.seed),case=int(case),replayed_mae=float(abs(pred[ix]-target[ix]).mean()),
                original_mae=float(abs(oldpred[ix]-target[ix]).mean()),max_abs_prediction_difference=float(abs(pred[ix]-oldpred[ix]).max())))
        directory=out/r.fold/f'seed{int(r.seed)}';directory.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),directory/'last.pt');torch.save(init,directory/'initial.pt')
        np.savez_compressed(directory/'predictions.npz',case=cases,run=c['table']['run'],target=target,replayed=pred,original=oldpred)
    df=pd.DataFrame(rows);df.to_csv(out/'replay_comparison.csv',index=False)
    maxdiff=float(df.max_abs_prediction_difference.max())
    write_json(out/'audit.json',dict(replay_runs=10,epochs_each=500,max_abs_difference_cpu_vs_original=maxdiff,
        legacy_rng='fallback CT construction consumes random numbers before sensor initialization',
        controlled_rng='each predictor directly initialized with its seed; no fallback RNG consumption',
        interpretation='same seed integer does not imply identical predictor weights across runners'))
    print(df.groupby('fold')[['replayed_mae','original_mae']].mean().to_string());print('maxdiff',maxdiff)


if __name__=='__main__':main()
