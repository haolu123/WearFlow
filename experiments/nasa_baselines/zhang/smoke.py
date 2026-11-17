"""Small fixed-protocol smoke run; no full experiment dispatcher."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model import Monitor, ZhangPhysics, paired_loss


def adjacent(a):
    pairs = []
    for case in np.unique(a['case'][a['train_fit_mask']]):
        indices = np.flatnonzero(a['train_fit_mask'] & (a['case'] == case))
        indices = indices[np.argsort(a['time'][indices], kind='stable')]
        pairs.extend((int(i), int(j)) for i,j in zip(indices[:-1], indices[1:])
                     if a['time'][j] > a['time'][i])
    return np.asarray(pairs, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=Path, default=Path(__file__).resolve().parents[1]/'data/fold_a/inner_1')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'smoke')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); torch.manual_seed(0); np.random.seed(0)
    a = dict(np.load(args.split/'arrays.npz'))
    meta = json.loads((args.split/'metadata.json').read_text())
    # Standalone smoke refuses to silently omit labeled invalid records. Full adapter
    # must supply the same split/seed frozen CT fallback when any exist.
    for split in ('train','validation'):
        if np.any(a[split+'_mask'] & ~a['sensor_valid']):
            raise RuntimeError('Labeled invalid sensor requires shared frozen CT fallback')
    assert a['X'].shape[1] == 17
    x=torch.tensor(a['X'],dtype=torch.float32)
    y=torch.tensor(a['y_z'],dtype=torch.float32).reshape(-1,1)
    pairs=adjacent(a)
    if not len(pairs): raise ValueError('No valid positive-dt adjacent training pairs')
    pd.DataFrame(pairs,columns=['left_index','right_index']).to_csv(args.output/'train_pairs.csv',index=False)
    monitor,physics=Monitor(),ZhangPhysics(17)
    optimizer=torch.optim.Adam([{'params':monitor.parameters(),'lr':.01},
                                {'params':physics.parameters(),'lr':.001}])
    mean,std=meta['scalers']['wear']
    histories=[]; predictions=[]
    start=time.perf_counter()
    for epoch in range(1,4):
        monitor.train();physics.train();losses=np.zeros(4)
        lr=optimizer.param_groups[0]['lr']
        for batch in torch.randperm(len(pairs)).split(16):
            left,right=pairs[batch.numpy()].T
            optimizer.zero_grad()
            loss,terms=paired_loss(monitor,physics,x[left],x[right],y[left],y[right])
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            loss.backward()
            for p in list(monitor.parameters())+list(physics.parameters()):
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError('Nonfinite gradient')
            optimizer.step()
            losses+=np.array([v.item() for v in (loss,*terms)])*len(batch)/len(pairs)
        monitor.eval()
        with torch.no_grad(): pred=monitor(x).numpy().ravel()*std+mean
        record={'epoch':epoch,'lr_monitor':lr,'lr_physics':.001,
                **dict(zip(['loss','supervised','residual','direction'],losses))}
        for split in ('train','validation'):
            idx=np.flatnonzero(a[split+'_mask'])
            record[split+'_mae_mm']=float(np.mean(np.abs(pred[idx]-a['y_mm'][idx])))
            for i in idx:
                predictions.append(dict(epoch=epoch,split=split,case=int(a['case'][i]),run=int(a['run'][i]),
                    time=float(a['time'][i]),target_mm=float(a['y_mm'][i]),prediction_mm=float(pred[i]),
                    sensor_valid=bool(a['sensor_valid'][i])))
        histories.append(record)
        # Author's scheduler is advanced after epoch; full horizon stays 100.
        index=epoch-1
        optimizer.param_groups[0]['lr']=(.01+(.003-.01)*index/9 if index<10 else
            .0001+.5*(.003-.0001)*(1+math.cos(math.pi*(index-10)/90)))
        pd.DataFrame(histories).to_csv(args.output/'history.csv',index=False)
        pd.DataFrame(predictions).to_csv(args.output/'predictions.csv.gz',index=False,compression='gzip')
    elapsed=time.perf_counter()-start
    torch.save({'monitor':monitor.state_dict(),'physics':physics.state_dict(),'epoch':3},args.output/'last.pt')
    reloaded=Monitor();reloaded.load_state_dict(torch.load(args.output/'last.pt',weights_only=True)['monitor']);reloaded.eval()
    with torch.no_grad(): torch.testing.assert_close(monitor(x),reloaded(x),rtol=0,atol=0)
    report=dict(status='smoke_only_passed',epochs=3,schedule_horizon=100,elapsed_seconds=elapsed,
        rough_100_epoch_seconds=elapsed/3*100,pairs=len(pairs),seed=0,device='cpu',threads=1,
        monitor_parameters=sum(p.numel() for p in monitor.parameters()),
        physics_parameters=sum(p.numel() for p in physics.parameters()),checkpoint_reload_exact=True,
        input_sha256=hashlib.sha256((args.split/'arrays.npz').read_bytes()).hexdigest(),split=str(args.split))
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))

if __name__=='__main__':main()
