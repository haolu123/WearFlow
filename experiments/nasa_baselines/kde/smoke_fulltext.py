"""Synthetic equation checks and actual NASA raw-window GPU training benchmark."""
from wearflow_paths import ROOT as _WF_ROOT
import json,time
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from method import *
ROOT=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    corrected,keep=adaptive_labels([1,2,3,4,5],[.1,.3,.2,.4,.35])
    np.testing.assert_allclose(corrected,[.1,.3,.35,.4,.4])
    t=np.arange(1.,8.);y=.001*t**3-.006*t**2+.04*t
    c=MonotoneCubic().fit(t,y)
    np.testing.assert_allclose(c.predict(t),y,atol=1e-6)
    assert np.min(c.derivative(t))>=-1e-8
    assert abs(c.crossing(float(c.predict(10)),after=7)-10)<1e-6
    assert abs(kde_mode([1.,1.,1.])-1)<1e-12
    assert kde_mode([1.,1.1,1.2,9.],bandwidth='literal_eq9')<3
    data=loadmat(str(_WF_ROOT / 'mill/mill.mat'),squeeze_me=True,struct_as_record=False)['mill']
    a=dict(np.load(ROOT.parent/'data/fold_a/inner_1/arrays.npz'))
    meta=json.loads((ROOT.parent/'data/fold_a/inner_1/metadata.json').read_text())
    keys={(int(c),int(r)) for c,r in zip(a['case'][a['train_fit_mask']],a['run'][a['train_fit_mask']])}
    channels=['smcAC','smcDC','vib_table','vib_spindle','AE_table','AE_spindle']
    selected=[r for r in data if (int(r.case),int(r.run)) in keys]
    windows=np.concatenate([crop_windows(np.column_stack([getattr(r,k) for k in channels]),r.feed) for r in selected])
    # Benchmark labels repeated current VB: timing only, not final interpolated-label training.
    y=np.repeat([r.VB for r in selected],75).astype('float32')
    location=windows.mean((0,1));scale=windows.std((0,1));windows=((windows-location)/np.maximum(scale,1e-8)).astype('float32')
    device='cuda' if torch.cuda.is_available() else 'cpu';net=WearTransformer().to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=1e-3)
    batch_size=32;x=torch.from_numpy(windows[:batch_size]).to(device);target=torch.from_numpy(y[:batch_size,None]).to(device)
    if device=='cuda':torch.cuda.reset_peak_memory_stats()
    durations=[]
    for step in range(5):
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();optimizer.zero_grad();pred=net(x);loss=(pred-target).square().mean();loss.backward();optimizer.step()
        assert torch.isfinite(loss)
        if device=='cuda':torch.cuda.synchronize()
        durations.append(time.perf_counter()-start)
    net.eval()
    with torch.no_grad():before=net(x)
    path=ROOT/'smoke_fulltext.pt';torch.save(net.state_dict(),path)
    other=WearTransformer().to(device);other.load_state_dict(torch.load(path,weights_only=True));other.eval()
    with torch.no_grad():torch.testing.assert_close(before,other(x),rtol=0,atol=0)
    per_batch=float(np.median(durations[1:]));per_epoch=math.ceil(len(windows)/batch_size)*per_batch
    report=dict(status='passed_smoke_only',architecture_parameters=sum(p.numel() for p in net.parameters()),device=device,
        gpu=torch.cuda.get_device_name() if device=='cuda' else None,peak_allocated_MB=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else None,
        train_labeled_records=len(selected),windows=len(windows),windows_per_cut=75,batch_size=batch_size,step_seconds=durations,
        estimated_100_epoch_member_seconds=per_epoch*100,estimated_100_members_hours=per_epoch*100*100/3600,
        note='Timing excludes per-epoch eval and polynomial fits; synthetic repeated labels in benchmark only. No formal models trained.',
        scaler_fit='Only fold_a inner_1 train_fit records',equations_checked=['adaptive interpolation','cubic recovery','threshold root','KDE mode','checkpoint reload'])
    (ROOT/'smoke_fulltext_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
