"""Resume-safe 100-member raw Transformer study. Fixed last epoch, no selection."""
from wearflow_paths import ROOT as _WF_ROOT
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from method import WearTransformer
from raw_data import build,cut_predictions,ROOT
OUT=ROOT/'formal'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def jsonwrite(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2),encoding='utf8');tmp.replace(p)
def torchwrite(p,value):
    tmp=p.with_suffix(p.suffix+'.tmp');torch.save(value,tmp);tmp.replace(p)

def evaluate(net,data,batch):
    net.eval();pred=[]
    with torch.no_grad():
        flat=data['x'].reshape(-1,300,6)
        for offset in range(0,len(flat),batch):pred.append(net(torch.from_numpy(flat[offset:offset+batch]).cuda()).cpu().numpy().ravel())
    return np.concatenate(pred).reshape(-1,75)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--max-members',type=int);parser.add_argument('--stop-after-epoch',type=int);args=parser.parse_args()
    OUT.mkdir(exist_ok=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    profile=json.loads((ROOT/'batch_profile.json').read_text());batch=profile['selected_batch']
    sources={n:sha(ROOT/n) for n in ['method.py','raw_data.py','run_raw_ensemble.py']}
    protocol=dict(sources=sources,batch=batch,epochs=100,optimizer='Adam',lr=.001,dtype='float32',deterministic=True,
        fold_count=2,ensemble_seeds=5,members=10,architecture=dict(channels=6,embedding=20,heads=4,layers=5,feedforward=2048,dropout=.1),
        labels='training case monotone deletion then linear interpolation at crop window right endpoints in run-1+[300:50:4000]/4000',
        smoothing='trailing150 windows case-reset',coordinate='original run index; no origin reset',
        pdf_sha='57cb6b7a03f8474e6750b55a9b49aab6c8157510855753124ec4ed2338a637d5',raw_sha=sha(str(_WF_ROOT / 'mill/mill.mat')))
    frozen=OUT/'protocol.json'
    if frozen.exists():assert json.loads(frozen.read_text())==protocol,'Protocol identity mismatch'
    else:jsonwrite(frozen,protocol)
    pd.DataFrame([dict(fold=f,ensemble=e,member=m,seed=20261103+e*1000+m,epochs=100,run_dir=str(OUT/f/f'ensemble_{e}'/f'member_{m}')) for f in ['fold_a','fold_b'] for e in range(5) for m in range(10)]).to_csv(OUT/'study_manifest.csv',index=False)
    completed=0;invocation=time.perf_counter();alljobs=[]
    for fold in ['fold_a','fold_b']:
        d=build(fold);a=d['a'];ids=d['record_indices'];base=ROOT.parent/'data'/fold/'final'
        jsonwrite(OUT/(fold+'_data_audit.json'),dict(mean=d['mean'].tolist(),std=d['std'].tolist(),anchors=d['anchors'],
            record_indices=ids.tolist(),train_records=int(d['train_record'].sum()),test_labeled=int(a['test_mask'].sum()),
            schema='6 raw channels',split=d['meta']['split'],data_sha=sha(base/'arrays.npz'),metadata_sha=sha(base/'metadata.json')))
        assert not np.any(a['test_mask'] & ~a['sensor_valid']),'No CT fallback implemented for invalid labeled test sensor'
        x=torch.from_numpy(d['x'][d['train_record']].reshape(-1,300,6)).cuda();y=torch.from_numpy(d['window_y'][d['train_record']].reshape(-1,1)).cuda()
        for ensemble in range(5):
            for member in range(10):
                seed=20261103+ensemble*1000+member
                identity=dict(protocol=protocol,fold=fold,ensemble=ensemble,member=member,seed=seed,data_sha=sha(base/'arrays.npz'),metadata_sha=sha(base/'metadata.json'))
                folder=OUT/fold/f'ensemble_{ensemble}'/f'member_{member}';folder.mkdir(parents=True,exist_ok=True)
                if (folder/'completed.json').exists():
                    assert json.loads((folder/'identity.json').read_text())==identity
                    completed+=1;continue
                if args.max_members is not None and len(alljobs)>=args.max_members:return
                jsonwrite(folder/'identity.json',identity);torch.manual_seed(seed);net=WearTransformer().cuda();opt=torch.optim.Adam(net.parameters(),lr=.001)
                start_epoch=0;history=[];seconds=0.
                if (folder/'latest.pt').exists():
                    state=torch.load(folder/'latest.pt',weights_only=True);assert state['identity']==identity
                    net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);start_epoch=state['epoch'];seconds=state['elapsed_seconds']
                    torch.set_rng_state(state['rng_cpu'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in state['rng_cuda']]);history=state['history']
                else:
                    torchwrite(folder/'initial.pt',dict(model=net.state_dict(),seed=seed,identity=identity))
                    initial_windows=evaluate(net,d,batch)
                    np.savez_compressed(folder/'epoch_000_predictions.npz',window_pred_mm=initial_windows,pred_mm=cut_predictions(initial_windows,d),record_indices=ids)
                for epoch in range(start_epoch+1,101):
                    tick=time.perf_counter();net.train();loss_sum=0.;order=torch.randperm(len(x),device='cuda')
                    for ix in order.split(batch):
                        opt.zero_grad(set_to_none=True);loss=(net(x[ix])-y[ix]).square().mean();assert torch.isfinite(loss)
                        loss.backward()
                        assert torch.stack([torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None]).all(),'Nonfinite gradient'
                        opt.step();loss_sum+=float(loss.detach())*len(ix)
                    window_pred=evaluate(net,d,batch);cuts=cut_predictions(window_pred,d)
                    row=dict(epoch=epoch,train_window_mse=loss_sum/len(x))
                    for label in ['train','test']:
                        mask=a[label+'_mask'][ids]
                        row[label+'_raw_label_mae_mm']=float(np.abs(cuts[mask]-a['y_mm'][ids][mask]).mean())
                    details={'window_pred_mm':window_pred} if epoch==100 else {}
                    np.savez_compressed(folder/f'epoch_{epoch:03d}_predictions.npz',pred_mm=cuts,raw_last_window_mm=window_pred[:,-1],raw_mean_window_mm=window_pred.mean(1),record_indices=ids,case=a['case'][ids],run=a['run'][ids],target_mm=a['y_mm'][ids],train_mask=a['train_mask'][ids],test_mask=a['test_mask'][ids],**details)
                    history.append(row);pd.DataFrame(history).to_csv(folder/'history.csv',index=False)
                    torchwrite(folder/f'epoch_{epoch:03d}.pt',dict(model=net.state_dict(),epoch=epoch,identity=identity))
                    seconds+=time.perf_counter()-tick
                    torchwrite(folder/'latest.pt',dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch,identity=identity,history=history,elapsed_seconds=seconds,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all()))
                    jsonwrite(OUT/'status.json',dict(status='running',completed_members=completed,total_members=100,fold=fold,ensemble=ensemble,member=member,epoch=epoch,elapsed_invocation_seconds=time.perf_counter()-invocation))
                    if args.stop_after_epoch==epoch:
                        print(json.dumps(dict(profile_complete_epoch=epoch,seconds_including_eval_io=time.perf_counter()-tick)),flush=True);return
                reloaded=WearTransformer().cuda();reloaded.load_state_dict(torch.load(folder/'latest.pt',weights_only=True)['model'])
                if start_epoch==100:
                    if not (folder/'epoch_100_predictions.npz').exists():raise RuntimeError('Latest epoch100 exists but final prediction archive missing; restore/report repair required')
                    window_pred=np.load(folder/'epoch_100_predictions.npz')['window_pred_mm']
                np.testing.assert_array_equal(evaluate(reloaded,d,batch),window_pred);del reloaded
                jsonwrite(folder/'completed.json',dict(epochs=100,seconds=seconds,identity_hash=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()))
                completed+=1;alljobs.append(str(folder));print(f'{completed}/100 {fold} ensemble{ensemble} member{member} {seconds:.1f}s',flush=True)
        del x,y
    jsonwrite(OUT/'status.json',dict(status='training_complete',completed_members=completed,total_members=100))
if __name__=='__main__':
    try:main()
    except Exception:
        OUT.mkdir(exist_ok=True);(OUT/'execution_error.log').write_text(traceback.format_exc());raise
