"""Hardware-only batch profiling, then one complete training epoch; no test scoring."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,json,time
import numpy as np
import torch
from method import WearTransformer
from raw_data import build,ROOT

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    d=build('fold_a');x=torch.tensor(d['x'][d['train_record']].reshape(-1,300,6),device='cuda');y=torch.tensor(d['window_y'][d['train_record']].reshape(-1,1),device='cuda')
    results=[]
    for size in [32,64,128,256,512]:
        torch.manual_seed(0);net=WearTransformer().cuda();opt=torch.optim.Adam(net.parameters(),lr=.001);times=[]
        try:
            torch.cuda.reset_peak_memory_stats()
            for step in range(4):
                torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
                loss=(net(x[:size])-y[:size]).square().mean();loss.backward();opt.step();torch.cuda.synchronize();times.append(time.perf_counter()-t)
            results.append(dict(batch=size,seconds=float(np.median(times[1:])),windows_per_second=size/float(np.median(times[1:])),peak_MB=torch.cuda.max_memory_allocated()/2**20))
        except torch.cuda.OutOfMemoryError:
            results.append(dict(batch=size,oom=True))
        del net,opt;gc.collect();torch.cuda.empty_cache()
    usable=[r for r in results if not r.get('oom')];batch=max(usable,key=lambda r:r['windows_per_second'])['batch']
    torch.manual_seed(0);net=WearTransformer().cuda();opt=torch.optim.Adam(net.parameters(),lr=.001)
    torch.cuda.synchronize();start=time.perf_counter()
    order=torch.randperm(len(x),device='cuda')
    for ids in order.split(batch):
        opt.zero_grad(set_to_none=True);loss=(net(x[ids])-y[ids]).square().mean();loss.backward();opt.step()
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    result=dict(profiles=results,selected_batch=batch,train_records=int(d['train_record'].sum()),train_windows=len(x),full_training_epoch_seconds=elapsed,estimated_100_member_100_epoch_hours=elapsed*10000/3600,selection='maximum measured throughput; no held-out error inspected',dtype='float32',deterministic=True)
    (ROOT/'batch_profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
