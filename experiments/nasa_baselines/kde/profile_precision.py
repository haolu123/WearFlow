"""Numerical/throughput precision comparison using training data only."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import copy,json,time,gc
import numpy as np
import torch
from method import WearTransformer
from raw_data import ROOT,build

def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(0)
    d=build('fold_a');x=torch.tensor(d['x'][d['train_record']].reshape(-1,300,6)[:128],device='cuda');y=torch.tensor(d['window_y'][d['train_record']].reshape(-1,1)[:128],device='cuda')
    initial=copy.deepcopy(WearTransformer().state_dict());results=[];base=None
    for name,dtype,precision in [('float32',None,'highest'),('tf32',None,'high'),('bf16',torch.bfloat16,'highest'),('fp16',torch.float16,'highest')]:
        torch.set_float32_matmul_precision(precision);net=WearTransformer().cuda();net.load_state_dict(initial);net.eval();net.zero_grad()
        with torch.autocast('cuda',dtype=dtype or torch.float16,enabled=dtype is not None):pred=net(x);loss=(pred.float()-y).square().mean()
        loss.backward();gradient=torch.cat([p.grad.flatten().float() for p in net.parameters()]).detach();values=(pred.float().detach(),loss.detach(),gradient)
        if base is None:base=tuple(v.clone() for v in values)
        error=dict(output_relative_l2=float(torch.linalg.vector_norm(values[0]-base[0])/torch.linalg.vector_norm(base[0])),loss_relative=float(abs(values[1]-base[1])/abs(base[1])),gradient_relative_l2=float(torch.linalg.vector_norm(gradient-base[2])/torch.linalg.vector_norm(base[2])),finite=bool(torch.isfinite(gradient).all()))
        net.train();opt=torch.optim.Adam(net.parameters(),lr=.001);scaler=torch.amp.GradScaler('cuda',enabled=name=='fp16');times=[];torch.cuda.reset_peak_memory_stats()
        for i in range(4):
            torch.cuda.synchronize();tick=time.perf_counter();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=dtype or torch.float16,enabled=dtype is not None):z=net(x);l=(z.float()-y).square().mean()
            scaler.scale(l).backward();scaler.step(opt);scaler.update();torch.cuda.synchronize();times.append(time.perf_counter()-tick)
        results.append(dict(mode=name,seconds=float(np.median(times[1:])),peak_MB=torch.cuda.max_memory_allocated()/2**20,**error))
        del net,opt,scaler;gc.collect();torch.cuda.empty_cache()
    (ROOT/'precision_profile.json').write_text(json.dumps(results,indent=2));print(json.dumps(results))
if __name__=='__main__':main()
