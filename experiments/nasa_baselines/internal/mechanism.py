"""Random-start four-arm mechanism core; project components imported read-only."""
from wearflow_paths import ROOT as _WF_ROOT
import sys, copy, json, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.dont_write_bytecode=True
sys.path.insert(0, str(_WF_ROOT / ''))
from wear_monitoring_dynamics.models.feature_mlp import FeatureMLP
from wear_monitoring_dynamics.models.dynamics import WearDynamics
from wear_monitoring_dynamics.losses.derivative import derivative_loss
SUITE=Path(__file__).resolve().parents[1]
ARMS=('current_only','local_only','integral_only','both')
class Mechanism(nn.Module):
    def __init__(self):
        super().__init__()
        self.monitor=FeatureMLP({'selected_indices':list(range(12)), 'hidden_dims':[64,32]})
        self.dynamics=WearDynamics(32,20,True)
    def estimate(self,batch,indices,time=None):
        return self.monitor.estimate(batch,indices,time)
    def forecast(self,batch,pairs):
        a,b=pairs[:,0],pairs[:,1]
        return self.dynamics.propagate(self.estimate(batch,a),batch['condition'][a],batch['time'][b]-batch['time'][a])

def load_split(path):
    a=dict(np.load(path/'arrays.npz')); meta=json.loads((path/'metadata.json').read_text())
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32)
    batch={'sensor':tensor(a['sensor_z']),'condition':tensor(a['X_ct'][:,:4]),'time':tensor(a['X_ct'][:,4:5]),'target':tensor(a['y_z'][:,None]),'sensor_valid':torch.tensor(a['sensor_valid'])}
    obs=torch.tensor(np.flatnonzero(a['train_fit_mask']),dtype=torch.long)
    p=a['train_pairs']; p=p[a['sensor_valid'][p[:,0]]]
    assert np.all(a['train_fit_mask'][p[:,0]]) and np.all(a['train_mask'][p[:,1]])
    assert np.all(a['time'][p[:,1]]>a['time'][p[:,0]])
    assert np.allclose(batch['time'].numpy().ravel(),a['time']/meta['scalers']['time_scale'])
    return a,meta,batch,obs,torch.tensor(p,dtype=torch.long)

def components(model,batch,obs,pairs,arm):
    out={'current':(model.estimate(batch,obs)-batch['target'][obs]).square().mean()}
    if arm in ('local_only','both'): out['local']=derivative_loss(model,batch,obs)
    if arm in ('integral_only','both'): out['integral']=(model.forecast(batch,pairs)-batch['target'][pairs[:,1]]).square().mean()
    return out

def state_hash(model):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()

def gradients(loss,model):
    params=list(model.parameters()); g=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    n=len(list(model.monitor.parameters()))
    vec=torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params,g)])
    return vec,{'monitor':float(torch.linalg.vector_norm(torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params[:n],g[:n])]))),'rate':float(torch.linalg.vector_norm(torch.cat([(torch.zeros_like(p) if v is None else v).flatten() for p,v in zip(params[n:],g[n:])])))}

def smoke():
    torch.set_num_threads(1); torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    a,meta,batch,obs,pairs=load_split(SUITE/'data/fold_a/inner_1')
    initial=Mechanism(); initial_state=copy.deepcopy(initial.state_dict()); initial_hash=state_hash(initial)
    report={'lr':.001,'epochs':3,'initial_hash':initial_hash,'train_obs':len(obs),'train_pairs':len(pairs),'test_read':False,'arms':{}}
    baseline=initial.forecast(batch,pairs).detach(); start=time.perf_counter()
    for arm in ARMS:
        model=Mechanism(); model.load_state_dict(initial_state)
        assert state_hash(model)==initial_hash
        assert torch.equal(model.forecast(batch,pairs).detach(),baseline)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        rows=[]; t=time.perf_counter()
        for epoch in range(1,4):
            optimizer.zero_grad(set_to_none=True); losses=components(model,batch,obs,pairs,arm)
            diagnostics={}; vectors={}
            for name,loss in losses.items(): vectors[name],diagnostics[name]=gradients(loss,model)
            total=sum(losses.values()); total.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            rows.append({'epoch':epoch,'losses':{k:float(v.detach()) for k,v in losses.items()},'grad_norms':diagnostics})
        delta={key:max(float((model.state_dict()[k]-v).abs().max()) for k,v in initial_state.items() if k.startswith(key)) for key in ('monitor','dynamics')}
        assert delta['monitor']>0 and ((delta['dynamics']==0)==(arm=='current_only'))
        state=copy.deepcopy(model.state_dict()); reloaded=Mechanism(); reloaded.load_state_dict(state)
        assert torch.equal(model.forecast(batch,pairs),reloaded.forecast(batch,pairs))
        altered={k:v.clone() for k,v in batch.items()}; changed=int(pairs[0,1]); only=pairs[:1]
        altered['sensor'][changed]+=1000
        assert torch.equal(model.forecast(batch,only),model.forecast(altered,only))
        report['arms'][arm]={'seconds':time.perf_counter()-t,'rows':rows,'parameter_max_abs_changes':delta,'reload_exact':True,'future_sensor_invariance':True}
    report['seconds']=time.perf_counter()-start
    report['rough_96x100_training_seconds']=sum(v['seconds'] for v in report['arms'].values())/12*9600
    (Path(__file__).parent/'smoke_results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='arms'},indent=2))
if __name__=='__main__':smoke()

