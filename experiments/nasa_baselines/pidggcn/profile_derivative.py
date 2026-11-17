import copy,json,time
from pathlib import Path
import numpy as np
import torch
from model import PIDGGCN,diagonal_time_derivative,own_scalar_time_derivative
torch.set_num_threads(1)
ROOT=Path(__file__).parent

def evaluate(dtype,scalar,initial,x):
    model=PIDGGCN(dropout=0).to(dtype);model.load_state_dict(initial);model.eval()
    if scalar:
        leaves=[torch.tensor(v,dtype=dtype,requires_grad=True) for v in x[:,-1]];t=torch.stack(leaves)
    else:t=torch.tensor(x[:,-1],dtype=dtype,requires_grad=True)
    inputs=torch.cat([torch.tensor(x[:,:-1],dtype=dtype),t[:,None]],1)
    y=model(inputs)
    derivative=own_scalar_time_derivative(y,leaves,range(len(t))) if scalar else diagonal_time_derivative(y,t)
    loss=y.square().mean()+derivative.square().mean();loss.backward()
    grads=torch.cat([p.grad.flatten() for p in model.parameters()]).detach().double().numpy()
    return y.detach().double().numpy(),derivative.detach().double().numpy(),grads

torch.manual_seed(77);initial=copy.deepcopy(PIDGGCN(dropout=0).double().state_dict())
a=np.load(ROOT.parent/'data/fold_a/inner_1/arrays.npz');x=a['X'][(a['case']==3)&a['sensor_valid']]
outputs={};times={}
for dtype in (torch.float64,torch.float32):
 for scalar in (False,True):
    key=str(dtype)+'_'+str(scalar);t=time.perf_counter()
    for _ in range(3):out=evaluate(dtype,scalar,initial,x)
    times[key]=(time.perf_counter()-t)/3;outputs[key]=out
vector=outputs['torch.float64_False'];scalar=outputs['torch.float64_True']
err=[float(np.max(abs(a-b))) for a,b in zip(vector,scalar)]
for a,b in zip(vector,scalar):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-8)
f32=outputs['torch.float32_True'];f32err=[float(np.max(abs(a-b))) for a,b in zip(scalar,f32)]
assert all(np.isfinite(v).all() for v in f32)
agreement=all(np.allclose(a,b,atol=2e-5,rtol=2e-3) for a,b in zip(scalar,f32))
result={'seconds_per_case_forward_derivative_backward':times,'scalar_vs_vector_abs_errors_prediction_derivative_parameter_gradient':err,'float32_vs_float64_abs_errors':f32err,'float32_parameter_gradient_relative_l2':float(np.linalg.norm(scalar[2]-f32[2])/np.linalg.norm(scalar[2])),'scalar_equivalence_tests_pass':True,'float32_elementwise_agreement':agreement,'decision':'float64 scalar-time exact diagonal; float32 elementwise parameter gradient tolerance failed, no history detach'}
(ROOT/'derivative_profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
