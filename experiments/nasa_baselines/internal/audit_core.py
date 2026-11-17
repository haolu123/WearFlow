"""Analytic derivative/units and shared data mask audit; no training."""
import json, sys
from pathlib import Path
import torch
from torch import nn
from mechanism import load_split,SUITE
from wear_monitoring_dynamics.losses.derivative import derivative_residual
class Rate(nn.Module):
    def forward(self,z,c):return torch.zeros_like(z)+.25
class Linear(nn.Module):
    def __init__(self):
        super().__init__(); self.slope=nn.Parameter(torch.tensor(.75)); self.dynamics=nn.Module(); self.dynamics.rate=Rate()
    def estimate(self,b,idx,time=None):return self.slope*time + b['sensor'][idx,:1]*2
b={'time':torch.tensor([[.1],[.7]]),'sensor':torch.tensor([[4.],[7.]]),'condition':torch.zeros(2,4)}
m=Linear(); r=derivative_residual(m,b,torch.arange(2)); assert torch.allclose(r,torch.full_like(r,.5))
r.square().mean().backward(); assert abs(m.slope.grad.item()-1)<1e-7
# z slope .75, rate .25, sigma=.2, T=10 gives physical residual .01 mm/sec.
assert abs((r.detach()*.2/10)[0].item()-.01)<1e-7
rows=[]
for path in sorted((SUITE/'data').glob('*/*')):
    if not (path/'arrays.npz').exists():continue
    a,meta,batch,obs,pairs=load_split(path)
    assert torch.isfinite(batch['sensor'][obs]).all() and torch.isfinite(batch['target'][obs]).all()
    train=set(meta['split']['train']); assert set(a['case'][obs]).issubset(train)
    assert set(a['case'][pairs[:,1]]).issubset(train)
    rows.append({'split':str(path.relative_to(SUITE/'data')),'obs':len(obs),'pairs':len(pairs),'sigma_vb':meta['scalers']['wear'][1],'time_scale':meta['scalers']['time_scale']})
result={'analytic_derivative':True,'derivative_parameter_gradient':True,'physical_unit_conversion':True,'splits':rows}
Path(__file__).with_name('data_derivative_audit.json').write_text(json.dumps(result,indent=2))
print('PASS',len(rows),'splits')

