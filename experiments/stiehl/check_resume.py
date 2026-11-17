"""Compare resumed smoke state to an uninterrupted deterministic two-step replay."""
import json
import torch
from models import WearModel,objective
from run_formal import ROOT,OUT,setup

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
state=torch.load(OUT/'test9_inner/integral_seed0/last.pt',weights_only=False)
assert state['epoch']==2, 'Run this check before the formal queue reaches test9_inner.'
d=setup('test9_inner');torch.manual_seed(0);model=WearModel();model.load_state_dict(state['initial'])
optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
ti=d['train_index']
for _ in range(2):
    optimizer.zero_grad()
    loss,_,_=objective(model,d['x'][ti],d['y'][ti],d['train_tools'],d['train_pairs'],d['train_dt'],'integral')
    loss.backward();optimizer.step()
assert all(torch.equal(v,state['model'][k]) for k,v in model.state_dict().items())
result={'passed':True,'comparison':'resumed epoch 2 versus uninterrupted two-step replay','all_parameters_bitwise_equal':True}
(ROOT/'resume_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
