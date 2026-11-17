"""Current-record MLP for arbitrary feature schemas; no temporal history or ODE."""
import torch
from torch import nn

class FeatureMLP(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.dynamics=None
        self.register_buffer('selected_indices',torch.tensor(cfg['selected_indices'],dtype=torch.long))
        width=len(cfg['selected_indices'])+5;layers=[]
        for hidden in cfg['hidden_dims']:
            layers.extend([nn.Linear(width,hidden),nn.SiLU()]);width=hidden
        self.net=nn.Sequential(*layers,nn.Linear(width,1))

    def estimate(self,batch,indices,time=None):
        t=batch['time'][indices] if time is None else time
        values=torch.cat([batch['sensor'][indices][:,self.selected_indices],batch['condition'][indices],t],dim=-1)
        pred=self.net(values)
        if len(self.selected_indices) and 'fallback_vb_z' in batch:
            pred=torch.where(batch['sensor_valid'][indices,None],pred,batch['fallback_vb_z'][indices])
        return pred
