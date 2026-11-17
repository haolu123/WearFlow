"""Shared physical-state monitor and RK4 dynamics; no future sensor input."""
import torch
from torch import nn

class WearModel(nn.Module):
    def __init__(self,time_only=False):
        super().__init__()
        self.time_only=time_only
        self.monitor=nn.Sequential(nn.Linear(1 if time_only else 13,64),nn.SiLU(),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        self.rate=nn.Sequential(nn.Linear(1,32),nn.SiLU(),nn.Linear(32,32),nn.SiLU(),nn.Linear(32,1),nn.Softplus())

    def forward(self,x):
        return self.monitor(x[:,-1:] if self.time_only else x).squeeze(-1)

    def rhs(self,z):
        return self.rate(z.unsqueeze(-1)).squeeze(-1)

    def propagate(self,z,dt,steps=20):
        step=dt/steps
        for _ in range(steps):
            k1=self.rhs(z); k2=self.rhs(z+.5*step*k1)
            k3=self.rhs(z+.5*step*k2);k4=self.rhs(z+step*k3)
            z=z+step*(k1+2*k2+2*k3+k4)/6
        return z

def tool_mean(values,tools):
    return torch.stack([values[tools==t].mean() for t in torch.unique(tools)]).mean()

def objective(model,x,y,tools,pair_index,pair_dt,mode):
    x=x.detach().requires_grad_(mode=='local')
    current=model(x)
    loss=tool_mean((current-y).square(),tools)
    auxiliary=torch.zeros((),device=x.device)
    if mode=='local':
        partial=torch.autograd.grad(current.sum(),x,create_graph=True)[0][:,-1]
        auxiliary=tool_mean((partial-model.rhs(current)).square(),tools)
    if mode=='integral':
        i,j=pair_index.T
        future=model.propagate(current[i],pair_dt)
        auxiliary=tool_mean((future-y[j]).square(),tools[i])
    return loss+auxiliary,loss,auxiliary
