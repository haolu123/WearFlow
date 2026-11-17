"""Sun2020 forecasting module only; explicit NASA port of pinned CuDNNLSTM architecture."""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class SunLSTM(nn.Module):
    def __init__(self, horizon=3):
        super().__init__()
        self.horizon=horizon
        self.encoder1=nn.LSTM(1,64,batch_first=True)
        self.encoder2=nn.LSTM(64,32,batch_first=True)
        self.bridge=nn.Linear(32,32)
        self.decoder1=nn.LSTM(32,32,batch_first=True)
        self.decoder2=nn.LSTM(32,16,batch_first=True)
        self.head=nn.Linear(16,1)
        self.reset_parameters()
    def reset_parameters(self):
        # Documented migration choice: Keras-style Glorot input / orthogonal recurrent,
        # one effective +1 forget bias. No claim of reproducing original random draws.
        for module in self.modules():
            if isinstance(module,nn.LSTM):
                nn.init.xavier_uniform_(module.weight_ih_l0)
                nn.init.orthogonal_(module.weight_hh_l0)
                nn.init.zeros_(module.bias_ih_l0);nn.init.zeros_(module.bias_hh_l0)
                n=module.hidden_size
                with torch.no_grad():module.bias_ih_l0[n:2*n].fill_(1.)
            elif isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight);nn.init.zeros_(module.bias)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(2,1):raise ValueError("Expected [batch,history=2,feature=1]")
        z,_=self.encoder1(x);z,_=self.encoder2(z)
        z=self.bridge(z[:,-1,:]).unsqueeze(1).repeat(1,self.horizon,1)
        z,_=self.decoder1(z);z,_=self.decoder2(z)
        return self.head(z).squeeze(-1)

def logcosh_loss(pred,target):
    error=pred-target
    return (error+F.softplus(-2*error)-math.log(2.)).mean()

def numpy_lstm(x,module):
    w=module.weight_ih_l0.detach().numpy();u=module.weight_hh_l0.detach().numpy()
    b=(module.bias_ih_l0+module.bias_hh_l0).detach().numpy()
    n=module.hidden_size;h=np.zeros((len(x),n));c=h.copy();out=[]
    for t in range(x.shape[1]):
        z=x[:,t]@w.T+h@u.T+b
        i,f,g,o=np.split(z,4,axis=1)
        i=1/(1+np.exp(-i));f=1/(1+np.exp(-f));o=1/(1+np.exp(-o))
        c=f*c+i*np.tanh(g);h=o*np.tanh(c);out.append(h.copy())
    return np.stack(out,axis=1)
def numpy_forward(x,model):
    z=numpy_lstm(x,model.encoder1);z=numpy_lstm(z,model.encoder2)[:,-1]
    z=z@model.bridge.weight.detach().numpy().T+model.bridge.bias.detach().numpy()
    z=np.repeat(z[:,None],model.horizon,axis=1)
    z=numpy_lstm(z,model.decoder1);z=numpy_lstm(z,model.decoder2)
    return (z@model.head.weight.detach().numpy().T+model.head.bias.detach().numpy()).squeeze(-1)

