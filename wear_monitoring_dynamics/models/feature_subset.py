"""Explicit feature selection without changing legacy 54-feature monitors.

Attention keeps the nine statistic slots per channel, zero pads unselected
slots, and removes channels with no selected features. It never attends over
time or mistakes adjacent selected columns for a complete sensor channel.
"""
import math
import torch
from torch import nn
from ..data.features import SENSORS, STATISTICS


def feature_indices(names):
    universe=[s+'__'+f for s in SENSORS for f in STATISTICS]
    if not names or len(set(names))!=len(names) or any(n not in universe for n in names):
        raise ValueError('Selected sensor features must be unique known names')
    return [universe.index(n) for n in names]


class FeatureSubsetMonitor(nn.Module):
    def __init__(self, monitor, architecture, names):
        super().__init__()
        self.inputs=monitor.inputs
        if self.inputs!=['sensor','condition','time']:
            raise ValueError('Feature subset requires sensor, condition, time inputs')
        indices=feature_indices(names)
        self.register_buffer('selected_indices',torch.tensor(indices,dtype=torch.long))
        self.network=monitor
        self.architecture=architecture
        n=len(indices)
        if architecture=='mlp':
            monitor.net[0]=nn.Linear(n+5,monitor.net[0].out_features)
        elif architecture=='time_encoded':
            monitor.net[0]=nn.Linear(n+11,monitor.net[0].out_features)
        elif architecture=='film':
            monitor.sensor_encoder[0]=nn.Linear(n,monitor.sensor_encoder[0].out_features)
        elif architecture=='sensor_attention':
            channels=sorted(set(i//9 for i in indices))
            slots=[channels.index(i//9)*9+i%9 for i in indices]
            self.register_buffer('feature_slots',torch.tensor(slots,dtype=torch.long))
            self.active_channels=channels
            monitor.channel_embedding=nn.Parameter(monitor.channel_embedding.detach()[channels].clone())
        else:
            raise ValueError('Feature subset unsupported for '+architecture)

    def forward(self,sensor,condition,time):
        selected=sensor.index_select(1,self.selected_indices)
        if self.architecture!='sensor_attention':
            return self.network(selected,condition,time)
        m=self.network;b=len(selected);n=len(self.active_channels)
        padded=selected.new_zeros((b,n*9)).index_copy(1,self.feature_slots,selected).reshape(b,n,9)
        context=m.encode_context(condition,time)
        tokens=m.norm(m.sensor_encoder(padded)+m.channel_embedding)
        query=m.query(context).reshape(b,3,1,8)
        key=m.key(tokens).reshape(b,n,3,8).transpose(1,2)
        value=m.value(tokens).reshape(b,n,3,8).transpose(1,2)
        weights=((query@key.transpose(-1,-2))/math.sqrt(8)).softmax(dim=-1)
        features=(weights@value).reshape(b,24)
        return m.head(torch.cat((features,context),dim=-1))
