"""Finite, strictly causal sensor windows with explicit elapsed-time inputs."""
import torch
from torch import nn
from torch.nn import functional as F
from .feature_subset import feature_indices


class HistoryMonitor(nn.Module):
    def __init__(self, architecture, length, features):
        super().__init__()
        self.architecture, self.length = architecture, length
        self.register_buffer('selected_indices', torch.tensor(feature_indices(features)))
        width = len(features) + 4 + 2  # conditions, absolute time, elapsed time
        if architecture == 'history_mlp':
            hidden = max(8, round(6000 / (length * (width + 1) + 34)))
            self.net = nn.Sequential(nn.Linear(length * (width + 1), hidden), nn.SiLU(),
                                     nn.Linear(hidden, 32), nn.SiLU(), nn.Linear(32, 1))
        elif architecture == 'history_gru':
            self.cell = nn.GRUCell(width, 32)
            self.head = nn.Linear(32, 1)
        elif architecture == 'history_cnn':
            self.project = nn.Linear(width, 20)
            self.convs = nn.ModuleList([nn.Conv1d(20, 20, 3, dilation=d) for d in (1, 2, 4, 8)])
            self.head = nn.Linear(20, 1)
        else:
            raise ValueError(architecture)

    def forward(self, values, mask):
        values = values * mask.unsqueeze(-1)
        if self.architecture == 'history_mlp':
            return self.net(torch.cat([values, mask.unsqueeze(-1).to(values.dtype)], -1).flatten(1))
        if self.architecture == 'history_gru':
            state = values.new_zeros(len(values), 32)
            for k in range(values.shape[1]):
                state = torch.where(mask[:, k:k+1], self.cell(values[:, k], state), state)
            return self.head(state)
        hidden = self.project(values).transpose(1, 2) * mask.unsqueeze(1)
        for conv in self.convs:
            hidden = (hidden + F.silu(conv(F.pad(hidden, (2 * conv.dilation[0], 0))))) * mask.unsqueeze(1)
        return self.head(hidden[:, :, -1])


class HistoryCoupled(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.monitor = HistoryMonitor(cfg['architecture'], cfg['history_length'], cfg['selected_features'])
        self.dynamics = None

    def estimate(self, batch, indices, time=None):
        if time is not None:
            raise ValueError('History monitoring uses actual causal windows, not partial-time derivatives')
        idx = batch['history_indices'][indices]
        mask = batch['history_mask'][indices]
        values = torch.cat([batch['sensor'][idx][..., self.monitor.selected_indices],
                            batch['condition'][idx], batch['time'][idx], batch['history_delta'][indices]], -1)
        prediction = self.monitor(values, mask)
        if 'fallback_vb_z' in batch:
            prediction = torch.where(batch['sensor_valid'][indices,None],prediction,batch['fallback_vb_z'][indices])
        return prediction
