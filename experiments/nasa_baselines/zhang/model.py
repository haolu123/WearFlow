"""Input-width-only adapter to the verified Zhang author architecture."""
from wearflow_paths import ROOT as _WF_ROOT
import sys
sys.path.insert(0, str(_WF_ROOT / ''))
import torch
from torch import nn
from wear_monitoring_dynamics.models.zhang import ZhangMLP, ZhangPredictor, ZhangPhysics


class Monitor(nn.Module):
    def __init__(self, input_dim=17):
        super().__init__()
        self.encoder = ZhangMLP(input_dim, 32, 3, 32)
        self.predictor = ZhangPredictor()
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward_features(self, xt):
        return self.predictor(self.encoder.net(xt))

    forward = forward_features


def paired_loss(monitor, physics, x1, x2, y1, y2):
    u1, f1 = physics.residual(monitor, x1)
    u2, f2 = physics.residual(monitor, x2)
    supervised = .5 * ((u1-y1).square().mean() + (u2-y2).square().mean())
    residual = .5 * (f1.square().mean() + f2.square().mean())
    direction = torch.relu(-(u2-u1)*(y2-y1)).sum()
    return supervised + residual + .75*direction, (supervised, residual, direction)
