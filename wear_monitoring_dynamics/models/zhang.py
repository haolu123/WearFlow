"""NASA adaptation of zhangpengfei247/Tool-wear-monitoring-boring-PINN.

Architecture and contracted derivative convention follow notebook 2.
See docs/zhang_reproduction.md for provenance and deliberate data adaptations.
"""
import torch
from torch import nn
from .monitoring import DIMENSIONS


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class ZhangMLP(nn.Module):
    def __init__(self, input_dim, output_dim, layers_num, hidden_dim, dropout=.2):
        super().__init__()
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.extend([nn.Linear(input_dim, hidden_dim), Sin()])
            elif i == layers_num - 1:
                layers.append(nn.Linear(hidden_dim, output_dim))
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), Sin(), nn.Dropout(dropout)])
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)


class ZhangPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(.2), nn.Linear(32, 30), Sin(), nn.Linear(30, 1))

    def forward(self, x):
        return self.net(x)


class ZhangMonitor(nn.Module):
    def __init__(self, inputs, hidden_dims=None):
        super().__init__()
        if not inputs or inputs[-1] != 'time' or len(set(inputs)) != len(inputs):
            raise ValueError('Zhang monitor requires time as the last input block')
        self.inputs = list(inputs)
        self.input_dim = sum(DIMENSIONS[k] for k in inputs)
        self.encoder = ZhangMLP(self.input_dim, 32, 3, 32)
        self.predictor = ZhangPredictor()
        # The author reinitializes all solution_u linear weights and biases.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def pack(self, sensor, condition, time):
        values = {'sensor': sensor, 'condition': condition, 'time': time}
        return torch.cat([values[k] for k in self.inputs], dim=-1)

    def forward_features(self, xt):
        return self.predictor(self.encoder.net(xt))

    def forward(self, sensor, condition, time):
        return self.forward_features(self.pack(sensor, condition, time))


def author_derivatives(solution, xt):
    """Return u and F inputs, preserving the author's vector-Jacobian sums.

    u_xx is H_x @ ones, NOT the Hessian diagonal. u_xt is a scalar sum
    across feature-time mixed derivatives, NOT a vector of mixed partials.
    """
    xt = xt.detach().clone().requires_grad_(True)
    x, t = xt[:, :-1], xt[:, -1:]
    u = solution(torch.cat([x, t], dim=1))
    def derivative(y, z):
        return torch.autograd.grad(y.sum(), z, create_graph=True, retain_graph=True)[0]
    ut, ux = derivative(u, t), derivative(u, x)
    utt, uxt, uxx = derivative(ut, t), derivative(ux, t), derivative(ux, x)
    return u, ut, torch.cat([xt, u, ux, ut, utt, uxx, uxt], dim=1)


class ZhangPhysics(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = ZhangMLP(3 * input_dim + 2, 1, 4, 50)

    def residual(self, monitor, xt):
        u, ut, features = author_derivatives(monitor.forward_features, xt)
        return u, ut - self.network.net(features)
