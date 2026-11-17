"""Interval-conditioned direct transitions: no ODE evaluation or integration."""
import torch
from torch import nn
from torch.nn import functional as F


class DirectTransition(nn.Module):
    def __init__(self, hidden_dim=32, nonnegative=True):
        super().__init__()
        self.nonnegative = nonnegative
        self.net = nn.Sequential(nn.Linear(6, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def propagate(self, wear, conditions, delta_time):
        # The interval is an input: one forward pass predicts its average rate.
        # Multiplication by dt enforces identity at dt=0, but is not integration.
        rate = self.net(torch.cat([wear, conditions, delta_time], dim=-1))
        if self.nonnegative:
            rate = F.softplus(rate)
        return wear + delta_time * rate


