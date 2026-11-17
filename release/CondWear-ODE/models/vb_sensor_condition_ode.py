from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RateNet(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, activation: str = "softplus"):
        super().__init__()
        self.activation = str(activation)
        self.net = nn.Sequential(
            nn.Linear(1 + feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, wear: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        raw = self.net(torch.cat([wear, features], dim=-1))
        if self.activation == "softplus":
            return F.softplus(raw)
        if self.activation == "identity":
            return raw
        raise ValueError(f"Unknown rate activation: {self.activation}")


class VBSensorConditionMonotonicODE(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 32, rk4_steps: int = 20, rate_activation: str = "softplus"):
        super().__init__()
        self.rk4_steps = int(rk4_steps)
        self.feature_dim = int(feature_dim)
        self.rate = RateNet(feature_dim=self.feature_dim, hidden_dim=hidden_dim, activation=rate_activation)
        self.logvar_head = nn.Sequential(
            nn.Linear(1 + self.feature_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def rhs(self, wear: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.rate(wear, features)

    def integrate_rk4(self, w0: torch.Tensor, features: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        steps = max(1, self.rk4_steps)
        dt = delta_t / float(steps)
        wear = w0
        for _ in range(steps):
            k1 = self.rhs(wear, features)
            k2 = self.rhs(wear + 0.5 * dt * k1, features)
            k3 = self.rhs(wear + 0.5 * dt * k2, features)
            k4 = self.rhs(wear + dt * k3, features)
            wear = wear + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return wear

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        w0 = batch['w0']
        features = batch['features']
        delta_t = batch['delta_t']
        mu = self.integrate_rk4(w0, features, delta_t)
        raw_logvar = self.logvar_head(torch.cat([mu, features, delta_t], dim=-1))
        logvar = torch.clamp(raw_logvar, min=-8.0, max=4.0)
        return {'mu': mu, 'logvar': logvar}
