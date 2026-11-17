"""Small, continuously time-conditioned monitors for six sensor-statistic tokens.

All models consume current observations only. Time remains differentiable, so
the same monitors can be used with the existing physical-state ODE losses.
"""
import math
import torch
from torch import nn


class ContinuousTimeEncoding(nn.Module):
    """Raw normalized time plus three fixed, low-frequency Fourier pairs."""
    output_dim = 7

    def __init__(self):
        super().__init__()
        self.register_buffer('frequencies', torch.tensor([0.125, 0.25, 0.5]) * math.pi)

    def forward(self, time):
        phase = time * self.frequencies
        return torch.cat((time, phase.sin(), phase.cos()), dim=-1)


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, output_dim))


class EncodedMLPMonitor(nn.Module):
    """Isolate continuous time encoding from conditioning/fusion changes."""
    def __init__(self, inputs, hidden_dims):
        super().__init__()
        if inputs != ['sensor', 'condition', 'time']:
            raise ValueError('Encoded monitor requires ordered sensor, condition, time inputs')
        self.inputs = list(inputs)
        self.time_encoding = ContinuousTimeEncoding()
        self.net = nn.Sequential(nn.Linear(65, 60), nn.SiLU(), nn.Linear(60, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, sensor, condition, time):
        return self.net(torch.cat((sensor, condition, self.time_encoding(time)), dim=-1))


class ConditionedMonitor(nn.Module):
    """Shared continuous time/condition encoder; subclasses define sensor fusion."""
    def __init__(self, inputs, hidden_dims):
        super().__init__()
        if inputs != ['sensor', 'condition', 'time']:
            raise ValueError('Conditioned monitor requires ordered sensor, condition, time inputs')
        self.inputs = list(inputs)
        self.time_encoding = ContinuousTimeEncoding()
        self.context = mlp(11, 24, 24)

    def encode_context(self, condition, time):
        return self.context(torch.cat((condition, self.time_encoding(time)), dim=-1))


class FiLMMonitor(ConditionedMonitor):
    """Time and condition modulate normalized sensor features by scale and shift."""
    def __init__(self, inputs, hidden_dims):
        super().__init__(inputs, hidden_dims)
        self.sensor_encoder = mlp(54, 32, 24)
        self.norm = nn.LayerNorm(24)
        self.modulation = nn.Linear(24, 48)
        self.head = mlp(48, 24, 1)

    def forward(self, sensor, condition, time):
        context = self.encode_context(condition, time)
        gamma, beta = self.modulation(context).chunk(2, dim=-1)
        features = (1 + gamma) * self.norm(self.sensor_encoder(sensor)) + beta
        return self.head(torch.cat((features, context), dim=-1))


class SensorAttentionMonitor(ConditionedMonitor):
    """Three-head context query over six channel-specific sensor tokens.

    Explicit matmul/softmax supports higher-order autograd for derivative losses.
    Token order must match data.features.SENSORS (nine contiguous statistics each).
    Attention weights describe model routing, not causal sensor importance.
    """
    def __init__(self, inputs, hidden_dims):
        super().__init__(inputs, hidden_dims)
        self.sensor_encoder = mlp(9, 24, 24)
        self.channel_embedding = nn.Parameter(torch.empty(6, 24))
        nn.init.normal_(self.channel_embedding, std=0.02)
        self.norm = nn.LayerNorm(24)
        self.query = nn.Linear(24, 24)
        self.key = nn.Linear(24, 24)
        self.value = nn.Linear(24, 24)
        self.head = mlp(48, 40, 1)

    def attend(self, sensor, context):
        batch_size = sensor.shape[0]
        tokens = self.norm(self.sensor_encoder(sensor.reshape(batch_size, 6, 9)) + self.channel_embedding)
        query = self.query(context).reshape(batch_size, 3, 1, 8)
        key = self.key(tokens).reshape(batch_size, 6, 3, 8).transpose(1, 2)
        value = self.value(tokens).reshape(batch_size, 6, 3, 8).transpose(1, 2)
        weights = ((query @ key.transpose(-1, -2)) / math.sqrt(8)).softmax(dim=-1)
        features = (weights @ value).reshape(batch_size, 24)
        return features, weights.squeeze(-2)

    def attention_weights(self, sensor, condition, time):
        return self.attend(sensor, self.encode_context(condition, time))[1]

    def forward(self, sensor, condition, time):
        context = self.encode_context(condition, time)
        features, _ = self.attend(sensor, context)
        return self.head(torch.cat((features, context), dim=-1))
