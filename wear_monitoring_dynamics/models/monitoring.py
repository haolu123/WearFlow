import torch
from torch import nn

DIMENSIONS = {'sensor': 54, 'condition': 4, 'time': 1}


class MLPMonitor(nn.Module):
    """Predict standardized current VB; no true VB is accepted by this interface."""
    def __init__(self, inputs, hidden_dims):
        super().__init__()
        if not inputs or len(set(inputs)) != len(inputs) or not set(inputs) <= DIMENSIONS.keys():
            raise ValueError('Invalid monitoring input selection')
        self.inputs = list(inputs)
        layers, width = [], sum(DIMENSIONS[k] for k in inputs)
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.SiLU()])
            width = hidden
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, sensor, condition, time):
        values = {'sensor': sensor, 'condition': condition, 'time': time}
        return self.net(torch.cat([values[k] for k in self.inputs], dim=-1))
