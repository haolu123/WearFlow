"""Reuse the release rate network and RK4 implementation, without its variance head."""
import importlib.util
from torch import nn
from ..config import PROJECT

_path = PROJECT / 'release/CondWear-ODE/models/vb_sensor_condition_ode.py'
_spec = importlib.util.spec_from_file_location('condwear_release_dynamics', _path)
_release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_release)


class WearDynamics(nn.Module):
    def __init__(self, hidden_dim=32, rk4_steps=20, nonnegative_rate=True):
        super().__init__()
        self.rk4_steps = int(rk4_steps)
        if self.rk4_steps < 1:
            raise ValueError('RK4 steps must be positive')
        self.rate = _release.RateNet(4, hidden_dim, 'softplus' if nonnegative_rate else 'identity')

    def rhs(self, wear, conditions):
        return self.rate(wear, conditions)

    def propagate(self, wear, conditions, delta_time):
        return _release.VBSensorConditionMonotonicODE.integrate_rk4(self, wear, conditions, delta_time)
