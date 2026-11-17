from torch import nn


class CoupledMonitor(nn.Module):
    def __init__(self, monitor, dynamics=None):
        super().__init__()
        self.monitor = monitor
        self.dynamics = dynamics

    def estimate(self, batch, indices, time=None):
        return self.monitor(batch['sensor'][indices], batch['condition'][indices],
                            batch['time'][indices] if time is None else time)

    def forecast(self, batch, pairs):
        origin, target = pairs[:, 0], pairs[:, 1]
        wear = self.estimate(batch, origin)
        return self.dynamics.propagate(wear, batch['condition'][origin],
                                       batch['time'][target] - batch['time'][origin])
