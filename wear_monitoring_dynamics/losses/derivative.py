import torch


def derivative_residual(model, batch, observations):
    """Partial d(z_hat)/d(tau), with sensor/condition held fixed, minus dz/dtau."""
    time = batch['time'][observations].detach().clone().requires_grad_(True)
    estimate = model.estimate(batch, observations, time=time)
    derivative = torch.autograd.grad(estimate.sum(), time, create_graph=True)[0]
    return derivative - model.dynamics.rate(estimate, batch['condition'][observations])


def derivative_loss(model, batch, observations):
    return derivative_residual(model, batch, observations).square().mean()
