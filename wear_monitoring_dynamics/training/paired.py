"""Matched adjacent-pair minibatches for the Zhang transfer study."""
import math
import torch
from ..losses.derivative import derivative_loss
from ..losses.forecasting import forecasting_loss


def make_optimizer(model, cfg):
    train = cfg['training']
    groups = [{'params': model.monitor.parameters(), 'lr': train['warmup_lr'], 'name': 'monitor'}]
    if hasattr(model, 'zhang_physics'):
        groups.append({'params': model.zhang_physics.parameters(), 'lr': train['lr_F'], 'name': 'physics'})
    if model.dynamics is not None and any(p.requires_grad for p in model.dynamics.parameters()):
        raise ValueError('Paired study initially supports frozen ODE only')
    return torch.optim.Adam(groups)


def schedule_after_epoch(optimizer, epoch, train):
    # Original scheduler.step() occurs AFTER an epoch, including its first step.
    i, warmup = epoch - 1, train['lr_warmup_epochs']
    if i < warmup:
        lr = train['warmup_lr'] + (train['learning_rate'] - train['warmup_lr']) * i / max(warmup - 1, 1)
    else:
        lr = train['final_lr'] + .5 * (train['learning_rate'] - train['final_lr']) * (1 + math.cos(math.pi * (i-warmup)/(train['epochs']-warmup)))
    optimizer.param_groups[0]['lr'] = lr


def train_paired_epoch(model, batch, adjacent, forecast_pairs, optimizer, cfg):
    """One pass over adjacent labeled pairs; no pair crosses a case boundary."""
    weights = cfg['loss']
    permutation = torch.randperm(len(adjacent), device=adjacent.device)
    totals, samples = {}, 0
    for indices in permutation.split(cfg['training']['batch_size']):
        pair = adjacent[indices]
        left, right = pair[:, 0], pair[:, 1]
        optimizer.zero_grad(set_to_none=True)
        if hasattr(model, 'zhang_physics'):
            def forward(index):
                xt = model.monitor.pack(batch['sensor'][index], batch['condition'][index], batch['time'][index])
                return model.zhang_physics.residual(model.monitor, xt)
            u1, f1 = forward(left)
            u2, f2 = forward(right)
            components = {'pde': .5 * (f1.square().mean() + f2.square().mean()),
                          'direction': torch.relu(-(u2-u1)*(batch['target'][right]-batch['target'][left])).sum()}
        else:
            u1, u2 = model.estimate(batch, left), model.estimate(batch, right)
            components = {}
        components['monitoring'] = .5 * ((u1-batch['target'][left]).square().mean() + (u2-batch['target'][right]).square().mean())
        total = weights['monitoring_weight'] * components['monitoring']
        if 'pde' in components:
            total = total + cfg['zhang']['alpha']*components['pde'] + cfg['zhang']['beta']*components['direction']
        # Disable dropout only for deterministic physical constraints; the paired
        # data loss above retains the author's dropout. This defines a single
        # differentiable monitor function for ODE consistency/integration.
        model.monitor.eval()
        if weights['derivative_weight']:
            components['derivative'] = derivative_loss(model, batch, torch.cat([left, right]))
            total = total + weights['derivative_weight']*components['derivative']
        if weights['forecasting_weight']:
            selected = forecast_pairs[torch.isin(forecast_pairs[:, 0], left)]
            components['forecasting'] = forecasting_loss(model, batch, selected)
            total = total + weights['forecasting_weight']*components['forecasting']
        model.monitor.train()
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite paired loss')
        total.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError('Nonfinite paired gradient')
        optimizer.step()
        for key, value in {'loss': total, **{f'loss_{k}': v for k, v in components.items()}}.items():
            totals[key] = totals.get(key, 0.) + float(value.detach())*len(pair)
        samples += len(pair)
    return {key: value/samples for key, value in totals.items()}
