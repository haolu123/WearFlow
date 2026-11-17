from pathlib import Path
import torch
from ..config import digest


def save_checkpoint(path, model, epoch, cfg, scalers, split, signature):
    # CPU tensors make the saved state portable; no optimizer resume is implied.
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    temp = Path(path).with_suffix('.tmp')
    torch.save({'model': state, 'epoch': epoch, 'config': cfg, 'scalers': scalers,
                'split': split, 'data_signature': signature}, temp)
    temp.replace(path)


def load_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    return checkpoint


def load_pretrained(path, model, scalers, split, signature, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if state['data_signature'] != signature or digest(state['scalers']) != digest(scalers):
        raise ValueError('Pretrained ODE data/scalers do not match this training split')
    if state['split']['train'] != split['train']:
        raise ValueError('Pretrained ODE has different training cases; possible validation leakage')
    dynamics_cfg = state['config']['dynamics']
    if dynamics_cfg.get('transition', 'ode') == 'ode':
        activation = 'softplus' if dynamics_cfg['nonnegative_rate'] else 'identity'
        if not hasattr(model, 'rk4_steps') or model.rk4_steps != dynamics_cfg['rk4_steps'] or model.rate.activation != activation:
            raise ValueError('Pretrained ODE solver or rate constraint differs from requested model')
    elif not hasattr(model, 'nonnegative') or model.nonnegative != (dynamics_cfg['transition'] == 'direct_positive'):
        raise ValueError('Pretrained direct-transition constraint differs from requested model')
    model.load_state_dict(state['model'])
    return state
