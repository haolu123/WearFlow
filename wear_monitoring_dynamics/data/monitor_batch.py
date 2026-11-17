"""Build causal windows and a frozen, deployment-safe condition/time fallback."""
import numpy as np
import torch
from .datasets import tensors
from .history import add_history


def make_monitor_batch(cfg, table, scalers, device):
    batch=tensors(table,scalers,device)
    length=cfg['monitor'].get('history_length')
    if length:add_history(batch,table,length,scalers['time_scale'],device)
    checkpoint=cfg['monitor'].get('fallback_checkpoint')
    if checkpoint:
        from ..models.registry import build_model
        state=torch.load(checkpoint,map_location=device,weights_only=True)
        assert state['scalers']['train_cases']==scalers['train_cases']
        assert state['config']['monitor']['inputs']==['condition','time']
        model=build_model(state['config']).to(device);model.load_state_dict(state['model']);model.eval()
        original=tensors(table,state['scalers'],device)
        with torch.no_grad():
            y=model.estimate(original,np.arange(len(table['case'])))
            raw=y*state['scalers']['wear'][1]+state['scalers']['wear'][0]
            batch['fallback_vb_z']=(raw-scalers['wear'][0])/scalers['wear'][1]
    if 'sensor_valid' in table:
        batch['sensor_valid']=torch.tensor(table['sensor_valid'],device=device)
        invalid=~torch.tensor(table['sensor_valid'],device=device)
        batch['sensor'][invalid]=0.  # padding cannot carry invalid values through 0*NaN
    return batch
