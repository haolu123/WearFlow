"""Soft feature gates controlled exclusively by current condition/time."""

import torch

from torch import nn

from .feature_mlp import FeatureMLP



class GatedFeatureMLP(FeatureMLP):

    def __init__(self, cfg):

        # Construct the predictor first: identical seed -> identical predictor weights.

        super().__init__(cfg)

        self.gate_kind = cfg['gate']['kind']

        d = len(cfg['selected_indices'])

        if self.gate_kind == 'static':

            self.gate_logits = nn.Parameter(torch.zeros(d))

        elif self.gate_kind in ('condition', 'time_condition'):

            width = 4 if self.gate_kind == 'condition' else 5

            self.gate_net = nn.Sequential(nn.Linear(width, 16), nn.SiLU(), nn.Linear(16, d))

            nn.init.zeros_(self.gate_net[-1].weight)

            nn.init.zeros_(self.gate_net[-1].bias)

        else:

            raise ValueError('Unknown gate kind')



    def gates(self, batch, indices, time=None):

        if self.gate_kind == 'static':

            return self.gate_logits.sigmoid().expand(len(indices), -1)

        x = batch['condition'][indices]

        if self.gate_kind == 'time_condition':

            t = batch['time'][indices] if time is None else time

            x = torch.cat([x, t], dim=-1)

        return self.gate_net(x).sigmoid()



    def estimate(self, batch, indices, time=None):

        t = batch['time'][indices] if time is None else time

        sensor = batch['sensor'][indices][:, self.selected_indices]

        sensor = sensor * self.gates(batch, indices, time)

        pred = self.net(torch.cat([sensor, batch['condition'][indices], t], dim=-1))

        if 'fallback_vb_z' in batch:

            pred = torch.where(batch['sensor_valid'][indices,None], pred, batch['fallback_vb_z'][indices])

        return pred



    def gate_statistics(self, batch, indices):

        with torch.no_grad():

            g = self.gates(batch, indices)

            w = self.net[0].weight[:, :len(self.selected_indices)]

            return {'mean': float(g.mean()), 'sum': float(g.sum(-1).mean()),

                    'fraction_lt_01': float((g<.1).float().mean()),

                    'fraction_gt_09': float((g>.9).float().mean()),

                    'input_weight_norm': float(w.norm()),

                    'effective_input_weight_norm': float((w*g.mean(0)[None,:]).norm())}

