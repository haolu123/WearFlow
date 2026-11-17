"""Read-only post-update diagnostics, including explicitly requested outer-test curves."""
import csv
import gzip
import numpy as np
import torch
from ..data.datasets import observations, pair_tensors


class EpochRecorder:
    def __init__(self, directory, table, batch, scalers, split, horizons, pretrain):
        self.table, self.batch, self.scalers, self.pretrain = table, batch, scalers, pretrain
        cases = {'train': split['train'], 'validation': split['validation'],
                 'test_diagnostic': split['outer_test_reserved']}
        self.groups = {name: (observations(table, values), pair_tensors(table, values, horizons, batch['time'].device))
                       for name, values in cases.items() if values}
        self.stream = gzip.open(directory / 'epoch_predictions.csv.gz', 'wt', encoding='utf-8', newline='')
        self.writer = csv.writer(self.stream)
        self.writer.writerow(['epoch', 'split', 'task', 'case', 'origin_run', 'target_run', 'horizon', 'target_vb_mm', 'predicted_vb_mm'])

    def close(self):
        self.stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _record(self, epoch, split, task, idx, pred, target_idx=None, horizons=None):
        table = self.table
        target_idx = idx if target_idx is None else target_idx
        target = table['VB'][target_idx]
        prediction = pred.detach().cpu().numpy().reshape(-1)*self.scalers['wear'][1]+self.scalers['wear'][0]
        if not np.isfinite(prediction).all():
            raise FloatingPointError(f'Nonfinite {split}/{task} epoch predictions')
        case = table['case'][idx]
        error = prediction-target
        values = {'mae': float(abs(error).mean()), 'rmse': float(np.sqrt(np.mean(error**2))),
                  'case_macro_mae': float(np.mean([abs(error[case == c]).mean() for c in np.unique(case)])),
                  'bias': float(error.mean()), 'n': len(idx)}
        self.writer.writerows(zip([epoch]*len(idx), [split]*len(idx), [task]*len(idx), case.astype(int),
                                  table['run'][idx].astype(int), table['run'][target_idx].astype(int),
                                  np.zeros(len(idx), int) if horizons is None else horizons, target, prediction))
        if horizons is not None:
            for h in np.unique(horizons):
                values[f'h{h}_mae'] = float(abs(error[horizons == h]).mean())
        return {f'{split}_{task}_{key}': value for key, value in values.items()}

    def record(self, model, epoch):
        result = {}
        model.eval()
        with torch.no_grad():
            for name, (idx, pairs) in self.groups.items():
                if not self.pretrain:
                    result.update(self._record(epoch, name, 'monitoring', idx, model.estimate(self.batch, idx)))
                if self.pretrain or model.dynamics is not None:
                    raw = pairs.cpu().numpy()
                    if not len(raw):
                        continue
                    if self.pretrain:
                        a, b = pairs[:, 0], pairs[:, 1]
                        prediction = model.propagate(self.batch['target'][a], self.batch['condition'][a], self.batch['time'][b]-self.batch['time'][a])
                    else:
                        prediction = model.forecast(self.batch, pairs)
                    result.update(self._record(epoch, name, 'forecasting', raw[:, 0], prediction, raw[:, 1], raw[:, 2]))
        self.stream.flush()
        return result
