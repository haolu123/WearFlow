import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

from wear_monitoring_dynamics.config import PACKAGE, load_config
from wear_monitoring_dynamics.data.preprocessing import fit_scalers, forecasting_pairs
from wear_monitoring_dynamics.data.splits import make_split
from wear_monitoring_dynamics.models.registry import build_model
from wear_monitoring_dynamics.models.dynamics import WearDynamics
from wear_monitoring_dynamics.losses.derivative import derivative_loss, derivative_residual
from wear_monitoring_dynamics.losses.forecasting import forecasting_loss
from wear_monitoring_dynamics.training.trainer import validate_config, run_training
from wear_monitoring_dynamics.training.checkpoint import save_checkpoint, load_pretrained
from wear_monitoring_dynamics.study import build_tasks, variants, select_epoch
from wear_monitoring_dynamics.evaluation.bootstrap import paired_case_bootstrap


def synthetic_table():
    return {'case': np.repeat([1, 2], 5), 'run': np.tile(np.arange(5), 2),
            'time': np.tile(np.arange(5)*2., 2), 'VB': np.tile(np.arange(5)*.1, 2),
            'DOC': np.ones(10), 'feed': np.ones(10)*.5, 'material': np.ones(10),
            'sensor': np.arange(540).reshape(10, 54).astype(float)}


class ScientificTests(unittest.TestCase):
    def test_raw_sensor_quarantine_is_label_independent(self):
        from wear_monitoring_dynamics.data.records import load_table
        from wear_monitoring_dynamics.data.features import SENSORS
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            records = []
            for run, magnitude in [(1, 1.), (2, 1e30)]:
                item = dict(case=1, run=run, time=float(run), VB=.1, DOC=1.5, feed=.5, material=1)
                item.update({key: np.ones(10)*magnitude for key in SENSORS})
                records.append(item)
            np.save(path/'data.npy', np.array(records, dtype=object))
            table, _ = load_table(path/'data.npy', path/'cache')
            self.assertEqual(table['run'].tolist(), [1.])
            records[1]['VB'] = 9999.
            np.save(path/'data.npy', np.array(records, dtype=object))
            other, _ = load_table(path/'data.npy', path/'cache')
            self.assertEqual(table['run'].tolist(), other['run'].tolist())

    def test_scalers_ignore_heldout_case(self):
        table = synthetic_table()
        expected = fit_scalers(table, [1])
        changed = copy.deepcopy(table)
        for key in ('VB', 'time', 'DOC', 'feed', 'sensor'):
            changed[key][changed['case'] == 2] = 99999.
        self.assertEqual(expected, fit_scalers(changed, [1]))
        self.assertEqual(expected['time_scale'], 6.)

    def test_missing_labels_and_pair_horizons(self):
        table = synthetic_table()
        table['VB'][1] = np.nan
        pairs = forecasting_pairs(table, [1], [1, 2, 3])
        self.assertEqual(len(pairs), 6)
        self.assertEqual(pairs[0].tolist(), [0, 2, 1])
        self.assertTrue(np.all(table['case'][pairs[:, 0]] == table['case'][pairs[:, 1]]))

    def test_outer_test_never_used_for_smoke(self):
        split = make_split('fold_a', 'smoke', 8)
        self.assertEqual(split['validation'], [8])
        self.assertFalse(split['test'])
        self.assertFalse(set(split['train']) & set(split['outer_test_reserved']))

    def test_frozen_dynamics_backpropagates_to_monitor(self):
        cfg = load_config()
        cfg['dynamics']['mode'] = 'frozen'
        model = build_model(cfg)
        model.dynamics.requires_grad_(False)
        batch = {'sensor': torch.randn(4, 54), 'condition': torch.randn(4, 4),
                 'time': torch.arange(4.).reshape(-1, 1), 'target': torch.randn(4, 1)}
        pairs = torch.tensor([[0, 1, 1], [1, 3, 2]])
        forecasting_loss(model, batch, pairs).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.monitor.parameters()), 0)
        self.assertTrue(all(p.grad is None for p in model.dynamics.parameters()))

    def test_derivative_loss_has_monitor_and_rate_gradients(self):
        cfg = load_config()
        cfg['dynamics']['mode'] = 'joint_scratch'
        model = build_model(cfg)
        batch = {'sensor': torch.randn(4, 54), 'condition': torch.randn(4, 4), 'time': torch.randn(4, 1)}
        derivative_loss(model, batch, np.arange(4)).backward()
        for module in (model.monitor, model.dynamics):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None), 0)

    def test_forecast_does_not_read_future_sensor_or_true_origin_vb(self):
        cfg = load_config()
        cfg['dynamics']['mode'] = 'joint_scratch'
        model = build_model(cfg)
        batch = {'sensor': torch.randn(4, 54), 'condition': torch.ones(4, 4),
                 'time': torch.arange(4.).reshape(-1, 1), 'target': torch.randn(4, 1)}
        pairs = torch.tensor([[0, 3, 3]])
        before = model.forecast(batch, pairs)
        batch['sensor'][1:] = 99999.
        batch['target'][:] = 99999.
        torch.testing.assert_close(before, model.forecast(batch, pairs), rtol=0, atol=0)

    def test_analytic_derivative_and_physical_units(self):
        class ConstantRate:
            def rate(self, wear, conditions):
                return wear*0 + 3.
        class Analytic:
            dynamics = ConstantRate()
            def estimate(self, batch, idx, time=None):
                return 3.*time + batch['sensor'][idx, :1]
        batch = {'time': torch.randn(4, 1), 'sensor': torch.randn(4, 54), 'condition': torch.zeros(4, 4)}
        residual = derivative_residual(Analytic(), batch, np.arange(4))
        torch.testing.assert_close(residual, torch.zeros_like(residual))
        # dz/dtau=3, wear std=.2 mm, time scale=10 s -> .06 mm/s.
        self.assertAlmostEqual(3.*.2/10., .06)

    def test_rk4_constant_rate_and_zero_interval(self):
        dynamics = WearDynamics()
        for parameter in dynamics.parameters():
            torch.nn.init.zeros_(parameter)
        wear, condition = torch.tensor([[.3]]), torch.zeros(1, 4)
        torch.testing.assert_close(dynamics.propagate(wear, condition, torch.zeros(1, 1)), wear)
        torch.testing.assert_close(dynamics.propagate(wear, condition, torch.ones(1, 1)), wear+np.log(2.), atol=1e-6, rtol=1e-6)

    def test_pretraining_split_mismatch_rejected(self):
        cfg = load_config()
        model = WearDynamics()
        split = make_split('fold_a', 'inner', 8)
        scalers = {'wear': [0., 1.]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.pt'
            save_checkpoint(path, model, 1, cfg, scalers, split, 'data',)
            wrong = copy.deepcopy(split)
            wrong['train'] = wrong['train'] + [8]
            with self.assertRaisesRegex(ValueError, 'training cases'):
                load_pretrained(path, model, scalers, wrong, 'data', 'cpu')

    def test_invalid_derivative_configuration_rejected(self):
        cfg = load_config()
        cfg['monitor']['inputs'] = ['sensor']
        cfg['dynamics']['mode'] = 'joint_scratch'
        cfg['loss']['derivative_weight'] = .1
        with self.assertRaisesRegex(ValueError, 'Time input'):
            validate_config(cfg)

    def test_plan_counts_and_dependencies(self):
        base = load_config()
        study = yaml.safe_load((PACKAGE / 'configs/studies/core.yaml').read_text())
        self.assertEqual(len(variants(base, study)), 24)
        tasks = build_tasks(base, study)
        self.assertEqual(len(tasks), 592)
        seen = set()
        for task in tasks:
            self.assertTrue(set(task['dependencies']) <= seen)
            self.assertNotIn(task['id'], seen)
            seen.add(task['id'])
        self.assertEqual(sum(t['type'] == 'pretrain' for t in tasks), 16)
        self.assertEqual(sum(t['type'] == 'monitor' and t['config']['experiment']['role'] == 'final' for t in tasks), 240)

    def test_epoch_selection_uses_mean_across_seven_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(7):
                path = Path(tmp) / str(i)
                path.mkdir()
                pd.DataFrame({'epoch': [1, 2, 3], 'validation_case_macro_mae': [3., 1., 2.]}).to_csv(path/'history.csv', index=False)
                paths.append(path)
            epoch, curve = select_epoch(paths)
            self.assertEqual(epoch, 2)

    def test_case_bootstrap_direction(self):
        a = pd.Series([.1, .2, .3], index=[1, 2, 3])
        result = paired_case_bootstrap(a, a+.05, repeats=100)
        self.assertAlmostEqual(result['delta_mae'], -.05)
        self.assertLess(result['ci_high'], 0)

    def test_final_training_and_summary_on_synthetic_data_only(self):
        import json
        from wear_monitoring_dynamics.data.features import SENSORS
        from wear_monitoring_dynamics.scripts.summarize_study import summarize
        records = []
        for case in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 16]:
            for run in range(1, 6):
                record = dict(case=case, run=run, time=float(run*2), VB=.02*run,
                              DOC=.75*(1+case%2), feed=.25, material=1+case%2)
                record.update({name: np.arange(12, dtype=float)*.01+run*.001 for name in SENSORS})
                records.append(record)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root/'data.npy', np.array(records, dtype=object))
            cfg = load_config()
            cfg['data'].update(path=str(root/'data.npy'), cache=str(root/'cache'))
            cfg['output_root'] = str(root/'output')
            cfg['training'].update(device='cpu', epochs=2, warmup_epochs=1,
                                   record_all_epoch_splits=True, selection_epoch=1, save_final_epoch_checkpoints=True)
            cfg['experiment'].update(role='final', holdout=None)
            directory = run_training(cfg)
            metrics = json.loads((directory/'metrics.json').read_text())
            self.assertEqual(metrics['evaluation_split'], 'test')
            self.assertEqual(metrics['selected_epoch'], 1)
            self.assertTrue(metrics['reload_verified'])
            curve = pd.read_csv(directory/'history.csv')
            self.assertEqual(len(curve), 2)
            self.assertTrue(curve['test_diagnostic_monitoring_case_macro_mae'].notna().all())
            predictions = pd.read_csv(directory/'epoch_predictions.csv.gz')
            self.assertEqual(set(predictions['epoch']), {1, 2})
            self.assertEqual(set(predictions['split']), {'train', 'test_diagnostic'})
            self.assertTrue((directory/'epochs/epoch_0002.pt').exists())
            self.assertEqual(directory, run_training(cfg))
            study_root = root/'output'/cfg['study_id']
            pd.DataFrame([dict(status='completed', role='final', type='monitor', experiment=cfg['experiment']['name'],
                               fold='fold_a', seed=0, run_dir=str(directory))]).to_csv(study_root/'study_manifest.csv', index=False)
            summarize(study_root)
            self.assertTrue((study_root/'summary/monitoring_leaderboard.csv').exists())


if __name__ == '__main__':
    unittest.main()
