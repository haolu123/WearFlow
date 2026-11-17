import copy
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from wear_monitoring_dynamics.config import load_config, PACKAGE
from wear_monitoring_dynamics.study import build_tasks
from wear_monitoring_dynamics.scripts import run_fixed_conditioning as runner


class FixedConditioningTests(unittest.TestCase):
    def test_fixed_architectures_and_complete_coverage(self):
        tasks = build_tasks(load_config(PACKAGE/'configs/time_conditioning.yaml'), runner.specification())
        self.assertEqual(len(tasks), 256)
        monitor_tasks = [t for t in tasks if t['type'] == 'monitor']
        self.assertEqual(len(monitor_tasks), 240)
        for name, _, architecture, inputs in runner.MODELS:
            for suffix in ('', '_forecast'):
                group = [t for t in monitor_tasks if t['config']['experiment']['name'] == name+suffix]
                self.assertEqual(len(group), 24)
                self.assertEqual({t['config']['monitor']['architecture'] for t in group}, {architecture})
                for task in group:
                    self.assertEqual(task['config']['monitor']['inputs'], inputs)
                    self.assertNotIn('fold_overrides', task['config'])

    def test_reuse_only_ignores_bookkeeping(self):
        cfg = load_config()
        renamed = copy.deepcopy(cfg)
        renamed['study_id'] = 'new_study'
        renamed['experiment']['name'] = 'new_label'
        renamed['output_root'] = 'elsewhere'
        self.assertEqual(runner.normalized_config(cfg), runner.normalized_config(renamed))
        for section, key, value in [('training', 'epochs', 501), ('training', 'selection_epoch', 20),
                                    ('monitor', 'architecture', 'film'), ('experiment', 'fold', 'fold_b'),
                                    ('loss', 'forecasting_weight', 1.)]:
            changed = copy.deepcopy(cfg)
            changed[section][key] = value
            self.assertNotEqual(runner.normalized_config(cfg), runner.normalized_config(changed))

    def test_parent_checkpoint_contents_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            first, second = Path(temp)/'first.pt', Path(temp)/'second.pt'
            first.write_bytes(b'same checkpoint')
            second.write_bytes(first.read_bytes())
            a, b = load_config(), load_config()
            a['dynamics']['checkpoint'], b['dynamics']['checkpoint'] = str(first), str(second)
            self.assertEqual(runner.normalized_config(a), runner.normalized_config(b))
            second.write_bytes(b'different checkpoint')
            self.assertNotEqual(runner.normalized_config(a), runner.normalized_config(b))

    def test_changed_training_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            package = project/'wear_monitoring_dynamics'
            package.mkdir()
            model = package/'model.py'
            model.write_text('value = 1\n')
            dynamics = project/'release/CondWear-ODE/models/vb_sensor_condition_ode.py'
            dynamics.parent.mkdir(parents=True)
            dynamics.write_text('value = 2\n')
            archive = project/'snapshot.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                for path in (model, dynamics):
                    z.write(path, path.relative_to(project).as_posix())
            with patch.object(runner, 'PROJECT', project), patch.object(runner, 'PACKAGE', package):
                self.assertEqual(len(runner.verify_source(archive)), 2)
                model.write_text('value = 3\n')
                with self.assertRaises(ValueError):
                    runner.verify_source(archive)
