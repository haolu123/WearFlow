"""Independent upstream parity and analytic checks for the transfer adapter."""
import ast
import json
import unittest
from pathlib import Path
import torch
from torch import nn
from wear_monitoring_dynamics.config import PROJECT, load_config
from wear_monitoring_dynamics.models.zhang import ZhangMonitor, ZhangPhysics, author_derivatives
from wear_monitoring_dynamics.models.registry import build_model
from wear_monitoring_dynamics.losses.forecasting import forecasting_loss
from wear_monitoring_dynamics.study import build_tasks
import yaml


def reference_classes():
    notebook = next((PROJECT/'external/zhang_boring_pinn').glob('2*.ipynb'))
    content = json.loads(notebook.read_text(encoding='utf-8'))
    source = '\n\n'.join(''.join(c['source']) for c in content['cells'] if c['cell_type'] == 'code')
    nodes = ast.parse(source).body
    chosen = [n for n in nodes if isinstance(n, ast.ClassDef) and n.name in ('Sin', 'MLP', 'Predictor', 'Solution_u')]
    # Execute only reviewed architecture definitions, never notebook data/IO cells.
    namespace = {'torch': torch, 'nn': nn}
    exec(compile(ast.Module(body=chosen, type_ignores=[]), str(notebook), 'exec'), namespace)
    pinn = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == 'PINN')
    forward = next(n for n in pinn.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(notebook), 'exec'), {'grad': torch.autograd.grad, 'torch': torch}, namespace)
    return namespace


class ZhangTests(unittest.TestCase):
    def test_upstream_forward_residual_and_parameter_gradient_parity(self):
        ref = reference_classes()
        monitor = ZhangMonitor(['condition', 'time']).double()
        # Preserve the author's entire network; adapt only the input linear size.
        original = ref['Solution_u']().double()
        original.encoder.net[0] = nn.Linear(5, 32).double()
        original.load_state_dict(monitor.state_dict())
        physics = ZhangPhysics(5).double()
        original_f = ref['MLP'](input_dim=17, output_dim=1, layers_num=4, hidden_dim=50, droupout=.2).double()
        original_f.load_state_dict(physics.network.state_dict())
        holder = nn.Module()
        holder.solution_u, holder.dynamical_F = original, original_f
        xt = torch.randn(4, 5, dtype=torch.float64)
        for training in (False, True):
            monitor.train(training); physics.train(training); holder.train(training)
            monitor.zero_grad(); physics.zero_grad(); holder.zero_grad()
            torch.manual_seed(91)
            a, fa = physics.residual(monitor, xt)
            (a.square().mean()+fa.square().mean()).backward()
            torch.manual_seed(91)
            b, fb = ref['forward'](holder, xt.clone())
            (b.square().mean()+fb.square().mean()).backward()
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(fa, fb, rtol=1e-12, atol=1e-12)
            for p, q in zip(monitor.parameters(), original.parameters()):
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-10)
            for p, q in zip(physics.parameters(), original_f.parameters()):
                torch.testing.assert_close(p.grad, q.grad, rtol=1e-10, atol=1e-10)

    def test_contracted_derivatives_are_not_hessian_diagonal(self):
        def f(xt):
            x, y, t = xt.split(1, dim=1)
            return x*x + 3*x*y + 2*y*y + x*t + 2*y*t + t*t
        u, ut, features = author_derivatives(f, torch.tensor([[1., 2., 3.]], dtype=torch.float64))
        # [xt(3), u(1), ux(2), ut(1), utt(1), uxx(2), uxt(1)]
        torch.testing.assert_close(features[:, 7:], torch.tensor([[2., 5., 7., 3.]], dtype=torch.float64))

    def test_frozen_ode_forecast_still_trains_zhang_monitor(self):
        cfg = load_config(PROJECT/'wear_monitoring_dynamics/configs/zhang_transfer.yaml')
        cfg['dynamics']['mode'] = 'frozen'
        model = build_model(cfg)
        model.dynamics.requires_grad_(False)
        batch = {'sensor': torch.randn(4,54), 'condition': torch.randn(4,4),
                 'time': torch.arange(4.).reshape(-1,1), 'target': torch.randn(4,1)}
        loss = forecasting_loss(model, batch, torch.tensor([[0,1,1],[1,2,1]]))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.monitor.parameters()))
        self.assertTrue(all(p.grad is None for p in model.dynamics.parameters()))

    def test_study_has_twelve_variants_and_304_tasks(self):
        cfg = load_config(PROJECT/'wear_monitoring_dynamics/configs/zhang_transfer.yaml')
        study = yaml.safe_load((PROJECT/'wear_monitoring_dynamics/configs/studies/zhang_transfer_20260922.yaml').read_text())
        tasks = build_tasks(cfg, study)
        self.assertEqual(len(tasks), 304)
        self.assertEqual(len({t['config']['experiment']['name'] for t in tasks if t['type']=='monitor'}), 12)


if __name__ == '__main__':
    unittest.main()
