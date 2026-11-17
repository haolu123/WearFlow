"""Conditioning, tokenization, continuous derivatives, and study isolation."""
import copy
import unittest
import tempfile
from pathlib import Path
import pandas as pd
import torch
import yaml
from wear_monitoring_dynamics.config import PACKAGE, load_config
from wear_monitoring_dynamics.models.registry import build_model
from wear_monitoring_dynamics.study import build_tasks


def test_continuous_derivatives_and_ode_gradients():
    for architecture in ('time_encoded', 'film', 'sensor_attention'):
        torch.manual_seed(7)
        cfg = load_config()
        cfg['monitor']['architecture'] = architecture
        cfg['dynamics']['mode'] = 'joint_scratch'
        model = build_model(cfg).double()
        sensor = torch.randn(4, 54, dtype=torch.double)
        condition = torch.randn(4, 4, dtype=torch.double)
        time = torch.randn(4, 1, dtype=torch.double, requires_grad=True)
        value = model.monitor(sensor, condition, time)
        derivative = torch.autograd.grad(value.sum(), time, create_graph=True)[0]
        epsilon = 1e-5
        difference = (model.monitor(sensor, condition, time + epsilon) - model.monitor(sensor, condition, time - epsilon)) / (2 * epsilon)
        torch.testing.assert_close(derivative, difference, atol=1e-7, rtol=1e-5)
        future = model.dynamics.propagate(value, condition, torch.full_like(time, 0.2))
        (derivative.square().mean() + future.square().mean()).backward()
        grads = [p.grad for p in model.monitor.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0


def test_attention_uses_six_tokens_and_context():
    torch.manual_seed(3)
    cfg = load_config()
    cfg['monitor']['architecture'] = 'sensor_attention'
    model = build_model(cfg).monitor
    sensor, condition, time = torch.randn(4, 54), torch.randn(4, 4), torch.randn(4, 1)
    weights = model.attention_weights(sensor, condition, time)
    assert weights.shape == (4, 3, 6)
    torch.testing.assert_close(weights.sum(-1), torch.ones(4, 3))
    assert not torch.allclose(weights, model.attention_weights(sensor, condition, time + 1))
    assert not torch.allclose(model(sensor, condition, time), model(sensor + 1, condition, time))


def test_budget_and_explicit_fold_overrides():
    base = load_config(PACKAGE / 'configs/time_conditioning.yaml')
    study = yaml.safe_load((PACKAGE / 'configs/studies/time_conditioning_20260922.yaml').read_text())
    tasks = build_tasks(base, study)
    assert len(tasks) == 120 and all(t['type'] == 'monitor' for t in tasks)
    counts = {}
    for architecture in ('mlp', 'time_encoded', 'film', 'sensor_attention'):
        cfg = copy.deepcopy(base)
        cfg['monitor']['architecture'] = architecture
        counts[architecture] = sum(p.numel() for p in build_model(cfg).monitor.parameters())
    assert all(abs(n / counts['mlp'] - 1) < .05 for n in counts.values())
    study['variants'] = [study['variants'][1]]
    study['variants'][0]['fold_overrides'] = {'fold_b': {'monitor': {'architecture': 'film'}}}
    for task in build_tasks(base, study):
        cfg = task['config']
        assert cfg['monitor']['architecture'] == ('film' if cfg['experiment']['fold'] == 'fold_b' else 'mlp')
        assert 'fold_overrides' not in cfg


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_continuous_derivatives_and_ode_gradients,
        test_attention_uses_six_tokens_and_context,
        test_budget_and_explicit_fold_overrides,
        test_selection_is_fold_local_and_needs_no_test_metrics,
    ))


def test_selection_is_fold_local_and_needs_no_test_metrics():
    from wear_monitoring_dynamics.scripts.run_time_conditioning import select_architectures, stage_two_spec, CANDIDATES
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for fold in ('fold_a', 'fold_b'):
            for i, name in enumerate(CANDIDATES):
                directory = root/'epoch_selection'/name
                directory.mkdir(parents=True, exist_ok=True)
                score = i + 1 if fold == 'fold_a' else 3 - i
                pd.DataFrame({'epoch': [1, 2], 'case_macro_mae': [score + 1, score]}).to_csv(directory/f'{fold}_inner_mean.csv', index=False)
        choices, _ = select_architectures(root)
        assert choices['fold_a']['architecture'] == 'time_encoded'
        assert choices['fold_b']['architecture'] == 'sensor_attention'
        tasks = build_tasks(load_config(), stage_two_spec(choices))
        assert len(tasks) == 88
        for task in tasks:
            if task['config']['experiment']['name'] == 'time_selected_new_forecast':
                fold = task['config']['experiment']['fold']
                assert task['config']['monitor']['architecture'] == choices[fold]['architecture']
