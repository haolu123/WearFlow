"""Full-batch training with split-safe pretraining and explicit final evaluation."""

import copy

import json

import platform

import time

import traceback

import zipfile

from importlib.metadata import distributions

from pathlib import Path

from contextlib import nullcontext



import numpy as np

import pandas as pd

import torch

import yaml



from ..config import PACKAGE, PROJECT, digest, file_hash, project_path, source_hash, write_json

from ..data.records import load_table

from ..data.splits import make_split

from ..data.preprocessing import fit_scalers

from ..data.datasets import tensors, observations, pair_tensors

from ..models.registry import build_model, build_dynamics

from ..losses import monitoring_loss, derivative_loss, forecasting_loss

from ..evaluation.monitoring import evaluate_monitor

from ..evaluation.forecasting import evaluate_forecast

from ..evaluation.epoch_logging import EpochRecorder

from .checkpoint import save_checkpoint, load_checkpoint, load_pretrained

from .reproducibility import setup, synchronize

from .paired import make_optimizer, train_paired_epoch, schedule_after_epoch





def validate_config(cfg, pretrain=False):

    # MONOTONICITY_EXTENSION_BEGIN

    mono = cfg.get('monotonicity', {})

    if any(not np.isfinite(mono.get(k, 0.)) or mono.get(k, 0.) < 0 for k in ('weight', 'tolerance_z')):

        raise ValueError('Monotonicity weight/tolerance must be finite and nonnegative')

    if not pretrain and mono.get('weight', 0.) and cfg['training'].get('protocol') == 'zhang_paired':

        raise ValueError('Trajectory monotonicity currently supports full-batch training only')

    # MONOTONICITY_EXTENSION_END

    gate = cfg['monitor'].get('gate', {})

    if not np.isfinite(gate.get('l1',0.)) or gate.get('l1',0.) < 0:

        raise ValueError('Gate L1 must be finite and nonnegative')

    weights = cfg['loss']

    if cfg['dynamics'].get('transition', 'ode') != 'ode' and weights['derivative_weight']:

        raise ValueError('Derivative loss requires an ODE rate field')

    if cfg['training']['epochs'] < 1 or cfg['dynamics']['pretrain_epochs'] < 1:

        raise ValueError('Epoch counts must be positive')

    if any(v < 0 for v in weights.values()) or weights['monitoring_weight'] <= 0:

        raise ValueError('Loss weights must be nonnegative with positive monitoring weight')

    if cfg['data']['derivative_observations'] not in ('labeled', 'all'):

        raise ValueError('derivative_observations must be labeled or all')

    horizons = cfg['data']['horizons']

    if not horizons or any(h <= 0 or int(h) != h for h in horizons):

        raise ValueError('Horizons must be positive integers')

    mode = cfg['dynamics']['mode']

    if mode not in ('none', 'frozen', 'joint_pretrained', 'joint_scratch'):

        raise ValueError(f'Invalid dynamics mode {mode}')

    if not pretrain:

        auxiliary = weights['derivative_weight'] + weights['forecasting_weight']

        if bool(auxiliary) != (mode != 'none'):

            raise ValueError('Dynamics and an auxiliary objective must be enabled together')

        if weights['derivative_weight'] and 'time' not in cfg['monitor']['inputs']:

            raise ValueError('Time input is required for the partial derivative loss')

        if mode in ('frozen', 'joint_pretrained') and not cfg['dynamics']['checkpoint']:

            raise ValueError('This dynamics mode requires a matching pretrained checkpoint')





def run_training(config, pretrain=False):

    started = time.perf_counter()

    cfg = copy.deepcopy(config)

    validate_config(cfg, pretrain)

    exp, train = cfg['experiment'], cfg['training']

    if pretrain:

        exp['seed'] = cfg['dynamics']['pretrain_seed']

        train['epochs'] = cfg['dynamics']['pretrain_epochs']

    device = setup(exp['seed'], train['device'], train['deterministic'])

    table, signature = load_table(project_path(cfg['data']['path']), project_path(cfg['data']['cache']), cfg['data']['raw_sensor_abs_limit'])

    split = make_split(exp['fold'], exp['role'], exp.get('holdout'))

    from ..data.subsets import restrict_training_records

    table = restrict_training_records(table, split['train'], cfg['data'].get('train_records'))

    from ..data.sensor_quality import apply_quality

    table = apply_quality(table,cfg['data'].get('sensor_quality'))

    if cfg['data'].get('feature_archive'):

        from ..data.feature_archive import attach_features

        table=attach_features(table,cfg['data']['feature_archive'])

    scalers = fit_scalers(table, split['train'],cfg['data'].get('sensor_preprocessing'))

    source = source_hash()

    parent = cfg['dynamics'].get('checkpoint') if not pretrain else None

    identity = {'config': cfg, 'data_signature': signature, 'source_hash': source,

                'torch': str(torch.__version__), 'numpy': str(np.__version__),

                'python': platform.python_version(), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),

                'parent_hash': file_hash(parent) if parent else None, 'pretrain': pretrain}

    run_id = digest(identity)[:12]

    scope = f'{exp["role"]}_holdout{exp.get("holdout")}' if exp['role'] != 'final' else 'final'

    name = 'ode_pretrain' if pretrain else exp['name']

    directory = project_path(cfg['output_root']) / cfg['study_id'] / 'runs' / name / exp['fold'] / scope / f'seed_{exp["seed"]}' / run_id

    status_file = directory / 'status.json'

    if status_file.exists() and json.loads(status_file.read_text())['status'] == 'completed':

        print(f'SKIP completed: {directory}', flush=True)

        return directory

    directory.mkdir(parents=True, exist_ok=True)

    write_json(status_file, {'status': 'running', 'run_id': run_id})

    (directory / 'config_resolved.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    for filename, value in [('provenance.json', identity), ('split.json', split), ('scalers.json', scalers)]:

        write_json(directory / filename, value)

    quality_path = project_path(cfg['data']['cache']) / f'features_{signature[:16]}.json'

    write_json(directory / 'data_quality.json', json.loads(quality_path.read_text(encoding='utf-8')))

    environment = {d.metadata['Name']: d.version for d in distributions() if d.metadata['Name']}

    write_json(directory / 'environment.json', environment)

    snapshot = project_path(cfg['output_root']) / 'sources' / f'{source}.zip'

    snapshot.parent.mkdir(parents=True, exist_ok=True)

    if not snapshot.exists():

        with zipfile.ZipFile(snapshot, 'w', zipfile.ZIP_DEFLATED) as archive:

            for path in sorted(PACKAGE.rglob('*.py')) + [PROJECT / 'release/CondWear-ODE/models/vb_sensor_condition_ode.py']:

                archive.write(path, str(path.relative_to(PROJECT)))

    try:

        metrics = _train(cfg, table, signature, split, scalers, device, directory, pretrain)

        metrics['wall_seconds'] = time.perf_counter() - started

        write_json(directory / 'metrics.json', metrics)

        write_json(status_file, {'status': 'completed', 'run_id': run_id, 'wall_seconds': metrics['wall_seconds']})

        print(f'COMPLETE {name}: {directory}', flush=True)

        return directory

    except Exception:

        error = traceback.format_exc()

        (directory / 'error.log').write_text(error, encoding='utf-8')

        write_json(status_file, {'status': 'failed', 'run_id': run_id, 'error': error})

        raise





def _train(cfg, table, signature, split, scalers, device, directory, pretrain):

    train, loss_cfg = cfg['training'], cfg['loss']

    from ..data.monitor_batch import make_monitor_batch

    batch = make_monitor_batch(cfg,table,scalers,device)

    obs = observations(table, split['train'])

    if (cfg['monitor'].get('history_length') or cfg['monitor'].get('skip_invalid_sensor')) and 'sensor_valid' in table:

        obs = obs[table['sensor_valid'][obs]]

    derivative_obs = observations(table, split['train'], cfg['data']['derivative_observations'] == 'labeled')

    pairs = pair_tensors(table, split['train'], cfg['data']['horizons'], device)

    # MONOTONICITY_EXTENSION_BEGIN

    mono_cfg = cfg.get('monotonicity', {})

    mono_weight = mono_cfg.get('weight', 0.) if not pretrain else 0.

    mono_pairs = pair_tensors(table, split['train'], [1], device) if mono_weight else None

    # MONOTONICITY_EXTENSION_END

    paired = train.get('protocol') == 'zhang_paired' and not pretrain

    adjacent = pair_tensors(table, split['train'], [1], device) if paired else None

    model = (build_dynamics(cfg['dynamics']) if pretrain else build_model(cfg)).to(device)

    if not pretrain and cfg['dynamics']['mode'] in ('frozen', 'joint_pretrained'):

        pretrained = load_pretrained(cfg['dynamics']['checkpoint'], model.dynamics, scalers, split, signature, device)

        if pretrained['config']['data'].get('train_records') != cfg['data'].get('train_records'):

            raise ValueError('Pretrained ODE training records differ from monitor training records')

        if cfg['dynamics']['mode'] == 'frozen':

            model.dynamics.requires_grad_(False)

    optimizer = (make_optimizer(model, cfg) if paired else

                 torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],

                                   lr=train.get('pretrain_learning_rate', train['learning_rate']) if pretrain else train['learning_rate'],

                                   weight_decay=train['weight_decay']))

    history, best = [], float('inf')

    if device.type == 'cuda':

        torch.cuda.reset_peak_memory_stats(device)

    recorder = EpochRecorder(directory, table, batch, scalers, split, cfg['data']['horizons'], pretrain) if train.get('record_all_epoch_splits') else None

    with (directory / 'train.log').open('w', encoding='utf-8') as log, (recorder if recorder else nullcontext()):

        for epoch in range(1, train['epochs'] + 1):

            synchronize(device)

            start = time.perf_counter()

            model.train()

            optimizer.zero_grad(set_to_none=True)

            if paired:

                row = {'epoch': epoch, **train_paired_epoch(model, batch, adjacent, pairs, optimizer, cfg)}

                row['monitor_learning_rate'] = optimizer.param_groups[0]['lr']

                schedule_after_epoch(optimizer, epoch, train)

            elif pretrain:

                origin, target = pairs[:, 0], pairs[:, 1]

                prediction = model.propagate(batch['target'][origin], batch['condition'][origin],

                                              batch['time'][target] - batch['time'][origin])

                components = {'pretraining': (prediction-batch['target'][target]).square().mean()}

                total = components['pretraining']

            else:

                components = {'monitoring': monitoring_loss(model, batch, obs)}

                total = loss_cfg['monitoring_weight'] * components['monitoring']

                if hasattr(model, 'gates'):

                    components['gate_l1'] = model.gates(batch, obs).mean()

                    total = total + cfg['monitor']['gate']['l1'] * components['gate_l1']

                if loss_cfg['derivative_weight']:

                    components['derivative'] = derivative_loss(model, batch, derivative_obs)

                    total = total + loss_cfg['derivative_weight'] * components['derivative']

                if loss_cfg['forecasting_weight']:

                    components['forecasting'] = forecasting_loss(model, batch, pairs)

                    total = total + loss_cfg['forecasting_weight'] * components['forecasting']

                # MONOTONICITY_EXTENSION_BEGIN

                if mono_weight:

                    from ..losses.monotonicity import monotonicity_loss

                    components['monotonicity'] = monotonicity_loss(model, batch, mono_pairs, mono_cfg.get('tolerance_z', 0.))

                    total = total + mono_weight * components['monotonicity']

                # MONOTONICITY_EXTENSION_END

            if not paired:

                if not torch.isfinite(total):

                    raise FloatingPointError('Nonfinite loss')

                total.backward()

                if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):

                    raise FloatingPointError('Nonfinite gradient')

                optimizer.step()

                row = {'epoch': epoch, 'loss': float(total.detach()),

                       **{f'loss_{k}': float(v.detach()) for k, v in components.items()}}

            if hasattr(model, 'gates'):

                for label, cases in [('train', split['train']), ('validation',split['validation']), ('test_diagnostic', split.get('outer_test_reserved',split['test']))]:

                    ids = observations(table, cases)

                    ids = ids[table['sensor_valid'][ids]]

                    if len(ids):

                        row.update({f'gate_{label}_{k}':v for k,v in model.gate_statistics(batch,ids).items()})

            improved = False

            if recorder:

                row.update(recorder.record(model, epoch))

            if split['validation'] and not pretrain:

                if recorder:

                    score = row['validation_monitoring_case_macro_mae']

                else:

                    val, _, _ = evaluate_monitor(model, batch, table, scalers, split['validation'])

                    score = val['case_macro_mae']

                row['validation_case_macro_mae'] = score

                improved = score < best

                if improved:

                    best = score

            synchronize(device)

            row['compute_seconds'] = time.perf_counter() - start

            if improved:

                save_checkpoint(directory / 'best.pt', model, epoch, cfg, scalers, split, signature)

            if not pretrain and cfg['experiment']['role'] == 'final':

                if epoch == train.get('selection_epoch'):

                    save_checkpoint(directory / 'selected.pt', model, epoch, cfg, scalers, split, signature)

                if train.get('save_final_epoch_checkpoints'):

                    epoch_dir = directory / 'epochs'

                    epoch_dir.mkdir(exist_ok=True)

                    save_checkpoint(epoch_dir / f'epoch_{epoch:04d}.pt', model, epoch, cfg, scalers, split, signature)

            synchronize(device)

            row['epoch_seconds'] = time.perf_counter() - start

            history.append(row)

            log.write(json.dumps(row) + '\n')

            log.flush()

            # Append after each epoch so interrupted runs retain their curves.

            pd.DataFrame([row]).to_csv(directory / 'history.csv', mode='w' if epoch == 1 else 'a', header=epoch == 1, index=False)

    save_checkpoint(directory / 'last.pt', model, epoch, cfg, scalers, split, signature)

    pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)

    times = [r['epoch_seconds'] for r in history[train['warmup_epochs']:]]

    metrics = {'task': 'pretrain' if pretrain else 'monitoring', 'device': str(device),

               'epochs': epoch, 'train_observations': len(obs), 'train_pairs': len(pairs),

               'steady_epoch_seconds': float(np.median(times)) if times else None,

               'peak_cuda_memory_mb': torch.cuda.max_memory_allocated(device)/1024**2 if device.type == 'cuda' else 0}

    if pretrain:

        model.eval()

        with torch.no_grad():

            before = model.propagate(batch['target'][pairs[:, 0]], batch['condition'][pairs[:, 0]],

                                     batch['time'][pairs[:, 1]]-batch['time'][pairs[:, 0]])

        restored = build_dynamics(cfg['dynamics']).to(device)

        load_checkpoint(directory / 'last.pt', restored, device)

        with torch.no_grad():

            after = restored.propagate(batch['target'][pairs[:, 0]], batch['condition'][pairs[:, 0]],

                                       batch['time'][pairs[:, 1]]-batch['time'][pairs[:, 0]])

        torch.testing.assert_close(before, after, rtol=0, atol=0)

        metrics['reload_verified'] = True

        return metrics

    selected = directory / ('best.pt' if split['validation'] else ('selected.pt' if train.get('selection_epoch') else 'last.pt'))

    if train.get('fixed_final_epoch'):

        selected = directory / 'last.pt'

    selected_state = load_checkpoint(selected, model, device)

    # Inner and smoke runs never evaluate the reserved outer test pool.

    cases = split['validation'] or split['test']

    result, frame, per_case = evaluate_monitor(model, batch, table, scalers, cases)

    frame.to_csv(directory / 'monitoring_predictions.csv', index=False)

    per_case.to_csv(directory / 'per_case_metrics.csv', index=False)

    restored = build_model(cfg).to(device)

    load_checkpoint(selected, restored, device)

    again, reloaded, _ = evaluate_monitor(restored, batch, table, scalers, cases)

    if not np.allclose(frame.predicted_vb_mm, reloaded.predicted_vb_mm, rtol=0, atol=1e-7):

        raise AssertionError('Checkpoint reload predictions disagree')

    if model.dynamics is not None:

        auxiliary, predictions = evaluate_forecast(restored, batch, table, scalers, cases, cfg['data']['horizons'])

        predictions.to_csv(directory / 'forecasting_predictions.csv', index=False)

        metrics['forecasting_by_horizon'] = auxiliary

        if 'time' in cfg['monitor']['inputs'] and cfg['dynamics'].get('transition', 'ode') == 'ode':

            from ..evaluation.derivative import evaluate_derivative

            diagnostic, rates = evaluate_derivative(restored, batch, table, scalers, cases)

            rates.to_csv(directory / 'derivative_diagnostics.csv', index=False)

            metrics['derivative_diagnostics'] = diagnostic

    metrics.update(monitoring=result, selected_checkpoint=selected.name, selected_epoch=selected_state['epoch'],

                   evaluation_split='validation' if split['validation'] else 'test', reload_verified=True)

    return metrics

