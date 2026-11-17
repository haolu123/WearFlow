"""Declarative 24-configuration study and its 592-task dependency graph."""
import copy
import pandas as pd
import numpy as np
from .data.splits import POOLS

INPUTS = {'t': ['time'], 'ct': ['condition', 'time'], 's': ['sensor'],
          'sc': ['sensor', 'condition'], 'st': ['sensor', 'time'],
          'sct': ['sensor', 'condition', 'time']}


def variants(base, study):
    if study.get('family') == 'explicit':
        from .config import merge
        return [merge(base, {'study_id': study['study_id'], **item}) for item in study['variants']]
    if study.get('family') == 'zhang_transfer':
        return zhang_variants(base, study)
    result = []
    for mode in study['input_modes']:
        cfg = copy.deepcopy(base)
        cfg['study_id'] = study['study_id']
        cfg['experiment']['name'] = f'monitor_{mode}'
        cfg['monitor']['inputs'] = INPUTS[mode]
        cfg['dynamics']['mode'] = 'none'
        cfg['loss'].update(derivative_weight=0., forecasting_weight=0.)
        result.append(cfg)
    for inputs in study['coupled_input_modes']:
        for objective in study['auxiliaries']:
            for mode in study['dynamics_modes']:
                cfg = copy.deepcopy(base)
                cfg['study_id'] = study['study_id']
                cfg['experiment']['name'] = f'{inputs}_{objective}_{mode}'
                cfg['monitor']['inputs'] = INPUTS[inputs]
                cfg['dynamics']['mode'] = mode
                cfg['loss']['derivative_weight'] = study['derivative_weight'] if objective in ('derivative', 'both') else 0.
                cfg['loss']['forecasting_weight'] = study['forecasting_weight'] if objective in ('forecasting', 'both') else 0.
                result.append(cfg)
    return result


def zhang_variants(base, study):
    result = []
    for inputs in ('ct', 'sct'):
        for objective in ('mlp', 'supervised', 'pinn', 'derivative', 'forecasting', 'both'):
            cfg = copy.deepcopy(base)
            cfg['study_id'] = study['study_id']
            cfg['experiment']['name'] = f'zhang_{inputs}_{objective}'
            cfg['monitor'].update(architecture='mlp' if objective == 'mlp' else 'zhang', inputs=INPUTS[inputs])
            cfg['zhang']['physics'] = objective == 'pinn'
            cfg['dynamics']['mode'] = 'frozen' if objective in ('derivative', 'forecasting', 'both') else 'none'
            cfg['loss']['derivative_weight'] = study['derivative_weight'] if objective in ('derivative', 'both') else 0.
            cfg['loss']['forecasting_weight'] = study['forecasting_weight'] if objective in ('forecasting', 'both') else 0.
            result.append(cfg)
    return result


def build_tasks(base, study):
    tasks = []
    models = variants(base, study)
    pretraining = {}
    needs_pretraining = any(m['dynamics']['mode'] in ('frozen', 'joint_pretrained') for m in models)
    for fold in study['folds'] if needs_pretraining else []:
        for role, holdout in [('inner', h) for h in POOLS[fold]] + [('final', None)]:
            cfg = copy.deepcopy(base)
            cfg['study_id'] = study['study_id']
            cfg['experiment'].update(name='ode_pretrain', fold=fold, role=role, holdout=holdout)
            key = f'ode/{fold}/{holdout}'
            pretraining[(fold, holdout)] = key
            tasks.append({'id': key, 'type': 'pretrain', 'config': cfg, 'dependencies': []})
    for variant in models:
        for fold in study['folds']:
            from .config import merge
            fold_variant = copy.deepcopy(variant)
            overrides = fold_variant.pop('fold_overrides', {})
            fold_variant = merge(fold_variant, overrides.get(fold, {}))
            inner_ids = []
            for holdout in POOLS[fold]:
                cfg = copy.deepcopy(fold_variant)
                cfg['experiment'].update(fold=fold, role='inner', holdout=holdout, seed=study['inner_seed'])
                key = f'{cfg["experiment"]["name"]}/{fold}/inner/{holdout}'
                deps = [pretraining[(fold, holdout)]] if cfg['dynamics']['mode'] in ('frozen', 'joint_pretrained') else []
                tasks.append({'id': key, 'type': 'monitor', 'config': cfg, 'dependencies': deps})
                inner_ids.append(key)
            for seed in study['seeds']:
                cfg = copy.deepcopy(fold_variant)
                cfg['experiment'].update(fold=fold, role='final', holdout=None, seed=seed)
                deps = list(inner_ids)
                if cfg['dynamics']['mode'] in ('frozen', 'joint_pretrained'):
                    deps.append(pretraining[(fold, None)])
                tasks.append({'id': f'{cfg["experiment"]["name"]}/{fold}/final/{seed}',
                              'type': 'monitor', 'config': cfg, 'dependencies': deps, 'inner_ids': inner_ids})
    return tasks


def select_epoch(run_dirs):
    curves = [pd.read_csv(p / 'history.csv').set_index('epoch')['validation_case_macro_mae'] for p in run_dirs]
    if len(curves) != 7 or any(not c.index.equals(curves[0].index) for c in curves):
        raise ValueError('Epoch selection requires seven complete aligned inner-CV curves')
    if any(not np.isfinite(c.to_numpy()).all() for c in curves):
        raise ValueError('Nonfinite validation curve')
    mean = pd.concat(curves, axis=1).mean(axis=1)
    return int(mean.idxmin()), mean
