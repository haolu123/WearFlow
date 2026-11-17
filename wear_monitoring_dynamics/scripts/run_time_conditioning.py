"""Two-stage study with per-outer-fold architecture selection using inner CV only.

Run --smoke first for real GPU validation/timing. Without --smoke, runs both
complete stages and audits every epoch. Test diagnostics are never read by the
architecture-selection function.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import yaml
from ..config import PACKAGE, PROJECT, load_config, project_path, write_json
from ..models.registry import build_model
from ..study import variants
from ..training.trainer import run_training
from ..evaluation.bootstrap import paired_case_bootstrap

STAGE_ONE = 'time_conditioning_20260922'
STAGE_TWO = 'time_conditioning_ode_20260922'
CANDIDATES = {'time_sct_encoded': 'time_encoded', 'time_sct_film': 'film',
              'time_sct_attention': 'sensor_attention'}


def select_architectures(root):
    """Only read seven-case averaged inner-validation curves, separately per fold."""
    choices, rows = {}, []
    for fold in ('fold_a', 'fold_b'):
        for name, architecture in CANDIDATES.items():
            curve = pd.read_csv(root/'epoch_selection'/name/f'{fold}_inner_mean.csv')
            best = curve.loc[curve.case_macro_mae.idxmin()]
            rows.append({'fold': fold, 'experiment': name, 'architecture': architecture,
                         'validation_case_macro_mae': float(best.case_macro_mae), 'epoch': int(best.epoch)})
        ranking = sorted([r for r in rows if r['fold'] == fold],
                         key=lambda r: (r['validation_case_macro_mae'], r['experiment']))
        choices[fold] = ranking[0]
    return choices, rows


def stage_two_spec(choices):
    result = {'study_id': STAGE_TWO, 'family': 'explicit', 'seeds': list(range(5)),
              'folds': ['fold_a', 'fold_b'], 'inner_seed': 0, 'variants': []}
    for name, inputs in [('time_ct_mlp_forecast', ['condition', 'time']),
                         ('time_sct_mlp_forecast', ['sensor', 'condition', 'time']),
                         ('time_selected_new_forecast', ['sensor', 'condition', 'time'])]:
        item = {'experiment': {'name': name}, 'monitor': {'architecture': 'mlp', 'inputs': inputs},
                'dynamics': {'mode': 'joint_pretrained'},
                'loss': {'derivative_weight': 0., 'forecasting_weight': 1.}}
        if name == 'time_selected_new_forecast':
            item['fold_overrides'] = {fold: {'monitor': {'architecture': selection['architecture']}}
                                      for fold, selection in choices.items()}
        result['variants'].append(item)
    return result


def smoke(base, study):
    base = copy.deepcopy(base)
    base['training']['epochs'] = 20
    base['dynamics']['pretrain_epochs'] = 5
    base['study_id'] = 'time_conditioning_smoke_20260922'
    study = copy.deepcopy(study)
    study['study_id'] = base['study_id']
    pretrain = run_training(base, pretrain=True)
    rows = []
    for cfg in variants(base, study):
        for forecast in (False, True):
            cfg = copy.deepcopy(cfg)
            if forecast:
                cfg['experiment']['name'] += '_forecast'
                cfg['dynamics'].update(mode='joint_pretrained', checkpoint=str(pretrain/'last.pt'))
                cfg['loss']['forecasting_weight'] = 1.
            directory = run_training(cfg)
            metrics = json.loads((directory/'metrics.json').read_text())
            rows.append({'experiment': cfg['experiment']['name'],
                         'parameters': sum(p.numel() for p in build_model(cfg).monitor.parameters()),
                         'steady_epoch_seconds': metrics['steady_epoch_seconds'],
                         'wall_seconds': metrics['wall_seconds'], 'run_dir': str(directory)})
            pd.DataFrame(rows).to_csv(project_path(base['output_root'])/base['study_id']/'timing.csv', index=False)
    print(pd.DataFrame(rows).drop(columns='run_dir').to_string(index=False), flush=True)


def run_study(config, study_path):
    subprocess.run([sys.executable, '-m', 'wear_monitoring_dynamics.scripts.run_study',
                    '--config', str(config), '--study', str(study_path), '--execute'], cwd=PROJECT, check=True)


def combined_summary(first, second, choices):
    """Matched case comparisons, including the fold-specific selected pure monitor."""
    from .analyze_completed import analyze
    analyze(first)
    analyze(second)
    cases1 = pd.read_csv(first/'summary/per_case_metrics.csv')
    cases2 = pd.read_csv(second/'summary/per_case_metrics.csv')
    selected = pd.concat([cases1[(cases1.fold == fold) & (cases1.experiment == choice['experiment'])]
                          for fold, choice in choices.items()], ignore_index=True)
    selected = selected.assign(experiment='time_selected_new_supervised')
    cases = pd.concat([cases1, selected, cases2], ignore_index=True)
    cases.to_csv(second/'summary/combined_per_case_metrics.csv', index=False)
    cases.groupby('experiment').mae.mean().sort_values().to_csv(second/'summary/combined_monitoring_mae.csv')
    pairs = [('time_sct_mlp', 'time_ct_mlp'), ('time_sct_encoded', 'time_sct_mlp'),
             ('time_sct_film', 'time_sct_mlp'), ('time_sct_attention', 'time_sct_mlp'),
             ('time_sct_film', 'time_sct_encoded'), ('time_sct_attention', 'time_sct_encoded'),
             ('time_selected_new_forecast', 'time_selected_new_supervised'),
             ('time_sct_mlp_forecast', 'time_sct_mlp'), ('time_ct_mlp_forecast', 'time_ct_mlp'),
             ('time_selected_new_forecast', 'time_sct_mlp_forecast'),
             ('time_selected_new_forecast', 'time_ct_mlp_forecast')]
    comparisons = []
    for left, right in pairs:
        a = cases[cases.experiment == left].groupby(['fold', 'case']).mae.mean()
        b = cases[cases.experiment == right].groupby(['fold', 'case']).mae.mean()
        comparisons.append({'left': left, 'right': right, **paired_case_bootstrap(a, b)})
    pd.DataFrame(comparisons).to_csv(second/'summary/combined_paired_comparisons.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    config = PACKAGE/'configs/time_conditioning.yaml'
    study_path = PACKAGE/'configs/studies/time_conditioning_20260922.yaml'
    base = load_config(config)
    study = yaml.safe_load(study_path.read_text())
    if args.smoke:
        smoke(base, study)
        return
    first = project_path(base['output_root'])/STAGE_ONE
    second = project_path(base['output_root'])/STAGE_TWO
    run_study(config, study_path)
    choices, ranking = select_architectures(first)
    # Persist the decision before any analysis reads outer-test performance.
    write_json(first/'architecture_selection.json', {'basis': 'per-fold inner validation only',
                                                   'choices': choices, 'ranking': ranking})
    second.mkdir(parents=True, exist_ok=True)
    second_spec = second/'study_spec.yaml'
    second_spec.write_text(yaml.safe_dump(stage_two_spec(choices), sort_keys=False), encoding='utf-8')
    print('INNER-CV ARCHITECTURE SELECTION: '+json.dumps(choices), flush=True)
    run_study(config, second_spec)
    combined_summary(first, second, choices)
    print('BOTH STAGES COMPLETE AND AUDITED', flush=True)


if __name__ == '__main__':
    main()
