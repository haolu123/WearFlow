"""Training-only feature selection and controlled fold-B wear coverage studies."""

import argparse

import copy

import json

from pathlib import Path

import numpy as np

import pandas as pd

from ..config import PACKAGE, load_config, project_path, write_json

from ..data.records import load_table

from ..data.features import SENSORS, STATISTICS

from ..data.splits import POOLS

from ..data.subsets import restrict_training_records

from ..training.trainer import run_training

from .rank_sensor_monotonicity import temporal_stats

from .run_fixed_conditioning import MODELS



ROOT = project_path('wear_monitoring_dynamics/outputs/train_only_coverage_20260928')

FEATURES = [f'{s}__{t}' for s in SENSORS for t in STATISTICS]





def rank_features(frame):

    """Original ranking rule, with direction and rank fitted only to supplied rows."""

    rows = []

    for feature in FEATURES:

        rho = [temporal_stats(g, feature, 1)['rho'] for _, g in frame.groupby('case')]

        direction = 1 if np.nanmedian(rho) >= 0 else -1

        values = pd.DataFrame([temporal_stats(g, feature, direction) for _, g in frame.groupby('case')])

        rows.append(dict(feature=feature, direction=direction,

                         case_macro_violation_rate=values.violation_rate.mean(),

                         mean_reverse_variation_fraction=values.reverse_variation_fraction.mean()))

    return pd.DataFrame(rows).sort_values(['case_macro_violation_rate', 'mean_reverse_variation_fraction', 'feature'])





def plan():

    base = load_config(PACKAGE / 'configs/time_conditioning.yaml')

    table, _ = load_table(project_path(base['data']['path']), project_path(base['data']['cache']))

    frame = pd.DataFrame({k: v for k, v in table.items() if k != 'sensor'})

    frame[FEATURES] = table['sensor']

    ROOT.mkdir(exist_ok=True)

    ranks = ROOT / 'feature_selection'; ranks.mkdir(exist_ok=True)

    groups = []

    for fold in POOLS:

        train = frame[frame.case.isin(POOLS[fold])]

        ranking = rank_features(train)

        ranking.to_csv(ranks / f'train_only_{fold}.csv', index=False)

        groups.append(dict(study='train_only', group='all_records', fold=fold,

                           features=ranking.head(18).feature.tolist(), records=None, subset_seed=None))

    full = frame[frame.case.isin(POOLS['fold_b']) & frame.VB.notna()]

    low = full[full.VB <= .74]

    assert len(full) == 74 and len(low) == 65

    ranking = rank_features(low)

    ranking.to_csv(ranks / 'coverage_low_only.csv', index=False)

    features = ranking.head(18).feature.tolist()

    def add_group(name, rows, seed=None):

        records = rows.sort_values(['case', 'run'])[['case', 'run']].astype(int).values.tolist()

        groups.append(dict(study='coverage', group=name, fold='fold_b', features=features,

                           records=records, subset_seed=seed))

    add_group('full74', full); add_group('low65', low)

    for seed in range(5):

        rng = np.random.default_rng(20260928 + seed)

        selected = []

        for case, rows in full.groupby('case'):

            high = rows[rows.VB > .74]

            lower = rows[rows.VB <= .74]

            count = len(lower) - len(high)

            assert count >= 0

            indices = rng.choice(lower.index.to_numpy(), count, replace=False)

            chosen = pd.concat([high, lower.loc[indices]])

            assert len(chosen) == len(lower)

            selected.append(chosen)

        matched = pd.concat(selected)

        assert len(matched) == 65 and (matched.VB > .74).sum() == 9

        add_group('matched65', matched, seed)

    write_json(ROOT / 'groups.json', groups)

    configs = []

    for group in groups:

        seeds = range(5) if group['subset_seed'] is None else [group['subset_seed']]

        for name, label, architecture, inputs in MODELS:

            for mode in ['pure', 'ode']:

                for seed in seeds:

                    cfg = copy.deepcopy(base)

                    cfg['study_id'] = ROOT.name

                    cfg['experiment'].update(name=f"{group['study']}_{group['group']}_{name}_{mode}",

                                             fold=group['fold'], role='final', holdout=None, seed=seed)

                    cfg['monitor'].update(architecture=architecture, inputs=inputs)

                    if 'sensor' in inputs:

                        cfg['monitor']['selected_features'] = group['features']

                    if group['records'] is not None:

                        cfg['data']['train_records'] = group['records']

                    cfg['dynamics']['mode'] = 'joint_pretrained' if mode == 'ode' else 'none'

                    cfg['loss'].update(derivative_weight=0., forecasting_weight=float(mode == 'ode'))

                    cfg['training']['selection_epoch'] = 500

                    configs.append(dict(study=group['study'], group=group['group'], model=name,

                                        mode=mode, fold=group['fold'], seed=seed, config=cfg))

    assert len(configs) == 250

    write_json(ROOT / 'planned_runs.json', configs)

    return configs





def report(rows):

    percase, curves = [], []

    for row in rows:

        directory = Path(row['run_dir'])

        meta = {k: row[k] for k in ('study', 'group', 'model', 'mode', 'fold', 'seed')}

        result = json.loads((directory / 'metrics.json').read_text())

        assert result['selected_epoch'] == 500 and result['reload_verified']

        case = pd.read_csv(directory / 'per_case_metrics.csv').assign(**meta)

        percase.append(case)

        history = pd.read_csv(directory / 'history.csv')

        assert history.epoch.tolist() == list(range(1, 501))

        assert abs(case.mae.mean() - history.iloc[-1].test_diagnostic_monitoring_case_macro_mae) < 1e-7

        curves.append(history.assign(**meta))

    cases = pd.concat(percase, ignore_index=True)

    cases.to_csv(ROOT / 'per_case_metrics.csv', index=False)

    pd.concat(curves, ignore_index=True).to_csv(ROOT / 'epoch_metrics.csv.gz', index=False)

    summary = cases.groupby(['study', 'group', 'model', 'mode', 'fold']).mae.mean().reset_index()

    summary.to_csv(ROOT / 'summary.csv', index=False)

    lines = ['# 结果：固定 500 epoch，训练集特征选择与磨损覆盖', '',

             '指标为当前 VB 的 case-macro MAE（mm），先按 case 计算，再对 case 和五次运行平均。', '',

             '| study | group | model | mode | fold | MAE |', '|---|---|---|---|---|---:|']

    for r in summary.itertuples():

        lines.append(f'| {r.study} | {r.group} | {r.model} | {r.mode} | {r.fold} | {r.mae:.5f} |')

    (ROOT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')

    write_json(ROOT / 'status.json', dict(status='completed', runs=len(rows), audited=True))

    print(summary.to_string(index=False), flush=True)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--execute', action='store_true')

    parser.add_argument('--smoke', action='store_true')

    parser.add_argument('--report-only', action='store_true')

    args = parser.parse_args()

    if args.report_only:

        report(pd.read_csv(ROOT / 'study_manifest.csv').to_dict('records')); return

    configs = plan()

    if not (args.execute or args.smoke): return

    parents, rows = {}, []

    write_json(ROOT / 'status.json', dict(status='smoke' if args.smoke else 'running', planned_runs=250))

    if args.smoke:

        configs = [next(c for c in configs if c['study'] == 'coverage' and c['group'] == 'low65'

                        and c['model'] == m and c['mode'] == mode)

                   for m, _, _, _ in MODELS for mode in ['pure', 'ode']]

    for item in configs:

        cfg = copy.deepcopy(item['config'])

        if args.smoke:

            cfg['study_id'] += '_smoke'

            cfg['training'].update(epochs=3, selection_epoch=3)

            cfg['dynamics']['pretrain_epochs'] = 3

        if item['mode'] == 'ode':

            key = json.dumps([item['fold'], cfg['data'].get('train_records')])

            if key not in parents:

                pre = copy.deepcopy(cfg)

                pre['dynamics']['mode'] = 'none'

                directory = run_training(pre, pretrain=True)

                parents[key] = str(directory / 'last.pt')

            cfg['dynamics']['checkpoint'] = parents[key]

        directory = run_training(cfg)

        row = {k: v for k, v in item.items() if k != 'config'}

        row['run_dir'] = str(directory)

        rows.append(row)

        pd.DataFrame(rows).to_csv(ROOT / ('smoke_manifest.csv' if args.smoke else 'study_manifest.csv'), index=False)

    if not args.smoke:

        report(rows)





if __name__ == '__main__':

    try:

        main()

    except Exception as error:

        write_json(ROOT / 'status.json', dict(status='failed', error=str(error)))

        raise

