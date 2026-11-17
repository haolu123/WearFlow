"""Compare current-VB estimates at epoch 500, including the no-sensor baseline.

Reuse saved predictions only. Range labels follow the original CondWear-ODE
definition: fold A test target VB <= 0.74 mm versus > 0.74 mm.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import project_path, write_json
from .run_fixed_conditioning import MODELS


EPOCH = 500
THRESHOLD_MM = 0.74
STUDY = 'fixed500_baseline_ranges_20260928'
PREDICTION_COLUMNS = [
    'epoch', 'split', 'task', 'case', 'origin_run',
    'target_vb_mm', 'predicted_vb_mm',
]


def load_predictions(output):
    """Add twenty baseline runs to the eighty previously audited sensor runs."""
    previous = output / 'top18_fixed500_20260928'
    frames = [pd.read_csv(previous / 'epoch500_test_predictions.csv')]
    sources = pd.read_csv(previous / 'source_runs.csv').to_dict('records')
    manifest = pd.read_csv(output / 'fixed_conditioning_20260923/study_manifest.csv')
    baseline = manifest[
        (manifest.role == 'final') & (manifest.type == 'monitor')
        & manifest.experiment.isin(['time_ct_mlp', 'time_ct_mlp_forecast'])
    ]
    assert len(baseline) == 20
    for run in baseline.itertuples():
        directory = Path(run.run_dir)
        assert json.loads((directory / 'status.json').read_text())['status'] == 'completed'
        assert (directory / 'epochs/epoch_0500.pt').exists()
        frame = pd.read_csv(directory / 'epoch_predictions.csv.gz', usecols=PREDICTION_COLUMNS)
        frame = frame[
            (frame.epoch == EPOCH) & (frame.split == 'test_diagnostic')
            & (frame.task == 'monitoring')
        ].copy()
        meta = dict(model='time_ct_mlp', mode='ode' if run.experiment.endswith('_forecast') else 'pure',
                    fold=run.fold, seed=run.seed, epoch=EPOCH)
        errors = (frame.predicted_vb_mm - frame.target_vb_mm).abs()
        macro_mae = errors.groupby(frame.case).mean().mean()
        history = pd.read_csv(directory / 'history.csv').set_index('epoch')
        assert abs(macro_mae - history.loc[EPOCH, 'test_diagnostic_monitoring_case_macro_mae']) < 1e-7
        frames.append(frame.assign(**meta))
        sources.append(dict(**meta, run_dir=str(directory),
                            prediction_file=str(directory / 'epoch_predictions.csv.gz')))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(sources)


def calculate_metrics(predictions):
    """Average per-case MAEs, then seeds; also export observation-pooled metrics."""
    masks = {
        'overall': np.ones(len(predictions), dtype=bool),
        'fold_a': predictions.fold == 'fold_a',
        'fold_b': predictions.fold == 'fold_b',
        'fold_a_in_range': (predictions.fold == 'fold_a') & (predictions.target_vb_mm <= THRESHOLD_MM),
        'fold_a_out_of_range': (predictions.fold == 'fold_a') & (predictions.target_vb_mm > THRESHOLD_MM),
    }
    metrics, case_metrics = [], []
    for subset, mask in masks.items():
        selected = predictions.loc[mask].copy()
        selected['absolute_error'] = (selected.predicted_vb_mm - selected.target_vb_mm).abs()
        selected['squared_error'] = (selected.predicted_vb_mm - selected.target_vb_mm) ** 2
        cases = selected.groupby(['model', 'mode', 'seed', 'fold', 'case']).agg(
            mae=('absolute_error', 'mean'), n=('absolute_error', 'size'),
        ).reset_index().assign(subset=subset)
        case_metrics.append(cases)
        for (model, mode), group in selected.groupby(['model', 'mode']):
            per_case = cases[(cases.model == model) & (cases['mode'] == mode)]
            per_seed = per_case.groupby('seed').mae.mean()
            metrics.append(dict(
                model=model, mode=mode, subset=subset,
                case_macro_mae=per_seed.mean(), seed_std_macro_mae=per_seed.std(ddof=1),
                pooled_mae=group.absolute_error.mean(), pooled_rmse=np.sqrt(group.squared_error.mean()),
                n_cases=group.case.nunique(), n_observations=len(group[['case', 'origin_run']].drop_duplicates()),
                n_seeds=group.seed.nunique(),
            ))
    return pd.DataFrame(metrics), pd.concat(case_metrics, ignore_index=True)


def write_report(root, metrics):
    columns = ['overall', 'fold_a', 'fold_b', 'fold_a_in_range', 'fold_a_out_of_range']
    lines = [
        '# 固定第 500 轮：无 sensor 基线与 top-18 sensor 网络', '',
        '所有结果均为当前 VB 估计的测试误差，直接读取已有第 500 轮预测，没有重新训练或重新选择 epoch。',
        '五种结构均比较纯监督与加 ODE（ODE 负责未来 VB 积分传播，当前 VB 仍由 monitoring 网络估计）。', '',
        '## 统计口径', '',
        '- 主表为 case-macro MAE，单位 mm，越小越好：先计算每个 case 的 MAE，再对 case 和 5 个 seed 等权平均。不是先集成预测再计算误差。',
        '- overall 为 14 个测试 case；fold A、B 各 7 个，两个 fold 互换训练和测试 case。case 6、15 均不参与。',
        '- fold A in-range：真实 VB ≤ 0.74 mm；out-of-range：真实 VB > 0.74 mm。阈值来自 fold A 训练集最大 VB，并沿用 release/CondWear-ODE/training/trainer.py 的定义。',
        '- 按观测的真实 VB 分组，同一个 case 可以同时出现在两个范围。范围内的 case-macro 只平均该范围有观测的 case，因此两列不能按观测数加权还原 fold A 的 case-macro。',
        '- metrics.csv 同时提供 pooled MAE、pooled RMSE、seed 间标准差与样本数；原始旧版 range 指标为 pooled，不能直接与本表的 macro 混用。', '',
    ]
    for mode, title in [('pure', '纯监督'), ('ode', '加 ODE')]:
        lines += [f'## {title}', '', '| 模型 | overall | fold A | fold B | A in-range | A out-of-range |',
                  '|---|---:|---:|---:|---:|---:|']
        for model, label, _, _ in MODELS:
            values = metrics[(metrics.model == model) & (metrics['mode'] == mode)].set_index('subset')
            lines.append('| ' + label + ' | ' + ' | '.join(f'{values.loc[c, "case_macro_mae"]:.5f}' for c in columns) + ' |')
        lines.append('')
    counts = metrics[(metrics.model == 'time_ct_mlp') & (metrics['mode'] == 'pure')]
    lines += ['## 数据量（不重复计入 5 个 seed）', '', '| 范围 | case 数 | 有标签观测数 |', '|---|---:|---:|']
    for row in counts.itertuples():
        lines.append(f'| {row.subset} | {row.n_cases} | {row.n_observations} |')
    lines += ['', '## 主要观察', '',
              '- 无 sensor＋ODE 的总体 MAE 最低（0.08331）；纯监督基线为 0.08349，两者仅相差约 0.22%。',
              '- fold A 及其 in-range 上，无 sensor 基线两版都优于四种 sensor 网络。out-of-range 最低也是无 sensor＋ODE（0.18778）；纯监督时间编码网络略优于纯监督基线（0.19559 vs 0.19842）。',
              '- fold B 上 FiLM＋ODE 最低（0.06890），优于无 sensor＋ODE（0.08287，约降低 16.86%）。因此 sensor 的作用明显依赖 fold，不能概括成完全无用。',
              '- FiLM 加 ODE 后两个 fold 都改善，但 fold A out-of-range 改善很小（0.28531 → 0.28398）。attention 加 ODE 后 fold A 总体改善，但 out-of-range 变差（0.25027 → 0.27225）。',
              '- 两个 fold 的工况与 case 组成也不同；该分组只能描述误差分布，不能单独证明差异完全由磨损范围外推造成。', '',
              '## 复现与文件', '',
              '`python -m wear_monitoring_dynamics.scripts.compare_fixed500_ranges`', '',
              '- `metrics.csv`：所有结构、目标与范围的指标。',
              '- `per_case_metrics.csv`：各范围逐 seed、逐 case 的 MAE。',
              '- `epoch500_test_predictions.csv`：本次比较的全部测试预测。',
              '- `source_runs.csv`：100 个原始训练目录与预测文件。',
              '- `range_membership.csv`：每个观测的范围标签。',
              '- `audit.json`：运行完整性、相同测试观测与目标、历史指标一致性核查。', '',
              '这是已有测试集上的探索性复盘；18 个特征由全体 case 的 sensor 轨迹筛选，并非完全嵌套特征选择。固定 500 轮也是查看过已有结果后的对照，不能当作新的独立测试。',
              'out-of-range 样本很少，各 case 观测数也不相等；点估计差异不等于显著提升。', '']
    (root / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    output = project_path('wear_monitoring_dynamics/outputs')
    root = output / STUDY
    root.mkdir(exist_ok=True)
    predictions, sources = load_predictions(output)
    keys = ['model', 'mode', 'fold', 'seed']
    assert len(sources) == 100 and not sources.duplicated(keys).any()
    assert len(predictions) == 6900
    assert not predictions.duplicated(keys + ['case', 'origin_run']).any()
    assert np.isfinite(predictions[['target_vb_mm', 'predicted_vb_mm']].to_numpy()).all()
    for fold, fold_rows in predictions.groupby('fold'):
        reference = None
        for _, group in fold_rows.groupby(['model', 'mode', 'seed']):
            observed = group.sort_values(['case', 'origin_run'])[['case', 'origin_run', 'target_vb_mm']].to_numpy()
            if reference is None:
                reference = observed
            np.testing.assert_allclose(observed, reference, rtol=0, atol=1e-12)
    raw = pd.read_csv(output / 'sensor_feature_inspection_20260927/raw_features_and_vb.csv')
    assert abs(raw[raw.case.isin([1, 2, 3, 4, 5, 7, 8])].VB.max() - THRESHOLD_MM) < 1e-12
    metrics, cases = calculate_metrics(predictions)
    old = pd.read_csv(output / 'top18_fixed500_20260928/comparison.csv')
    for row in old.itertuples():
        for mode in ['pure', 'ode']:
            value = metrics[(metrics.model == row.model) & (metrics['mode'] == mode) & (metrics.subset == 'overall')].case_macro_mae.item()
            assert abs(value - getattr(row, mode + '_mae')) < 1e-12
    membership = predictions[['fold', 'case', 'origin_run', 'target_vb_mm']].drop_duplicates().copy()
    membership['range'] = np.where(membership.fold != 'fold_a', 'fold_b',
                                  np.where(membership.target_vb_mm <= THRESHOLD_MM, 'in_range', 'out_of_range'))
    metrics.to_csv(root / 'metrics.csv', index=False)
    cases.to_csv(root / 'per_case_metrics.csv', index=False)
    predictions.to_csv(root / 'epoch500_test_predictions.csv', index=False)
    sources.to_csv(root / 'source_runs.csv', index=False)
    membership.to_csv(root / 'range_membership.csv', index=False)
    write_report(root, metrics)
    write_json(root / 'audit.json', dict(epoch=EPOCH, threshold_mm=THRESHOLD_MM,
        final_runs=len(sources), prediction_rows=len(predictions), all_checks_passed=True,
        same_test_observations_and_targets=True, sensor_results_match_previous_report=True,
        baseline_matches_saved_history=True, retrained=False))
    print(metrics.pivot(index=['model', 'mode'], columns='subset', values='case_macro_mae').to_string())
    print(membership.groupby('range').agg(n_cases=('case', 'nunique'), n_observations=('case', 'size')).to_string())


if __name__ == '__main__':
    main()
