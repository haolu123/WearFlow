"""Audit the full epoch-logged study and separate CV-selected from exploratory test-best results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path, write_json
from .summarize_study import summarize


def analyze(root):
    manifest = pd.read_csv(root/'study_manifest.csv')
    planned = json.loads((root/'planned_tasks.json').read_text())
    if set(manifest.task_id) != {t['id'] for t in planned} or not (manifest.status == 'completed').all():
        raise ValueError('Analysis requires all planned tasks completed')
    summarize(root)
    out = root/'summary'
    curves, audit = [], []
    for run in manifest.itertuples():
        directory = Path(run.run_dir)
        metrics = json.loads((directory/'metrics.json').read_text())
        history = pd.read_csv(directory/'history.csv')
        config = yaml.safe_load((directory/'config_resolved.yaml').read_text(encoding='utf-8'))
        expected = config['training']['epochs']
        if history.epoch.tolist() != list(range(1, expected+1)):
            raise AssertionError(f'Missing epochs: {directory}')
        key = 'forecasting' if run.type == 'pretrain' else 'monitoring'
        required = [f'{split}_{key}_{metric}' for split in ['train', 'test_diagnostic']
                    for metric in ['mae', 'rmse', 'case_macro_mae']]
        if run.role == 'inner':
            required.extend([f'validation_{key}_mae', f'validation_{key}_case_macro_mae'])
        if not np.isfinite(history[required].to_numpy()).all():
            raise AssertionError(f'Nonfinite/missing metrics: {directory}')
        counts = {}
        for chunk in pd.read_csv(directory/'epoch_predictions.csv.gz', chunksize=100000):
            if not np.isfinite(chunk[['target_vb_mm', 'predicted_vb_mm']].to_numpy()).all():
                raise AssertionError(f'Nonfinite epoch predictions: {directory}')
            for group, n in chunk.groupby(['epoch', 'split', 'task']).size().items():
                counts[group] = counts.get(group, 0)+int(n)
        for epoch in range(1, expected+1):
            row = history.iloc[epoch-1]
            for name in history.columns:
                if name.endswith('_n') and any(name.startswith(s+'_') for s in ['train', 'validation', 'test_diagnostic']):
                    prefix = name[:-2]
                    split, task = prefix.rsplit('_', 1)
                    if counts.get((epoch, split, task)) != int(row[name]):
                        raise AssertionError(f'Prediction count mismatch: {directory} {epoch} {name}')
        if not metrics['reload_verified']:
            raise AssertionError('Reload check failed')
        if run.type == 'monitor' and run.role == 'final':
            if len(list((directory/'epochs').glob('epoch_*.pt'))) != expected:
                raise AssertionError('Missing final epoch checkpoints')
        history['experiment'], history['fold'], history['seed'], history['role'] = run.experiment, run.fold, run.seed, run.role
        history['task_id'] = run.task_id
        curves.append(history)
        audit.append({'task_id': run.task_id, 'epochs': expected, 'prediction_rows': sum(counts.values()),
                      'reload_verified': True, 'run_dir': str(directory)})
    all_curves = pd.concat(curves, ignore_index=True)
    all_curves.to_csv(out/'all_epoch_metrics.csv.gz', index=False, compression='gzip')
    pd.DataFrame(audit).to_csv(out/'audit_runs.csv', index=False)
    final = all_curves[all_curves.role == 'final']
    final = final[final.experiment != 'ode_pretrain']
    mean_curve = final.groupby(['experiment', 'fold', 'epoch']).agg(
        mean_test_case_macro_mae=('test_diagnostic_monitoring_case_macro_mae', 'mean'),
        sd_test_case_macro_mae=('test_diagnostic_monitoring_case_macro_mae', 'std'),
        mean_test_mae=('test_diagnostic_monitoring_mae', 'mean'),
        mean_test_rmse=('test_diagnostic_monitoring_rmse', 'mean'),
        seeds=('seed', 'nunique')).reset_index()
    if not (mean_curve.seeds == 5).all():
        raise AssertionError('Missing seeds in final epoch curves')
    mean_curve.to_csv(out/'final_epoch_curves_5seed.csv', index=False)
    best = mean_curve.loc[mean_curve.groupby(['experiment', 'fold']).mean_test_case_macro_mae.idxmin()].copy()
    best['selection_basis'] = 'outer_test_exploratory_NOT_unbiased'
    best.to_csv(out/'exploratory_test_best_epochs.csv', index=False)
    best.groupby('experiment').agg(exploratory_test_selected_mae=('mean_test_case_macro_mae', 'mean')).sort_values('exploratory_test_selected_mae').to_csv(out/'exploratory_test_selected_ranking.csv')
    epochs = []
    for path in (root/'epoch_selection').glob('*/*.json'):
        item = json.loads(path.read_text())
        epochs.append({'experiment': path.parent.name, 'fold': path.stem, 'selected_epoch': item['selected_epoch']})
    pd.DataFrame(epochs).to_csv(out/'inner_cv_selected_epochs.csv', index=False)
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    for experiment, group in mean_curve.groupby('experiment'):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for ax, (fold, values) in zip(axes, group.groupby('fold')):
            x, mean, sd = values.epoch.to_numpy(), values.mean_test_case_macro_mae.to_numpy(), values.sd_test_case_macro_mae.to_numpy()
            ax.plot(x, mean, label='Outer-test diagnostic mean')
            ax.fill_between(x, np.maximum(0, mean-sd), mean+sd, alpha=.2, label='Across-seed SD')
            epoch = next(e['selected_epoch'] for e in epochs if e['experiment'] == experiment and e['fold'] == fold)
            ax.axvline(epoch, linestyle='--', color='black', label=f'Inner-CV epoch {epoch}')
            ax.set(xlabel='Epoch', ylabel='Case-macro MAE (mm)', title=fold)
            ax.legend(fontsize=7)
        fig.suptitle(experiment+' (test curves are diagnostic)')
        fig.savefig(figures/f'{experiment}_epoch_curves.png', dpi=150)
        plt.close(fig)
    leaderboard = pd.read_csv(out/'monitoring_leaderboard.csv')
    final_tasks = set(manifest[(manifest.type == 'monitor') & (manifest.role == 'final')].task_id)
    completed = {'tasks': len(audit), 'recorded_epochs': sum(a['epochs'] for a in audit),
                 'prediction_rows': sum(a['prediction_rows'] for a in audit), 'final_refits': len(final_tasks),
                 'final_epoch_checkpoints': sum(a['epochs'] for a in audit if a['task_id'] in final_tasks), 'all_checks_passed': True}
    write_json(out/'audit_summary.json', completed)
    text = '# 完整研究结果\n\n'
    text += f'全部 {len(audit)} 个任务完成；记录 {completed["recorded_epochs"]:,} 个 epoch、{completed["prediction_rows"]:,} 条逐样本预测。\n\n'
    text += '主要排名采用内部验证选定 epoch 的 checkpoint；所有最终训练都继续到配置的完整轮数，额外曲线只作诊断。\n\n'
    text += '## 内部验证选定模型的监测结果\n\n|模型|case-macro MAE (mm)|平均 pooled MAE (mm)|\n|---|---:|---:|\n'
    for r in leaderboard.itertuples():
        text += f'|{r.experiment}|{r.case_macro_mae:.5f}|{r.pooled_fold_mae:.5f}|\n'
    text += '\n## 文件导航\n\n'
    text += '- monitoring_leaderboard.csv：全部配置，主要指标 case-macro MAE。\n'
    text += '- inner_cv_selected_epochs.csv：每种配置每折由内部验证选出的轮数。\n'
    text += '- final_epoch_curves_5seed.csv：最终 5-seed 逐 epoch 测试曲线。\n'
    text += '- all_epoch_metrics.csv.gz：含预训练、内层、最终训练的所有逐 epoch 指标。\n'
    text += '- exploratory_test_best_epochs.csv：按外层测试曲线选择的最佳轮数，仅探索性，不能当作无偏泛化成绩。\n'
    text += '- paired_comparisons.csv：case 级 bootstrap；CI 未作多重比较校正。\n'
    text += '- audit_summary.json、audit_runs.csv：完整性验证。\n'
    text += '- figures/：每种配置的双折逐 epoch 曲线。\n'
    text += '\n每次运行目录包含 epoch_predictions.csv.gz、history.csv 和配置；最终模型还有 epochs/ 下的全部 epoch checkpoint。预训练预测采用实测 VB 初值，监测耦合预测采用估计初值，不混合比较。\n'
    (out/'RESULTS.md').write_text(text, encoding='utf-8')
    print(json.dumps(completed, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study-id', default='core_epochlog_20260921')
    args = parser.parse_args()
    analyze(project_path('wear_monitoring_dynamics/outputs')/args.study_id)


if __name__ == '__main__':
    main()
