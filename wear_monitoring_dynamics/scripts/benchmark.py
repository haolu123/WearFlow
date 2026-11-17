"""Bounded validation only: 11 short jobs, no outer-test evaluation or core study launch."""
import argparse
import copy
import json
import time
from pathlib import Path
import pandas as pd
import yaml
from ..config import PACKAGE, load_config, project_path, write_json
from ..study import variants
from ..training.trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--study-id', default='benchmark_v1')
    args = parser.parse_args()
    if not 4 <= args.epochs <= 20:
        raise ValueError('Benchmark is bounded to 4-20 epochs per short run')
    base = load_config()
    study = yaml.safe_load((PACKAGE / 'configs/studies/core.yaml').read_text())
    candidates = {c['experiment']['name']: c for c in variants(base, study)}
    names = ['monitor_ct', 'monitor_sct', 'sct_derivative_frozen', 'sct_derivative_joint_scratch',
             'sct_forecasting_frozen', 'sct_forecasting_joint_scratch', 'sct_both_frozen', 'sct_both_joint_pretrained']
    root = project_path(base['output_root']) / args.study_id
    root.mkdir(parents=True, exist_ok=True)
    rows, parents = [], {}
    start = time.perf_counter()
    def execute(cfg, pretrain=False):
        cfg['study_id'] = args.study_id
        cfg['experiment']['role'] = 'smoke'
        cfg['training']['epochs'] = args.epochs
        cfg['dynamics']['pretrain_epochs'] = args.epochs
        directory = run_training(cfg, pretrain=pretrain)
        result = json.loads((directory / 'metrics.json').read_text())
        rows.append({'name': 'ode_pretrain' if pretrain else cfg['experiment']['name'],
                     'fold': cfg['experiment']['fold'], 'epochs': args.epochs,
                     'steady_epoch_seconds': result['steady_epoch_seconds'],
                     'wall_seconds': result['wall_seconds'], 'peak_cuda_memory_mb': result['peak_cuda_memory_mb'],
                     'reload_verified': result['reload_verified'], 'run_dir': str(directory)})
        pd.DataFrame(rows).to_csv(root / 'timing_runs.csv', index=False)
        return directory
    for fold, holdout in [('fold_a', 8), ('fold_b', 14)]:
        cfg = copy.deepcopy(base)
        cfg['experiment'].update(fold=fold, holdout=holdout)
        parents[fold] = execute(cfg, pretrain=True) / 'last.pt'
    for name, fold, holdout in [(name, 'fold_a', 8) for name in names] + [('sct_both_joint_scratch', 'fold_b', 14)]:
        cfg = copy.deepcopy(candidates[name])
        cfg['experiment'].update(fold=fold, holdout=holdout)
        if cfg['dynamics']['mode'] in ('frozen', 'joint_pretrained'):
            cfg['dynamics']['checkpoint'] = str(parents[fold])
        execute(cfg)
    frame = pd.DataFrame(rows)
    rates = frame[frame.fold == 'fold_a'].set_index('name').steady_epoch_seconds.to_dict()
    # Joint-from-scratch and joint-pretrained have the same per-step graph.
    b_rate = frame[(frame.fold == 'fold_b') & (frame.name == 'sct_both_joint_scratch')].steady_epoch_seconds.iloc[0]
    fold_factor = max(1., b_rate / rates['sct_both_joint_pretrained'])
    def rate(cfg):
        mode, name = cfg['dynamics']['mode'], cfg['experiment']['name']
        if mode == 'none':
            return rates['monitor_sct' if 'sensor' in cfg['monitor']['inputs'] else 'monitor_ct']
        loss = cfg['loss']
        objective = 'both' if loss['derivative_weight'] and loss['forecasting_weight'] else ('derivative' if loss['derivative_weight'] else 'forecasting')
        suffix = 'frozen' if mode == 'frozen' else ('joint_pretrained' if objective == 'both' else 'joint_scratch')
        return rates[f'sct_{objective}_{suffix}']
    per_epoch_sum = sum(rate(cfg) for cfg in candidates.values())
    # 14 inner fits/config at 200 epochs; 10 refits/config at selected E <= 200.
    inner_seconds = per_epoch_sum * 14 * base['training']['epochs']
    pretrain_seconds = frame[frame.name == 'ode_pretrain'].steady_epoch_seconds.max() * 16 * base['dynamics']['pretrain_epochs']
    overhead = max(0., float((frame.wall_seconds-frame.epochs*frame.steady_epoch_seconds).median())) * 592
    scenarios = []
    for refit_epochs in [50, 100, 200]:
        estimate = inner_seconds + per_epoch_sum*10*refit_epochs + pretrain_seconds + overhead
        scenarios.append({'assumed_final_epochs': refit_epochs, 'estimated_hours': estimate/3600,
                          'conservative_hours': (estimate*fold_factor*1.5)/3600})
    report = {'actual_short_jobs': len(rows), 'epochs_per_job': args.epochs,
              'benchmark_wall_seconds': time.perf_counter()-start, 'planned_full_tasks': 592,
              'inner_epochs': base['training']['epochs'], 'pretrain_epochs': base['dynamics']['pretrain_epochs'],
              'fold_runtime_factor': float(fold_factor), 'scenarios': scenarios,
              'limitations': ['Timing only; smoke accuracy is not a scientific result.',
                             'Full-batch MLP 64/32, RK4=20, one GPU, sequential execution.',
                             'Only two inner holdouts measured; other splits and background load vary.',
                             'Intervals are engineering allowances, not statistical confidence intervals.',
                             'No architecture/loss-weight sweep or publication plotting time included.']}
    write_json(root / 'time_estimate.json', report)
    text = '# 少量运行验证与耗时估计\n\n仅在开发池内验证；未使用外层测试集，未启动 592 次正式研究任务。\n\n'
    text += f'完成 {len(rows)} 次短训练，每次 {args.epochs} epoch；所有保存重载检查通过。实际总耗时 {report["benchmark_wall_seconds"]:.1f} 秒。\n\n'
    text += '|最终 refit 假设轮数|基础估计（小时）|保守预算（小时）|\n|---:|---:|---:|\n'
    for s in scenarios:
        text += f'|{s["assumed_final_epochs"]}|{s["estimated_hours"]:.2f}|{s["conservative_hours"]:.2f}|\n'
    text += '\n内部验证每次最多 200 epoch，ODE 预训练每次 100 epoch；最终轮数需由内部验证决定。预算不是保证，未包含新增超参数扫描。\n\n'
    text += '完整逐运行计时见 timing_runs.csv；详细公式、设备限制和预算见 time_estimate.json。\n'
    (root / 'BENCHMARK.md').write_text(text, encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
