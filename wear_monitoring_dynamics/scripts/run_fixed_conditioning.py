"""Complete the fixed architecture x supervision matrix, with audited reuse.

Existing runs stay in their original directories. Only bookkeeping differences
may be ignored when matching a run; training config, runtime, source modules,
data signature, parent checkpoint and CV-selected epoch must agree.
"""
import argparse
import copy
import json
import platform
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml
from ..config import PACKAGE, PROJECT, load_config, project_path, write_json, digest, file_hash
from ..data.records import load_table
from ..study import build_tasks, select_epoch
from ..training.trainer import run_training
from ..evaluation.bootstrap import paired_case_bootstrap

STUDY = 'fixed_conditioning_20260923'
MODELS = [
    ('time_ct_mlp', '工况＋时间，无 sensor', 'mlp', ['condition', 'time']),
    ('time_sct_mlp', '当前拼接 MLP', 'mlp', ['sensor', 'condition', 'time']),
    ('time_sct_encoded', '时间编码后拼接 MLP', 'time_encoded', ['sensor', 'condition', 'time']),
    ('time_sct_film', '时间＋工况特征调制（FiLM）', 'film', ['sensor', 'condition', 'time']),
    ('time_sct_attention', '时间＋工况 query 的传感器 attention', 'sensor_attention', ['sensor', 'condition', 'time']),
]


def specification():
    variants = []
    for name, label, architecture, inputs in MODELS:
        for forecast in (False, True):
            variants.append({'experiment': {'name': name + ('_forecast' if forecast else '')},
                             'monitor': {'architecture': architecture, 'inputs': inputs},
                             'dynamics': {'mode': 'joint_pretrained' if forecast else 'none'},
                             'loss': {'derivative_weight': 0., 'forecasting_weight': float(forecast)}})
    return dict(study_id=STUDY, family='explicit', variants=variants,
                folds=['fold_a', 'fold_b'], seeds=list(range(5)), inner_seed=0)


def normalized_config(config, pretrain=False):
    cfg = copy.deepcopy(config)
    for key in ('study_id', 'output_root'):
        cfg.pop(key, None)
    cfg['experiment'].pop('name', None)
    if pretrain:
        cfg['training']['epochs'] = cfg['dynamics']['pretrain_epochs']
        cfg['experiment']['seed'] = cfg['dynamics']['pretrain_seed']
    parent = cfg['dynamics'].get('checkpoint')
    cfg['dynamics']['checkpoint'] = file_hash(parent) if parent and not pretrain else None
    return cfg


def verify_source(archive):
    """Allow orchestration/tests to change; all execution modules must be identical."""
    checked = []
    with zipfile.ZipFile(archive) as snapshot:
        names = set(snapshot.namelist())
        current = [p for p in PACKAGE.rglob('*.py')
                   if p.relative_to(PACKAGE).parts[0] not in ('scripts', 'tests', 'outputs')]
        current.append(PROJECT/'release/CondWear-ODE/models/vb_sensor_condition_ode.py')
        for path in current:
            name = path.relative_to(PROJECT).as_posix()
            if name not in names or snapshot.read(name) != path.read_bytes():
                raise ValueError(f'Reuse rejected: changed execution source {name}')
            checked.append(name)
    return checked


def reuse_catalog(base):
    output = project_path(base['output_root'])
    _, signature = load_table(project_path(base['data']['path']), project_path(base['data']['cache']), base['data']['raw_sensor_abs_limit'])
    catalog, checks, verified = {}, [], {}
    for study in ('time_conditioning_20260922', 'time_conditioning_ode_20260922'):
        root = output/study
        assert json.loads((root/'summary/audit_summary.json').read_text())['all_checks_passed']
        manifest = pd.read_csv(root/'study_manifest.csv')
        assert (manifest.status == 'completed').all()
        for run in manifest.itertuples():
            directory = Path(run.run_dir)
            identity = json.loads((directory/'provenance.json').read_text())
            status = json.loads((directory/'status.json').read_text())
            cfg = yaml.safe_load((directory/'config_resolved.yaml').read_text(encoding='utf-8'))
            assert status['status'] == 'completed' and status['run_id'] == digest(identity)[:12]
            assert cfg == identity['config']
            assert identity['data_signature'] == signature
            assert identity['torch'] == str(torch.__version__) and identity['numpy'] == str(np.__version__)
            assert identity['python'] == platform.python_version()
            assert identity['device_name'] == torch.cuda.get_device_name(0)
            source = identity['source_hash']
            if source not in verified:
                verified[source] = verify_source(output/'sources'/f'{source}.zip')
            pretrain = run.type == 'pretrain'
            key = digest({'config': normalized_config(cfg, pretrain), 'pretrain': pretrain})
            if not pretrain and cfg['dynamics']['checkpoint']:
                assert identity['parent_hash'] == file_hash(cfg['dynamics']['checkpoint'])
            if key in catalog:
                raise ValueError('Ambiguous reused config')
            catalog[key] = directory
            checks.append({'source_study': study, 'source_task_id': run.task_id,
                           'run_dir': str(directory), 'config_match_key': key, 'source_hash': source})
    return catalog, dict(data_signature=signature, execution_source_checks=verified, runs=checks)


def report(root):
    from .analyze_completed import analyze
    analyze(root)
    out = root/'summary'
    cases = pd.read_csv(out/'per_case_metrics.csv')
    metrics = pd.read_csv(out/'run_metrics.csv')
    matrix, comparisons, forecasts = [], [], []
    for name, label, _, _ in MODELS:
        before = cases[cases.experiment == name].groupby(['fold', 'case']).mae.mean()
        after = cases[cases.experiment == name+'_forecast'].groupby(['fold', 'case']).mae.mean()
        paired = paired_case_bootstrap(after, before)
        comparisons.append({'model': name, 'label': label, 'left': name+'_forecast', 'right': name, **paired})
        row = {'model': name, 'label': label, 'supervised_mae': before.mean(), 'ode_mae': after.mean(),
               'reduction_percent': (1-after.mean()/before.mean())*100, **paired}
        for fold in ('fold_a', 'fold_b'):
            row[fold+'_supervised_mae'] = before.loc[fold].mean()
            row[fold+'_ode_mae'] = after.loc[fold].mean()
        for suffix, mode in (('', 'supervised'), ('_forecast', 'ode')):
            row[mode+'_rmse'] = metrics[metrics.experiment == name+suffix].rmse.mean()
        matrix.append(row)
    pd.DataFrame(matrix).to_csv(out/'fixed_network_matrix.csv', index=False)
    pd.DataFrame(comparisons).to_csv(out/'ode_within_network_comparisons.csv', index=False)
    manifest = pd.read_csv(root/'study_manifest.csv')
    for run in manifest[(manifest.type == 'monitor') & (manifest.role == 'final')].itertuples():
        m = json.loads((Path(run.run_dir)/'metrics.json').read_text())
        for horizon, values in m.get('forecasting_by_horizon', {}).items():
            forecasts.append({'experiment': run.experiment, 'fold': run.fold, 'seed': run.seed, 'horizon': horizon, **values})
    pd.DataFrame(forecasts).to_csv(out/'forecasting_run_metrics.csv', index=False)
    if forecasts:
        pd.DataFrame(forecasts).groupby(['experiment', 'horizon']).mean(numeric_only=True).drop(columns='seed').to_csv(out/'forecasting_summary.csv')
    audit = json.loads((out/'audit_summary.json').read_text())
    fresh = manifest[manifest.origin == 'new']
    seconds = sum(json.loads((Path(p)/'metrics.json').read_text())['wall_seconds'] for p in fresh.run_dir)
    audit.update(new_runs=len(fresh), reused_runs=int((manifest.origin == 'reused').sum()), new_training_seconds=seconds)
    write_json(out/'completion.json', audit)
    text = '# 固定网络 × ODE 完整消融\n\n'
    text += f'全部 {len(manifest)} 次运行完成并审计，其中新增 {len(fresh)} 次，复用 {len(manifest)-len(fresh)} 次；新增训练 {seconds/60:.1f} 分钟。\n\n'
    text += '每行在两个外层 fold 中使用同一个固定网络；每个配置独立用七个内层验证 case 选 epoch，最终五个 seed。以下为当前 VB 的 case-macro MAE（mm），不是未来预测误差。\n\n'
    text += '|固定监测网络|纯监督|＋ODE 未来监督|平均误差下降|ODE−纯监督 95% 区间|\n|---|---:|---:|---:|---|\n'
    for r in matrix:
        text += f'|{r["label"]}|{r["supervised_mae"]:.5f}|{r["ode_mae"]:.5f}|{r["reduction_percent"]:.1f}%|[{r["ci_low"]:.5f}, {r["ci_high"]:.5f}]|\n'
    text += '\n区间先平均各 case 的五个 seed，再对 14 case 做 10,000 次配对 bootstrap；未校正多重比较。负差值表示 ODE 改善，跨零时不应声称稳定收益。\n\n'
    text += '协议保持 500 epoch、full-batch AdamW、lr .001、weight decay .0001；ODE 为 split-matched 预训练后 joint training，forecast loss 权重 1，h1/2/3，RK4 20 步；不加 derivative loss。测试诊断不选模型、不选 epoch。既有测试池已被多轮研究使用；本次是完整消融补充，不是新独立测试。\n\n'
    text += '## 结果文件\n\n- `summary/fixed_network_matrix.csv`：五行两列主表，含分 fold、RMSE、配对区间。\n- `summary/ode_within_network_comparisons.csv`：同网络加 ODE 前后。\n- `summary/paired_comparisons.csv`：所有网络间探索性比较。\n- `summary/forecasting_summary.csv`：未来 VB 预测独立汇总，不混入主指标。\n- `summary/inner_cv_selected_epochs.csv`、`final_epoch_curves_5seed.csv`：所选轮数与全程诊断。\n- `summary/audit_summary.json`、`completion.json`：全部 epoch、预测和 checkpoint 审计。\n- `reuse_audit.json`、`study_manifest.csv`：复用核验与原始路径；旧结果没有复制或冒充新训练。\n\n'
    text += '每次运行的 `history.csv`、`epoch_predictions.csv.gz` 和 `epochs/` 仍在 manifest 指向的原始目录。完整主表不再合并不同网络为“验证选出的新结构”。\n'
    (root/'README.md').write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    base = load_config(PACKAGE/'configs/time_conditioning.yaml')
    root = project_path(base['output_root'])/STUDY
    root.mkdir(parents=True, exist_ok=True)
    spec = specification()
    tasks = build_tasks(base, spec)
    catalog, audit = reuse_catalog(base)
    write_json(root/'reuse_audit.json', audit)
    write_json(root/'planned_tasks.json', tasks)
    (root/'study_spec.yaml').write_text(yaml.safe_dump(spec, sort_keys=False), encoding='utf-8')
    write_json(root/'status.json', {'status': 'running' if args.execute else 'validated_plan'})
    # Independently count all configurations ignoring only selection_epoch, which
    # is determined later using validation curves. No test scores are consulted.
    old_shapes = []
    for directory in catalog.values():
        cfg = yaml.safe_load((directory/'config_resolved.yaml').read_text(encoding='utf-8'))
        cfg['training'].pop('selection_epoch', None)
        cfg['dynamics']['checkpoint'] = None
        old_shapes.append(digest(normalized_config(cfg, cfg['experiment']['name'] == 'ode_pretrain')))
    reusable = 0
    for task in tasks:
        cfg = copy.deepcopy(task['config'])
        cfg['dynamics']['checkpoint'] = None
        reusable += digest(normalized_config(cfg, task['type'] == 'pretrain')) in old_shapes
    assert len(tasks) == 256 and reusable == 208
    print(f'Validated 256 tasks: {reusable} reusable, {len(tasks)-reusable} new.', flush=True)
    if not args.execute:
        return
    completed, rows = {}, []
    for task in tasks:
        cfg = copy.deepcopy(task['config'])
        parents = [key for key in task['dependencies'] if key.startswith('ode/')]
        if parents:
            cfg['dynamics']['checkpoint'] = str(completed[parents[0]]/'last.pt')
        if task.get('inner_ids'):
            epoch, curve = select_epoch([completed[key] for key in task['inner_ids']])
            cfg['training']['selection_epoch'] = epoch
            dest = root/'epoch_selection'/cfg['experiment']['name']
            dest.mkdir(parents=True, exist_ok=True)
            curve.to_csv(dest/f'{cfg["experiment"]["fold"]}_inner_mean.csv', header=['case_macro_mae'])
            write_json(dest/f'{cfg["experiment"]["fold"]}.json', {'selected_epoch': epoch, 'inner_runs': [str(completed[key]) for key in task['inner_ids']]})
        key = digest({'config': normalized_config(cfg, task['type'] == 'pretrain'), 'pretrain': task['type'] == 'pretrain'})
        origin = 'reused' if key in catalog else 'new'
        directory = catalog[key] if origin == 'reused' else run_training(cfg, pretrain=task['type'] == 'pretrain')
        completed[task['id']] = directory
        exp = cfg['experiment']
        rows.append(dict(task_id=task['id'], type=task['type'], experiment=exp['name'], fold=exp['fold'], role=exp['role'],
                         seed=cfg['dynamics']['pretrain_seed'] if task['type'] == 'pretrain' else exp['seed'],
                         status='completed', origin=origin, run_dir=str(directory)))
        pd.DataFrame(rows).to_csv(root/'study_manifest.csv', index=False)
    assert sum(row['origin'] == 'new' for row in rows) == 48
    report(root)
    write_json(root/'status.json', {'status': 'completed', 'new_runs': 48, 'reused_runs': 208, 'audit_passed': True})
    print('COMPLETE: fixed matrix, all epoch audits and reports saved.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        write_json(project_path('wear_monitoring_dynamics/outputs')/STUDY/'status.json', {'status': 'failed', 'error': str(error)})
        raise
