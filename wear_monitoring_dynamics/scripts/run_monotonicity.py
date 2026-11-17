"""Five fixed monitors x four losses. Reuse baseline runs; never select architectures."""
import argparse
import ast
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
from ..evaluation.monotonicity import per_case_monotonicity, epoch_monotonicity
from .run_fixed_conditioning import MODELS, normalized_config

STUDY = 'trajectory_monotonicity_20260923'
BASELINE = 'fixed_conditioning_20260923'
MODES = {'supervised': '', 'mono': '_mono', 'ode': '_forecast', 'ode_mono': '_forecast_mono'}
NEW_MODULES = {'wear_monitoring_dynamics/losses/monotonicity.py', 'wear_monitoring_dynamics/evaluation/monotonicity.py',
               'wear_monitoring_dynamics/models/feature_subset.py'}
TRAINER = 'wear_monitoring_dynamics/training/trainer.py'


def strip_extension(source):
    """Reconstruct original trainer AST by removing explicitly delimited additions."""
    lines, active, count = [], False, 0
    for line in source.splitlines(keepends=True):
        if line.strip() == '# MONOTONICITY_EXTENSION_BEGIN':
            if active:
                raise ValueError('Nested extension marker')
            active, count = True, count+1
        elif line.strip() == '# MONOTONICITY_EXTENSION_END':
            if not active:
                raise ValueError('Unmatched extension marker')
            active = False
        elif not active:
            lines.append(line)
    if active or count != 3:
        raise ValueError('Unexpected trainer extension layout')
    return ''.join(lines)


def verify_compatible_source(archive):
    result = {'identical_modules': [], 'trainer_change': 'three optional monotonicity blocks; old AST preserved; zero-weight parity tested',
              'new_modules': sorted(NEW_MODULES)}
    with zipfile.ZipFile(archive) as snapshot:
        files = [p for p in PACKAGE.rglob('*.py') if p.relative_to(PACKAGE).parts[0] not in ('scripts', 'tests', 'outputs')]
        files.append(PROJECT/'release/CondWear-ODE/models/vb_sensor_condition_ode.py')
        for path in files:
            name = path.relative_to(PROJECT).as_posix()
            if name in NEW_MODULES:
                continue
            old = snapshot.read(name)
            if name == TRAINER:
                assert ast.dump(ast.parse(strip_extension(path.read_text(encoding='utf-8')))) == ast.dump(ast.parse(old.decode('utf-8')))
            elif name == 'wear_monitoring_dynamics/models/registry.py':
                # Legacy configs do not activate the optional feature-subset branch.
                addition = "    if m.get('selected_features') is not None:\n        from .feature_subset import FeatureSubsetMonitor\n        monitor = FeatureSubsetMonitor(monitor, m['architecture'], m['selected_features'])\n"
                current = path.read_text(encoding='utf-8')
                assert current.count(addition) == 1
                assert ast.dump(ast.parse(current.replace(addition, ''))) == ast.dump(ast.parse(old.decode('utf-8')))
            else:
                assert path.read_bytes() == old, f'Changed execution module: {name}'
                result['identical_modules'].append(name)
    return result


def specification():
    variants = []
    for name, _, architecture, inputs in MODELS:
        for mode, suffix in MODES.items():
            cfg = {'experiment': {'name': name+suffix}, 'monitor': {'architecture': architecture, 'inputs': inputs},
                   'dynamics': {'mode': 'joint_pretrained' if mode.startswith('ode') else 'none'},
                   'loss': {'derivative_weight': 0., 'forecasting_weight': float(mode.startswith('ode'))}}
            if 'mono' in mode:
                cfg['monotonicity'] = {'weight': .1, 'tolerance_z': 0.}
            variants.append(cfg)
    return dict(study_id=STUDY, family='explicit', variants=variants, folds=['fold_a', 'fold_b'], seeds=list(range(5)), inner_seed=0)


def reuse_catalog(base):
    output = project_path(base['output_root'])
    root = output/BASELINE
    assert json.loads((root/'summary/completion.json').read_text())['all_checks_passed']
    manifest = pd.read_csv(root/'study_manifest.csv')
    assert len(manifest) == 256 and (manifest.status == 'completed').all()
    _, signature = load_table(project_path(base['data']['path']), project_path(base['data']['cache']), base['data']['raw_sensor_abs_limit'])
    catalog, sources, checks = {}, {}, []
    for run in manifest.itertuples():
        directory = Path(run.run_dir)
        identity = json.loads((directory/'provenance.json').read_text())
        cfg = yaml.safe_load((directory/'config_resolved.yaml').read_text(encoding='utf-8'))
        status = json.loads((directory/'status.json').read_text())
        assert status['status'] == 'completed' and status['run_id'] == digest(identity)[:12]
        assert cfg == identity['config'] and not cfg.get('monotonicity', {}).get('weight', 0.)
        assert identity['data_signature'] == signature
        assert (identity['torch'], identity['numpy'], identity['python'], identity['device_name']) == (str(torch.__version__), str(np.__version__), platform.python_version(), torch.cuda.get_device_name(0))
        source = identity['source_hash']
        if source not in sources:
            sources[source] = verify_compatible_source(output/'sources'/f'{source}.zip')
        pretrain = run.type == 'pretrain'
        if cfg['dynamics']['checkpoint'] and not pretrain:
            assert identity['parent_hash'] == file_hash(cfg['dynamics']['checkpoint'])
        key = digest({'config': normalized_config(cfg, pretrain), 'pretrain': pretrain})
        assert key not in catalog
        catalog[key] = directory
        checks.append(dict(task_id=run.task_id, run_dir=str(directory), source_hash=source, match_key=key))
    return catalog, dict(data_signature=signature, source_checks=sources, runs=checks)


def report(root):
    from .analyze_completed import analyze
    analyze(root)
    out = root/'summary'
    manifest = pd.read_csv(root/'study_manifest.csv')
    cases = pd.read_csv(out/'per_case_metrics.csv')
    runs = pd.read_csv(out/'run_metrics.csv')
    diagnostics, epoch_rows, forecast_rows = [], [], []
    for run in manifest[manifest.type == 'monitor'].itertuples():
        directory = Path(run.run_dir)
        ep = epoch_monotonicity(directory/'epoch_predictions.csv.gz')
        ep['experiment'], ep['fold'], ep['role'], ep['seed'], ep['task_id'] = run.experiment, run.fold, run.role, run.seed, run.task_id
        epoch_rows.append(ep)
        if run.role == 'final':
            frame = pd.read_csv(directory/'monitoring_predictions.csv')
            diag = per_case_monotonicity(frame)
            diag['experiment'], diag['fold'], diag['seed'] = run.experiment, run.fold, run.seed
            diagnostics.append(diag)
            metrics = json.loads((directory/'metrics.json').read_text())
            for horizon, values in metrics.get('forecasting_by_horizon', {}).items():
                forecast_rows.append(dict(experiment=run.experiment, fold=run.fold, seed=run.seed, horizon=horizon, **values))
    mono = pd.concat(diagnostics, ignore_index=True)
    mono.to_csv(out/'monotonicity_per_case.csv', index=False)
    epoch_frame = pd.concat(epoch_rows, ignore_index=True)
    epoch_frame.to_csv(out/'epoch_monotonicity.csv.gz', index=False, compression='gzip')
    epoch_frame[epoch_frame.role == 'final'].groupby(['experiment','fold','epoch','split']).agg(
        violation_rate=('violation_rate','mean'), mean_drop_mm=('mean_drop_mm','mean'), max_drop_mm=('max_drop_mm','mean')).to_csv(out/'final_epoch_monotonicity_5seed.csv')
    matrix, long, paired = [], [], []
    for name, label, _, _ in MODELS:
        matrix_row = dict(model=name, label=label)
        for mode, suffix in MODES.items():
            experiment = name+suffix
            c = cases[cases.experiment == experiment]
            d = mono[mono.experiment == experiment]
            result = dict(model=name, label=label, mode=mode, experiment=experiment,
                          case_macro_mae=float(c.mae.mean()), rmse=float(runs[runs.experiment == experiment].rmse.mean()),
                          pairs=int(d.pairs.sum()), decreases=int(d.decreases.sum()),
                          violation_rate=float(d.decreases.sum()/d.pairs.sum()),
                          case_macro_violation_rate=float(d.violation_rate.mean()),
                          mean_drop_mm=float((d.mean_drop_mm*d.pairs).sum()/d.pairs.sum()),
                          maximum_drop_mm=float(d.max_drop_mm.max()))
            for fold in ('fold_a','fold_b'):
                result[fold+'_mae'] = float(c[c.fold == fold].mae.mean())
            long.append(result)
            matrix_row[mode+'_mae'] = result['case_macro_mae']
            matrix_row[mode+'_violation_rate'] = result['violation_rate']
        matrix.append(matrix_row)
        for left, right in [('mono','supervised'), ('ode_mono','ode'), ('ode','supervised'), ('ode_mono','mono')]:
            a = cases[cases.experiment == name+MODES[left]].groupby(['fold','case']).mae.mean()
            b = cases[cases.experiment == name+MODES[right]].groupby(['fold','case']).mae.mean()
            paired.append(dict(model=name, label=label, left=left, right=right, **paired_case_bootstrap(a,b)))
    pd.DataFrame(matrix).to_csv(out/'monotonicity_matrix.csv', index=False)
    pd.DataFrame(long).to_csv(out/'monitoring_and_monotonicity.csv', index=False)
    pd.DataFrame(paired).to_csv(out/'within_network_comparisons.csv', index=False)
    pd.DataFrame(forecast_rows).to_csv(out/'forecasting_run_metrics.csv', index=False)
    pd.DataFrame(forecast_rows).groupby(['experiment','horizon']).mean(numeric_only=True).drop(columns='seed').to_csv(out/'forecasting_summary.csv')
    curves = pd.read_csv(out/'final_epoch_curves_5seed.csv')
    curves[curves.epoch == 500].groupby('experiment').mean_test_case_macro_mae.mean().to_csv(out/'fixed_epoch_500_diagnostic.csv')
    completed = json.loads((out/'audit_summary.json').read_text())
    fresh = manifest[manifest.origin == 'new']
    completed.update(new_runs=len(fresh), reused_runs=int((manifest.origin == 'reused').sum()),
                     new_training_seconds=sum(json.loads((Path(p)/'metrics.json').read_text())['wall_seconds'] for p in fresh.run_dir))
    write_json(out/'completion.json', completed)
    text = '# 轨迹单调约束：完整固定网络消融\n\n'
    text += f'496 次运行完成并审计：新增 {len(fresh)} 次，复用 {completed["reused_runs"]} 次。新增训练 {completed["new_training_seconds"]/3600:.2f} 小时。\n\n'
    text += '固定 λ_mono=0.1、标准化 VB、容忍量 0；只惩罚同一训练 case 相邻有效标注时刻的预测下降，不读取标签方向，不跨 case。是软约束，不保证测试单调。\n\n'
    text += '## 当前 VB case-macro MAE（mm）\n\n|固定网络|纯监督|＋单调|＋ODE|＋ODE＋单调|\n|---|---:|---:|---:|---:|\n'
    for r in matrix:
        text += f'|{r["label"]}|{r["supervised_mae"]:.5f}|{r["mono_mae"]:.5f}|{r["ode_mae"]:.5f}|{r["ode_mono_mae"]:.5f}|\n'
    text += '\n## 相邻当前预测下降比例\n\n|固定网络|纯监督|＋单调|＋ODE|＋ODE＋单调|\n|---|---:|---:|---:|---:|\n'
    for r in matrix:
        text += f'|{r["label"]}|{r["supervised_violation_rate"]:.1%}|{r["mono_violation_rate"]:.1%}|{r["ode_violation_rate"]:.1%}|{r["ode_mono_violation_rate"]:.1%}|\n'
    text += '\n上述均采用各配置内部验证独立选 epoch、双折、五个 final seed；不选择或淘汰网络。下降统计容差仅 1e-8 mm（数值精度），620 对预测来自 124 对观测 × 5 seed，不是 620 个独立样本。mean_drop_mm 对所有相邻对平均（含零），maximum_drop_mm 为所有 case/seed 的最大下降。不能只用单调性而忽略准确性。\n\n'
    text += '## 文件导航与限制\n\n- `summary/monotonicity_matrix.csv`：五行四列主表。\n- `summary/monitoring_and_monotonicity.csv`：MAE、RMSE、分折成绩、下降比例及幅度。\n- `summary/within_network_comparisons.csv`：同网络配对 case-bootstrap（10,000 次，先平均 seed，95% 区间未校正多重比较）。\n- `summary/epoch_monotonicity.csv.gz`：全部监测运行每个 epoch 的训练/验证/测试轨迹诊断，从保存的预测统一计算。\n- `summary/final_epoch_monotonicity_5seed.csv`、`final_epoch_curves_5seed.csv`：最终模型全程曲线。\n- `summary/inner_cv_selected_epochs.csv`、`fixed_epoch_500_diagnostic.csv`：验证选轮数与固定预算终点的次要诊断。\n- `summary/forecasting_summary.csv`：未来 VB 的平均 fold/seed pooled MAE，独立于当前 VB 主指标。\n- `summary/completion.json`、`reuse_audit.json`、`study_manifest.csv`：审计、耗时、复用和路径。\n\n'
    text += '固定 λ 只检验一个约束强度，不声称最佳权重。未新增时间偏导非负项或硬单调结构；没有对预测作累积最大值/事后修正。沿用既有测试池与 500 epoch 预算，需注意 epoch 选择及边界，不把测试最优当主成绩。旧 baseline 的实际预测和 checkpoint 保持原样。\n'
    (root/'README.md').write_text(text,encoding='utf-8')


def smoke(base, catalog):
    rows=[]
    spec=specification()
    parent_cfg=copy.deepcopy(base)
    parent_cfg['experiment'].update(name='ode_pretrain',fold='fold_a',role='inner',holdout=8)
    parent=catalog[digest({'config':normalized_config(parent_cfg,True),'pretrain':True})]
    from ..study import variants
    for cfg in variants(base,spec):
        if not cfg.get('monotonicity',{}).get('weight'):
            continue
        cfg['study_id']='trajectory_monotonicity_smoke_20260923'
        cfg['training']['epochs']=20
        if cfg['dynamics']['mode']!='none':
            cfg['dynamics']['checkpoint']=str(parent/'last.pt')
        directory=run_training(cfg)
        metrics=json.loads((directory/'metrics.json').read_text())
        rows.append(dict(experiment=cfg['experiment']['name'], seconds=metrics['wall_seconds'], steady_epoch_seconds=metrics['steady_epoch_seconds'], run_dir=str(directory)))
    output=project_path(base['output_root'])/'trajectory_monotonicity_smoke_20260923'
    pd.DataFrame(rows).to_csv(output/'timing.csv',index=False)
    print(pd.DataFrame(rows).drop(columns='run_dir').to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    base=load_config(PACKAGE/'configs/time_conditioning.yaml')
    root=project_path(base['output_root'])/STUDY
    root.mkdir(parents=True,exist_ok=True)
    catalog,audit=reuse_catalog(base)
    write_json(root/'reuse_audit.json',audit)
    if args.smoke:
        smoke(base,catalog)
        return
    spec=specification()
    tasks=build_tasks(base,spec)
    assert len(tasks)==496
    assert sum(t['type']=='monitor' and t['config'].get('monotonicity',{}).get('weight',0)>0 for t in tasks)==240
    write_json(root/'planned_tasks.json',tasks)
    (root/'study_spec.yaml').write_text(yaml.safe_dump(spec,sort_keys=False),encoding='utf-8')
    print('496 tasks: 256 audited baseline/pretrain runs, 240 new monotonicity runs.',flush=True)
    if not args.execute:
        return
    write_json(root/'status.json',{'status':'running'})
    completed,rows={},[]
    for task in tasks:
        cfg=copy.deepcopy(task['config'])
        parents=[key for key in task['dependencies'] if key.startswith('ode/')]
        if parents:
            cfg['dynamics']['checkpoint']=str(completed[parents[0]]/'last.pt')
        if task.get('inner_ids'):
            epoch,curve=select_epoch([completed[k] for k in task['inner_ids']])
            cfg['training']['selection_epoch']=epoch
            dest=root/'epoch_selection'/cfg['experiment']['name']
            dest.mkdir(parents=True,exist_ok=True)
            curve.to_csv(dest/f'{cfg["experiment"]["fold"]}_inner_mean.csv',header=['case_macro_mae'])
            write_json(dest/f'{cfg["experiment"]["fold"]}.json',{'selected_epoch':epoch,'inner_runs':[str(completed[k]) for k in task['inner_ids']]})
        key=digest({'config':normalized_config(cfg,task['type']=='pretrain'),'pretrain':task['type']=='pretrain'})
        origin='reused' if key in catalog else 'new'
        # Fail rather than silently retraining a baseline whose compatibility differs.
        assert (origin=='new') == bool(cfg.get('monotonicity',{}).get('weight',0))
        directory=catalog[key] if origin=='reused' else run_training(cfg,pretrain=task['type']=='pretrain')
        completed[task['id']]=directory
        exp=cfg['experiment']
        rows.append(dict(task_id=task['id'],type=task['type'],experiment=exp['name'],fold=exp['fold'],role=exp['role'],seed=exp['seed'],status='completed',origin=origin,run_dir=str(directory)))
        pd.DataFrame(rows).to_csv(root/'study_manifest.csv',index=False)
    assert sum(r['origin']=='new' for r in rows)==240
    report(root)
    write_json(root/'status.json',{'status':'completed','new_runs':240,'reused_runs':256,'audit_passed':True})
    print('COMPLETE: all monotonicity experiments, diagnostics and audits.',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write_json(project_path('wear_monitoring_dynamics/outputs')/STUDY/'status.json',{'status':'failed','error':str(error)})
        raise
