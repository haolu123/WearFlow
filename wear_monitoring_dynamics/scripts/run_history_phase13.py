"""Phase 1 causal-history ablation and Phase 3 frozen-ODE diagnostics."""
import argparse
import copy
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..config import PACKAGE, load_config, project_path, write_json
from ..data.records import load_table
from ..data.splits import POOLS, make_split
from ..data.datasets import tensors, pair_tensors
from ..data.history import add_history
from ..models.registry import build_model, build_dynamics
from ..training.trainer import run_training
from ..training.checkpoint import load_checkpoint
from .run_train_only_coverage import FEATURES, rank_features

ROOT = project_path('wear_monitoring_dynamics/outputs/history_phase13_20261008')


def plan(root, quality=False):
    base = load_config(PACKAGE / 'configs/time_conditioning.yaml')
    base['study_id'] = root.name
    base['training'].update(fixed_final_epoch=True, selection_epoch=500)
    table, _ = load_table(project_path(base['data']['path']), project_path(base['data']['cache']))
    frame = pd.DataFrame({k: v for k, v in table.items() if k != 'sensor'})
    frame[FEATURES] = table['sensor']
    if quality:
        from ..data.sensor_quality import raw_quality_metrics,fit_rules,assess,VERSION
        raw_records=np.load(project_path(base['data']['path']),allow_pickle=True).tolist()
        quality_metrics=raw_quality_metrics(raw_records)
    definitions = [('ct', 'mlp', 1)] + [(f'{arch}_L{length}', 'history_'+arch, length)
        for arch in ['mlp', 'gru', 'cnn'] for length in [1, 3, 5, 10]]
    tasks, rankings = [], []
    root.mkdir(exist_ok=True)
    quality_decisions=[];quality_channels=[];quality_sensitivity=[]
    for fold in POOLS:
        for role, holdout in [('inner', h) for h in POOLS[fold]] + [('final', None)]:
            split = make_split(fold, role, holdout)
            if quality:
                rules=fit_rules(quality_metrics,split['train'])
                decisions,channel_details=assess(raw_records,quality_metrics,rules)
                invalid=decisions[~decisions.sensor_valid][['case','run']].astype(int).values.tolist()
                quality_cfg=dict(version=VERSION,rules=rules,invalid_records=invalid,
                                 fallback='frozen_condition_time',retain_previous_hard_exclusions=True)
                folder=root/'quality_rules';folder.mkdir(exist_ok=True)
                write_json(folder/f'{fold}_{role}_{holdout}.json',quality_cfg)
                quality_decisions.append(decisions.assign(fold=fold,role=role,holdout=holdout))
                quality_channels.append(channel_details.assign(fold=fold,role=role,holdout=holdout))
                for multiplier in [4.,8.]:
                    sensitivity,_=assess(raw_records,quality_metrics,fit_rules(quality_metrics,split['train'],multiplier))
                    quality_sensitivity.append(sensitivity.assign(fold=fold,role=role,holdout=holdout,multiplier=multiplier))
                valid_rows=np.array([(int(c),int(r)) not in {tuple(x) for x in invalid} for c,r in zip(frame.case,frame.run)])
            else:valid_rows=np.ones(len(frame),dtype=bool)
            ranking = rank_features(frame[frame.case.isin(split['train']) & valid_rows])
            ranking = ranking.assign(fold=fold, role=role, holdout=holdout)
            rankings.append(ranking)
            features = ranking.head(18).feature.tolist()
            for name, arch, length in definitions:
                for seed in ([0] if role == 'inner' else range(5)):
                    cfg = copy.deepcopy(base)
                    if quality:cfg['data']['sensor_quality']=quality_cfg
                    cfg['experiment'].update(name=name, fold=fold, role=role, holdout=holdout, seed=seed)
                    cfg['monitor'].update(architecture=arch, inputs=['condition', 'time'] if name=='ct' else ['sensor', 'condition', 'time'])
                    if name != 'ct': cfg['monitor'].update(history_length=length, selected_features=features)
                    tasks.append(dict(id=f'{fold}/{role}/{holdout}/{name}/{seed}', type='monitor', name=name,
                                      fold=fold, role=role, holdout=holdout, seed=seed, config=cfg))
        for seed in range(5):
            cfg = copy.deepcopy(base)
            if quality:cfg['data']['sensor_quality']=quality_cfg  # final-fold rules
            cfg['experiment'].update(name='standalone_ode', fold=fold, role='final', holdout=None, seed=seed)
            cfg['dynamics'].update(pretrain_epochs=500, pretrain_seed=seed)
            tasks.append(dict(id=f'{fold}/ode/{seed}', type='ode', name='standalone_ode', fold=fold,
                              role='final', holdout=None, seed=seed, config=cfg))
    assert len(tasks) == 322
    pd.concat(rankings).to_csv(root/'feature_rankings.csv', index=False)
    if quality:
        quality_metrics.to_csv(root/'quality_raw_metrics.csv',index=False)
        pd.concat(quality_decisions).to_csv(root/'quality_decisions.csv',index=False)
        pd.concat(quality_channels).to_csv(root/'quality_channel_flags.csv',index=False)
        pd.concat(quality_sensitivity).to_csv(root/'quality_threshold_sensitivity.csv',index=False)
    write_json(root/'planned_tasks.json', tasks)
    counts = []
    for name, arch, length in definitions:
        cfg = next(t['config'] for t in tasks if t['name']==name)
        counts.append(dict(name=name, parameters=sum(p.numel() for p in build_model(cfg).parameters())))
    pd.DataFrame(counts).to_csv(root/'parameter_counts.csv', index=False)
    return tasks


def phase1_report(root, manifest, epochs):
    inner, final, history = [], [], []
    for r in manifest[manifest.type=='monitor'].itertuples():
        d = Path(r.run_dir)
        m = json.loads((d/'metrics.json').read_text())
        assert m['selected_epoch']==epochs and m['reload_verified']
        h = pd.read_csv(d/'history.csv')
        assert h.epoch.tolist()==list(range(1, epochs+1))
        history.append(h.assign(name=r.name, fold=r.fold, role=r.role, holdout=r.holdout, seed=r.seed))
        if r.role=='inner':
            inner.append(dict(name=r.name, fold=r.fold, holdout=r.holdout, mae=m['monitoring']['case_macro_mae']))
        else:
            p = pd.read_csv(d/'monitoring_predictions.csv').assign(name=r.name, fold=r.fold, seed=r.seed)
            final.append(p)
    inner = pd.DataFrame(inner)
    if epochs == 500:
        assert len(inner) == 182 and len(final) == 130
        for (fold, name), group in inner.groupby(['fold','name']):
            assert set(group.holdout) == set(POOLS[fold])
    scores = inner.groupby(['fold','name']).mae.mean().reset_index()
    scores.to_csv(root/'inner_validation_scores.csv', index=False)
    selection = {}
    for fold, g in scores.groupby('fold'):
        # Select architecture AND L using only held-out development cases.
        eligible = g[g.name.str.endswith(('_L3', '_L5', '_L10'))]
        selection[fold] = eligible.sort_values(['mae','name']).iloc[0]['name']
    write_json(root/'history_selection.json', selection)
    predictions = pd.concat(final)
    predictions['absolute_error'] = abs(predictions.predicted_vb_mm-predictions.vb_mm)
    predictions.to_csv(root/'phase1_test_predictions.csv', index=False)
    case = predictions.groupby(['name','fold','seed','case']).absolute_error.mean().rename('mae').reset_index()
    case.to_csv(root/'phase1_case_metrics.csv', index=False)
    rows = []
    for name, g in predictions.groupby('name'):
        masks = dict(overall=np.ones(len(g), bool), fold_a=g.fold=='fold_a', fold_b=g.fold=='fold_b',
                     fold_a_in_range=(g.fold=='fold_a')&(g.vb_mm<=.74),
                     fold_a_out_of_range=(g.fold=='fold_a')&(g.vb_mm>.74))
        for subset, mask in masks.items():
            q = g.loc[mask]
            rows.append(dict(name=name, subset=subset,
                mae=q.groupby(['fold','seed','case']).absolute_error.mean().mean(),
                pooled_rmse=float(np.sqrt(((q.predicted_vb_mm-q.vb_mm)**2).mean()))))
    summary = pd.DataFrame(rows); summary.to_csv(root/'phase1_summary.csv', index=False)
    pd.concat(history).to_csv(root/'phase1_epoch_metrics.csv.gz', index=False)
    return selection, summary


def load_monitor(directory, table, device):
    d = Path(directory)
    cfg = json.loads((d/'provenance.json').read_text())['config']
    scaler = json.loads((d/'scalers.json').read_text())
    from ..data.sensor_quality import apply_quality
    from ..data.monitor_batch import make_monitor_batch
    table=apply_quality(table,cfg['data'].get('sensor_quality'))
    batch=make_monitor_batch(cfg,table,scaler,device)
    model = build_model(cfg).to(device)
    load_checkpoint(d/'last.pt', model, device); model.eval()
    with torch.no_grad():
        z = model.estimate(batch, np.arange(len(table['case']))).flatten().cpu().numpy()
    return z*scaler['wear'][1]+scaler['wear'][0]


def phase3(root, manifest, selection):
    first = Path(manifest.iloc[0].run_dir)
    cfg = json.loads((first/'provenance.json').read_text())['config']
    table, _ = load_table(project_path(cfg['data']['path']), project_path(cfg['data']['cache']))
    device = torch.device('cuda'); torch.set_num_threads(1)
    forecasts, trajectories = [], []
    for run in manifest[manifest.type=='ode'].itertuples():
        d = Path(run.run_dir)
        cfg = json.loads((d/'provenance.json').read_text())['config']
        scaler = json.loads((d/'scalers.json').read_text())
        split = json.loads((d/'split.json').read_text())
        model = build_dynamics(cfg['dynamics']).to(device)
        metrics = json.loads((d/'metrics.json').read_text())
        assert metrics['reload_verified'] and metrics['epochs'] == cfg['dynamics']['pretrain_epochs']
        load_checkpoint(d/'last.pt', model, device); model.eval()
        batch = tensors(table, scaler, device)
        mean, std = scaler['wear']
        monitors = {}
        for name in ['ct','mlp_L1',selection[run.fold]]:
            match = manifest[(manifest.type=='monitor')&(manifest.role=='final')&(manifest.fold==run.fold)&(manifest.seed==run.seed)&(manifest.name==name)]
            monitors[name] = load_monitor(match.iloc[0].run_dir, table, device)
        pairs = pair_tensors(table, split['test'], [1,2,3], device)
        raw = pairs.cpu().numpy(); a,b = pairs[:,0],pairs[:,1]
        rng = np.random.default_rng(31008+run.seed)
        scenarios = [('oracle',0.,0,table['VB'])]
        for sigma in [.005,.01,.02,.03]:
            for rep in range(16):
                noisy = np.maximum(0.,table['VB']+rng.normal(0,sigma,len(table['VB'])))
                scenarios.append(('noisy',sigma,rep,noisy))
        scenarios += [(name,0.,0,z) for name,z in monitors.items()]
        with torch.no_grad():
            for method,sigma,rep,initials in scenarios:
                wear = torch.tensor((initials[raw[:,0]]-mean)/std,device=device,dtype=torch.float32).reshape(-1,1)
                result = model.propagate(wear,batch['condition'][a],batch['time'][b]-batch['time'][a]).flatten().cpu().numpy()*std+mean
                forecasts.append(pd.DataFrame(dict(fold=run.fold,seed=run.seed,method=method,sigma=sigma,replicate=rep,
                    case=table['case'][raw[:,0]],origin_run=table['run'][raw[:,0]],target_run=table['run'][raw[:,1]],
                    horizon=raw[:,2],initial_vb_mm=initials[raw[:,0]],target_vb_mm=table['VB'][raw[:,1]],predicted_vb_mm=result)))
            for case in split['test']:
                rows = np.flatnonzero(table['case']==case)
                first_label = np.flatnonzero(np.isfinite(table['VB'][rows]))[0]
                for start_kind,start in [('first_labeled',first_label),('first_record',0)]:
                    chosen = scenarios if start_kind=='first_labeled' else [x for x in scenarios if x[0] in monitors]
                    rows_after = rows[start:]
                    initial = np.array([x[3][rows_after[0]] for x in chosen])
                    state = torch.tensor((initial-mean)/std,device=device,dtype=torch.float32).reshape(-1,1)
                    for position,current in enumerate(rows_after):
                        if position:
                            prev = rows_after[position-1]
                            interval = batch['time'][current:current+1]-batch['time'][prev:prev+1]
                            assert np.all([table[k][prev]==table[k][current] for k in ['DOC','feed','material']])
                            state = model.propagate(state,batch['condition'][prev:prev+1].expand(len(chosen),-1),interval.expand(len(chosen),-1))
                        result = state.flatten().cpu().numpy()*std+mean
                        for j,(method,sigma,rep,_) in enumerate(chosen):
                            trajectories.append(dict(fold=run.fold,seed=run.seed,case=case,method=method,sigma=sigma,replicate=rep,
                                start_kind=start_kind,run=int(table['run'][current]),time=table['time'][current],
                                initial_run=int(table['run'][rows_after[0]]),is_initial=position==0,
                                target_vb_mm=table['VB'][current],predicted_vb_mm=float(result[j])))
    future = pd.concat(forecasts,ignore_index=True); recursive = pd.DataFrame(trajectories)
    assert np.isfinite(future[['target_vb_mm','predicted_vb_mm']].to_numpy()).all()
    assert np.isfinite(recursive.predicted_vb_mm).all()
    future.to_csv(root/'phase3_forecast_predictions.csv.gz',index=False)
    recursive.to_csv(root/'phase3_recursive_predictions.csv.gz',index=False)
    summaries=[]
    for task,frame in [('forecast',future),('recursive',recursive)]:
        q=frame[frame.target_vb_mm.notna()].copy()
        if task=='recursive': q=q[~q.is_initial]
        else:q['start_kind']='each_labeled_origin'
        q['error']=abs(q.predicted_vb_mm-q.target_vb_mm)
        # Monte Carlo replicates are NOT independent cases.
        per=q.groupby(['fold','seed','case','method','sigma','start_kind','replicate']).error.mean().reset_index()
        per=per.groupby(['fold','seed','case','method','sigma','start_kind']).error.mean().reset_index()
        per.to_csv(root/f'phase3_{task}_case_metrics.csv',index=False)
        s=per.groupby(['fold','method','sigma','start_kind']).error.mean().reset_index().rename(columns={'error':'mae'})
        summaries.append(s.assign(task=task))
    summary=pd.concat(summaries);summary.to_csv(root/'phase3_summary.csv',index=False)
    per_h=future.assign(error=abs(future.predicted_vb_mm-future.target_vb_mm)).groupby(
        ['fold','seed','case','method','sigma','replicate','horizon']).error.mean().reset_index()
    per_h.groupby(['fold','method','sigma','horizon']).error.mean().rename('mae').to_csv(root/'phase3_forecast_horizons.csv')
    return summary


def report(root,epochs):
    manifest=pd.read_csv(root/'study_manifest.csv')
    if epochs==500:assert len(manifest)==322 and manifest.id.nunique()==322
    selection,summary=phase1_report(root,manifest,epochs)
    dynamics=phase3(root,manifest,selection)
    lines=['# Phase 1 与 Phase 3 结果','', '主指标：case-macro MAE（mm）；内层验证仅使用开发 case，第末轮固定评估。', '',
           '## Phase 1','','| model | subset | MAE |','|---|---|---:|']
    for r in summary.itertuples():lines.append(f'| {r.name} | {r.subset} | {r.mae:.5f} |')
    lines+=['','历史结构/窗口按内层验证选择：'+json.dumps(selection), '',
            '## Phase 3','','| task | fold | initialization | sigma | start | MAE |','|---|---|---|---:|---|---:|']
    for r in dynamics.itertuples():lines.append(f'| {r.task} | {r.fold} | {r.method} | {r.sigma} | {r.start_kind} | {r.mae:.5f} |')
    lines+=['','带噪初值在 mm 空间添加独立高斯扰动并截断到非负；每个 sigma 16 次扰动先平均，不作独立 case。',
            '递推始终沿每条原始保留轨迹更新（包含无标签记录），中间不读取真实 VB 重置。初始点不计入递推误差。',
            'first_record 从原始保留轨迹的首条记录初始化；质量过滤运行中若首条 sensor 无效，使用冻结的工况＋时间后备模型。first_labeled 是与 oracle 对齐的诊断设置，二者分开报告。',
            '不同 fold 选出的历史网络可能不同，不能把该选择流程称为固定网络。']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    duration=sum(json.loads((Path(p)/'metrics.json').read_text())['wall_seconds'] for p in manifest.run_dir)
    write_json(root/'completion.json',dict(runs=len(manifest),training_seconds=duration,
        phase1_runs=int((manifest.type=='monitor').sum()),phase3_ode_runs=int((manifest.type=='ode').sum())))
    write_json(root/'status.json',dict(status='completed',runs=len(manifest),epochs=epochs))


def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--report-only',action='store_true');parser.add_argument('--quality-skip',action='store_true');args=parser.parse_args()
    if args.quality_skip:ROOT=ROOT.with_name('history_quality_skip_20261008')
    root=ROOT.with_name(ROOT.name+'_smoke') if args.smoke else ROOT
    if args.report_only:report(root,500);return
    tasks=plan(root,quality=args.quality_skip)
    if not(args.execute or args.smoke):return
    if args.smoke:
        # One inner and one final run for each of 13 architectures; one ODE.
        tasks=[t for t in tasks if t['fold']=='fold_a' and t['seed']==0 and
               (t['type']=='ode' or t['role']=='final' or t['holdout']==1)]
    rows=[];existing={}
    manifest_path=root/'study_manifest.csv'
    if manifest_path.exists() and not args.smoke: existing={r['id']:r for r in pd.read_csv(manifest_path).to_dict('records')}
    for task in tasks:
        cfg=copy.deepcopy(task['config'])
        if args.smoke:cfg['training'].update(epochs=3,selection_epoch=3);cfg['dynamics']['pretrain_epochs']=3
        if args.quality_skip and cfg['monitor'].get('history_length'):
            candidates=[r for r in rows if r['name']=='ct' and r['fold']==task['fold'] and r['role']==task['role'] and
                        r['holdout']==task['holdout'] and r['seed']==task['seed']]
            assert len(candidates)==1
            cfg['monitor']['fallback_checkpoint']=str(Path(candidates[0]['run_dir'])/'last.pt')
        old=existing.get(task['id'])
        # Config-verified completed work can be reused after reporting-only source changes.
        if old and json.loads((Path(old['run_dir'])/'status.json').read_text())['status']=='completed':
            identity=json.loads((Path(old['run_dir'])/'provenance.json').read_text())
            previous=identity['config']
            with zipfile.ZipFile(project_path('wear_monitoring_dynamics/outputs/sources')/(identity['source_hash']+'.zip')) as archive:
                for path in PACKAGE.rglob('*.py'):
                    if path.relative_to(PACKAGE).parts[0] not in ('scripts','tests','outputs'):
                        assert archive.read(path.relative_to(PACKAGE.parent).as_posix())==path.read_bytes(), 'Execution code changed; reuse requires review'
            expected=copy.deepcopy(cfg)
            if task['type']=='ode': expected['experiment']['seed']=expected['dynamics']['pretrain_seed'];expected['training']['epochs']=expected['dynamics']['pretrain_epochs']
            assert previous==expected
            directory=Path(old['run_dir'])
        else:directory=run_training(cfg,pretrain=task['type']=='ode')
        row={k:v for k,v in task.items() if k!='config'};row['run_dir']=str(directory);rows.append(row)
        pd.DataFrame(rows).to_csv(manifest_path,index=False)
        write_json(root/'status.json',dict(status='running',completed=len(rows),planned=len(tasks)))
    if args.smoke:write_json(root/'status.json',dict(status='smoke_completed',runs=len(rows)))
    else:report(root,500)


if __name__=='__main__':
    try:main()
    except Exception as error:
        write_json(ROOT/'failure.json',dict(error=str(error)));raise
