"""Matched direct-transition versus ODE experiment, with current and future VB."""

import argparse

import copy

import json

import time

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from ..config import project_path, write_json

from ..data.records import load_table

from ..data.datasets import tensors, pair_tensors

from ..models.registry import build_model

from ..training.trainer import run_training

from ..training.checkpoint import load_checkpoint

from ..evaluation.bootstrap import paired_case_bootstrap



ROOT=project_path('wear_monitoring_dynamics/outputs/direct_vs_ode_20260929')

PREVIOUS=project_path('wear_monitoring_dynamics/outputs/train_only_coverage_20260928')

KINDS=['direct','direct_positive','ode']





def plan():

    previous=json.loads((PREVIOUS/'planned_runs.json').read_text())

    configs=[]

    for old in previous:

        if old['study']!='train_only' or old['mode']!='ode':continue

        for kind in KINDS:

            cfg=copy.deepcopy(old['config'])

            cfg['study_id']=ROOT.name

            cfg['experiment']['name']=old['model']+'_'+kind

            cfg['dynamics']['transition']=kind

            cfg['dynamics']['checkpoint']=None

            configs.append(dict(model=old['model'],kind=kind,fold=old['fold'],seed=old['seed'],config=cfg))

    assert len(configs)==150

    ROOT.mkdir(exist_ok=True)

    write_json(ROOT/'planned_runs.json',configs)

    return configs





def evaluate_run(row, epoch=500):

    d=Path(row['run_dir'])

    cfg=json.loads((d/'provenance.json').read_text())['config']

    scalers=json.loads((d/'scalers.json').read_text())

    split=json.loads((d/'split.json').read_text())

    table,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))

    device=torch.device('cuda')

    batch=tensors(table,scalers,device)

    model=build_model(cfg).to(device)

    state=load_checkpoint(d/f'epochs/epoch_{epoch:04d}.pt',model,device)

    assert state['epoch']==epoch

    model.eval()

    pair=pair_tensors(table,split['test'],[1,2,3,4,5],device)

    a,b=pair[:,0],pair[:,1]

    dt=batch['time'][b]-batch['time'][a]

    raw=pair.cpu().numpy()

    records=[]

    with torch.no_grad():

        estimated=model.estimate(batch,a)

        for initial,wear in [('estimated',estimated),('measured',batch['target'][a])]:

            pred=wear if model.dynamics is None else model.dynamics.propagate(wear,batch['condition'][a],dt)

            rawpred=pred.cpu().numpy().ravel()*scalers['wear'][1]+scalers['wear'][0]

            frame=pd.DataFrame(dict(case=table['case'][raw[:,0]],origin_run=table['run'][raw[:,0]],

                target_run=table['run'][raw[:,1]],horizon=raw[:,2],delta_time=table['time'][raw[:,1]]-table['time'][raw[:,0]],

                target_vb_mm=table['VB'][raw[:,1]],predicted_vb_mm=rawpred,task='future',initial=initial))

            records.append(frame)

        # Also diagnose propagation independently of sensor initialization, at the same state.

        if model.dynamics is not None:

            whole=model.dynamics.propagate(estimated,batch['condition'][a],dt)

            half=model.dynamics.propagate(estimated,batch['condition'][a],dt/2)

            twice=model.dynamics.propagate(half,batch['condition'][a],dt/2)

            diagnostic=dict(semigroup_gap_mm=float((whole-twice).abs().mean()*scalers['wear'][1]),

                            decline_fraction=float((whole<estimated-1e-7).float().mean()))

        else:diagnostic=dict(semigroup_gap_mm=0.,decline_fraction=0.)

    current=pd.read_csv(d/'monitoring_predictions.csv').rename(columns={'run':'origin_run','vb_mm':'target_vb_mm'})

    current=current[['case','origin_run','target_vb_mm','predicted_vb_mm']].assign(task='current',initial='estimated',horizon=0)

    records.append(current)

    diagnostic['parameter_count']=sum(p.numel() for p in model.parameters())

    diagnostic['transition_parameters']=sum(p.numel() for p in model.dynamics.parameters()) if model.dynamics is not None else 0

    meta={k:row[k] for k in ['model','kind','fold','seed']}

    return pd.concat(records,ignore_index=True).assign(**meta),dict(**meta,**diagnostic)





def report(rows):

    frames=[];diags=[];audit=[]

    for row in rows:

        d=Path(row['run_dir']);m=json.loads((d/'metrics.json').read_text())

        assert m['epochs']==m['selected_epoch']==500 and m['reload_verified']

        h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,501))

        assert h.filter(regex='^test_diagnostic_').notna().all().all()

        assert (d/'epoch_predictions.csv.gz').is_file()

        pred,diag=evaluate_run(row);frames.append(pred);diags.append(diag)

        audit.append(dict(**row,wall_seconds=m['wall_seconds']))

    # Pure monitors are existing matched train-only runs; future baseline holds VB constant.

    old=pd.read_csv(PREVIOUS/'study_manifest.csv')

    for r in old[(old.study=='train_only')&(old['mode']=='pure')].itertuples():

        row=dict(model=r.model,kind='pure_persistence',fold=r.fold,seed=r.seed,run_dir=r.run_dir)

        pred,diag=evaluate_run(row);frames.append(pred);diags.append(diag)

    predictions=pd.concat(frames,ignore_index=True)

    assert np.isfinite(predictions[['target_vb_mm','predicted_vb_mm']].to_numpy()).all()

    predictions['absolute_error']=(predictions.predicted_vb_mm-predictions.target_vb_mm).abs()

    keys=['model','kind','fold','seed','task','initial','horizon','case']

    case=predictions.groupby(keys).absolute_error.mean().rename('mae').reset_index()

    # h=1..3 aggregate gives each case equal weight, not each horizon/case cell.

    q=predictions[(predictions.task=='future') & (predictions.horizon<=3)]

    combined=q.groupby([k for k in keys if k!='horizon']).absolute_error.mean().rename('mae').reset_index().assign(horizon=-1)

    case=pd.concat([case,combined],ignore_index=True)

    summary=case.groupby(['model','kind','fold','task','initial','horizon']).mae.mean().reset_index()

    overall=case.groupby(['model','kind','task','initial','horizon']).mae.mean().reset_index().assign(fold='overall')

    summary=pd.concat([summary,overall],ignore_index=True)

    # Same origin, target and interval for every model/seed/transition within a fold.

    for (fold,task,initial),group in predictions.groupby(['fold','task','initial']):

        reference=None

        cols=['case','origin_run','target_vb_mm']+(['target_run','horizon','delta_time'] if task=='future' else [])

        for _,g in group.groupby(['model','kind','seed']):

            values=g.sort_values(cols)[cols].to_numpy()

            if reference is None:reference=values

            np.testing.assert_allclose(values,reference,atol=1e-12,rtol=0)

    comparisons=[]

    for model,g in case[(case.task=='future')&(case.horizon==-1)].groupby('model'):

        for initial in ['estimated','measured']:

            sub=g[g.initial==initial]

            ode=sub[sub.kind=='ode'].groupby(['fold','case']).mae.mean()

            for kind in ['direct','direct_positive','pure_persistence']:

                other=sub[sub.kind==kind].groupby(['fold','case']).mae.mean()

                comparisons.append(dict(model=model,initial=initial,reference=kind,**paired_case_bootstrap(ode,other)))

    predictions.to_csv(ROOT/'epoch500_predictions.csv',index=False)

    case.to_csv(ROOT/'case_seed_metrics.csv',index=False)

    summary.to_csv(ROOT/'summary.csv',index=False)

    pd.DataFrame(diags).to_csv(ROOT/'transition_diagnostics.csv',index=False)

    pd.DataFrame(comparisons).to_csv(ROOT/'paired_case_comparisons.csv',index=False)

    pd.DataFrame(audit).to_csv(ROOT/'run_metrics.csv',index=False)

    lines=['# 当前 VB 与未来 VB：直接预测对比 ODE','',

           '固定 500 epoch，5 seed；case-macro MAE，mm。future 为 h=1/2/3 训练范围内所有预测对，先逐 case 计算 MAE；oracle 使用实测初值，仅作诊断。', '',

           '| model | transition | fold | current | future estimated | future measured (oracle) |',

           '|---|---|---|---:|---:|---:|']

    for (model,kind,fold),g in summary.groupby(['model','kind','fold']):

        def value(task,initial,horizon):

            return g[(g.task==task)&(g.initial==initial)&(g.horizon==horizon)].mae.item()

        lines.append(f'| {model} | {kind} | {fold} | {value("current","estimated",0):.5f} | {value("future","estimated",-1):.5f} | {value("future","measured",-1):.5f} |')

    lines+=['','逐 horizon 1–5 结果见 summary.csv；h=4/5 未参与训练，但未必超出训练集真实时间间隔范围，不应直接称为时间外推。',

            'paired_case_comparisons.csv 为以 case 为单位的 ODE−对照差及未作多重比较校正的 bootstrap 区间。',

            '物理状态单步直接映射与 ODE 的区别不能仅凭训练集拟合误差判断；若精度无提升，应如实报告，并结合耗时与传播一致性讨论。']

    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')

    write_json(ROOT/'status.json',dict(status='completed',new_runs=len(rows),pure_reference_runs=50,audit_passed=True))





def main():

    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()

    if args.report_only:

        report(pd.read_csv(ROOT/'study_manifest.csv').to_dict('records'));return

    configs=plan()

    if not(args.execute or args.smoke):return

    if args.smoke:

        configs=[c for c in configs if c['seed']==0 and c['fold']=='fold_a']

    parents={};rows=[]

    for item in configs:

        cfg=copy.deepcopy(item['config'])

        if args.smoke:

            cfg['study_id']+='_smoke';cfg['training'].update(epochs=3,selection_epoch=3);cfg['dynamics']['pretrain_epochs']=3

        write_json(ROOT/'status.json',dict(status='smoke' if args.smoke else 'running',completed=len(rows),planned=150))

        key=(item['kind'],item['fold'])

        if key not in parents:

            pre=copy.deepcopy(cfg);pre['dynamics']['mode']='none'

            parent=run_training(pre,pretrain=True);parents[key]=str(parent/'last.pt')

        cfg['dynamics']['checkpoint']=parents[key]

        directory=run_training(cfg)

        row={k:v for k,v in item.items() if k!='config'};row['run_dir']=str(directory);rows.append(row)

        pd.DataFrame(rows).to_csv(ROOT/('smoke_manifest.csv' if args.smoke else 'study_manifest.csv'),index=False)

    if not args.smoke:report(rows)





if __name__=='__main__':

    try:main()

    except Exception as error:

        write_json(ROOT/'status.json',dict(status='failed',error=str(error)));raise

