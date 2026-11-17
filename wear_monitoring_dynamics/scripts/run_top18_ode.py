"""All four fixed top-18 monitors + joint pretrained ODE, matched to pure runs."""
import argparse
import copy
import json
import zipfile
from pathlib import Path
import pandas as pd
from ..config import PACKAGE, PROJECT, load_config, project_path, write_json, digest
from ..study import build_tasks, select_epoch, variants
from ..training.trainer import run_training
from ..evaluation.bootstrap import paired_case_bootstrap
from ..evaluation.monotonicity import per_case_monotonicity, epoch_monotonicity
from .run_fixed_conditioning import MODELS, normalized_config
from .run_monotonicity import reuse_catalog

STUDY='top18_sensor_ode_20260927'
OUTPUT=project_path('wear_monitoring_dynamics/outputs')
PURE=OUTPUT/'top18_sensor_features_20260927'


def specification():
    spec=json.loads((PURE/'study_spec.json').read_text())
    spec['study_id']=STUDY
    for v in spec['variants']:
        v['experiment']['name']+='_forecast'
        v['dynamics']['mode']='joint_pretrained'
        v['loss'].update(derivative_weight=0.,forecasting_weight=1.)
    return spec


def restore_epoch_exports(root):
    m=pd.read_csv(root/'study_manifest.csv')
    for (experiment,fold),g in m[m.type=='monitor'].groupby(['experiment','fold']):
        paths=[Path(p) for p in g[g.role=='inner'].run_dir]
        epoch,curve=select_epoch(paths)
        for directory in g[g.role=='final'].run_dir:
            assert json.loads((Path(directory)/'metrics.json').read_text())['selected_epoch']==epoch
        dest=root/'epoch_selection'/experiment;dest.mkdir(parents=True,exist_ok=True)
        write_json(dest/f'{fold}.json',dict(selected_epoch=epoch,inner_runs=[str(p) for p in paths]))
        curve.to_csv(dest/f'{fold}.csv')


def audit_pure_comparison(root):
    old=pd.read_csv(PURE/'study_manifest.csv');new=pd.read_csv(root/'study_manifest.csv')
    checks=[];sources=set()
    for r in new[new.type=='monitor'].itertuples():
        ni=json.loads((Path(r.run_dir)/'provenance.json').read_text());nc=copy.deepcopy(ni['config'])
        candidates=old[(old.experiment==r.experiment.removesuffix('_forecast'))&(old.fold==r.fold)&(old.role==r.role)&(old.seed==r.seed)]
        if r.role=='inner':candidates=candidates[candidates.task_id.str.endswith('/'+str(nc['experiment']['holdout']))]
        assert len(candidates)==1
        olddir=Path(candidates.iloc[0].run_dir);oi=json.loads((olddir/'provenance.json').read_text());oc=copy.deepcopy(oi['config'])
        assert json.loads((olddir/'status.json').read_text())['status']=='completed'
        for k in ('data_signature','torch','numpy','python','device_name'):assert ni[k]==oi[k]
        assert nc['dynamics']['mode']=='joint_pretrained' and nc['loss']['forecasting_weight']==1.
        assert oc['dynamics']['mode']=='none' and oc['loss']['forecasting_weight']==0.
        for cfg in (nc,oc):
            cfg.pop('study_id');cfg.pop('output_root');cfg['experiment'].pop('name');cfg['training'].pop('selection_epoch',None)
            cfg['dynamics']['mode']='none';cfg['dynamics']['checkpoint']=None;cfg['loss']['forecasting_weight']=0.
        assert nc==oc
        source=oi['source_hash']
        if source not in sources:
            with zipfile.ZipFile(OUTPUT/'sources'/f'{source}.zip') as z:
                paths=[p for p in PACKAGE.rglob('*.py') if p.relative_to(PACKAGE).parts[0] not in ('scripts','tests','outputs')]
                paths.append(PROJECT/'release/CondWear-ODE/models/vb_sensor_condition_ode.py')
                for p in paths:assert p.read_bytes()==z.read(p.relative_to(PROJECT).as_posix()),str(p)
            sources.add(source)
        checks.append(dict(ode_run=r.run_dir,pure_run=str(olddir)))
    write_json(root/'summary/pure_comparison_audit.json',dict(pairs=checks,pure_sources=sorted(sources),execution_modules_identical=True))


def report(root):
    from .analyze_completed import analyze
    restore_epoch_exports(root);analyze(root);audit_pure_comparison(root)
    out=root/'summary';manifest=pd.read_csv(root/'study_manifest.csv')
    puremanifest=pd.read_csv(PURE/'study_manifest.csv')
    cases=pd.read_csv(out/'per_case_metrics.csv');purecases=pd.read_csv(PURE/'summary/per_case_metrics.csv')
    old54=pd.read_csv(OUTPUT/'fixed_conditioning_20260923/summary/per_case_metrics.csv')
    summaries=[];diags=[];curves=[];forecasts=[]
    for name,label,_,_ in MODELS[1:]:
        a=cases[cases.experiment==name+'_top18_forecast'].groupby(['fold','case']).mae.mean()
        b=purecases[purecases.experiment==name+'_top18'].groupby(['fold','case']).mae.mean()
        row=dict(model=name,label=label,pure18_mae=b.mean(),ode18_mae=a.mean(),**paired_case_bootstrap(a,b))
        row['pure54_mae']=old54[old54.experiment==name].mae.mean()
        row['ode54_mae']=old54[old54.experiment==name+'_forecast'].mae.mean()
        for fold in ('fold_a','fold_b'):
            row[fold+'_pure18_mae']=b.loc[fold].mean();row[fold+'_ode18_mae']=a.loc[fold].mean()
        for mode,m in [('pure',puremanifest[puremanifest.experiment==name+'_top18']),('ode',manifest[manifest.experiment==name+'_top18_forecast'])]:
            accumulated=[]
            for run in m[m.role=='final'].itertuples():
                directory=Path(run.run_dir)
                d=per_case_monotonicity(pd.read_csv(directory/'monitoring_predictions.csv'))
                d['model']=name;d['mode']=mode;d['fold']=run.fold;d['seed']=run.seed
                accumulated.append(d);diags.append(d)
                ep=epoch_monotonicity(directory/'epoch_predictions.csv.gz');ep['model']=name;ep['mode']=mode;ep['fold']=run.fold;ep['seed']=run.seed;curves.append(ep)
                if mode=='ode':
                    metrics=json.loads((directory/'metrics.json').read_text())
                    for horizon,values in metrics['forecasting_by_horizon'].items():
                        forecasts.append(dict(model=name,fold=run.fold,seed=run.seed,horizon=horizon,**values))
            d=pd.concat(accumulated)
            row[mode+'_violation_rate']=d.decreases.sum()/d.pairs.sum()
            row[mode+'_mean_drop_mm']=(d.mean_drop_mm*d.pairs).sum()/d.pairs.sum()
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/'pure_vs_ode.csv',index=False)
    pd.concat(diags).to_csv(out/'monotonicity_per_case.csv',index=False)
    pd.concat(curves).to_csv(out/'final_epoch_monotonicity.csv.gz',index=False)
    pd.DataFrame(forecasts).to_csv(out/'forecasting_run_metrics.csv',index=False)
    pd.DataFrame(forecasts).groupby(['model','horizon']).mean(numeric_only=True).drop(columns='seed').to_csv(out/'forecasting_summary.csv')
    purecurve=pd.read_csv(PURE/'summary/final_epoch_curves_5seed.csv');odecurve=pd.read_csv(out/'final_epoch_curves_5seed.csv')
    pd.concat([purecurve,odecurve]).query('epoch==500').groupby('experiment').mean_test_case_macro_mae.mean().to_csv(out/'fixed_epoch500.csv')
    audit=json.loads((out/'audit_summary.json').read_text())
    audit.update(new_runs=96,reused_pretrains=16,new_training_seconds=sum(json.loads((Path(p)/'metrics.json').read_text())['wall_seconds'] for p in manifest[manifest.type=='monitor'].run_dir))
    write_json(out/'completion.json',audit)
    print(pd.DataFrame(summaries).drop(columns='label').to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    root=OUTPUT/STUDY;root.mkdir(exist_ok=True)
    if args.report_only:
        report(root);write_json(root/'status.json',{'status':'completed','audit_passed':True});return
    base=load_config(PACKAGE/'configs/time_conditioning.yaml');spec=specification()
    tasks=build_tasks(base,spec);assert len(tasks)==112
    write_json(root/'study_spec.json',spec);write_json(root/'planned_tasks.json',tasks)
    write_json(root/'feature_selection.json',json.loads((PURE/'feature_selection.json').read_text()))
    if not args.execute and not args.smoke:return
    catalog,audit=reuse_catalog(base);write_json(root/'pretrain_reuse_audit.json',audit)
    parents={}
    for task in tasks:
        if task['type']=='pretrain':
            key=digest(dict(config=normalized_config(task['config'],True),pretrain=True))
            parents[task['id']]=catalog[key]
    assert len(parents)==16
    if args.smoke:
        for cfg in variants(base,spec):
            cfg['study_id']=STUDY+'_smoke';cfg['training']['epochs']=10
            cfg['dynamics']['checkpoint']=str(parents['ode/fold_a/8']/'last.pt')
            run_training(cfg)
        return
    write_json(root/'status.json',{'status':'running'})
    rows=[];done={}
    for task in tasks:
        cfg=copy.deepcopy(task['config'])
        if task['type']=='pretrain':directory=parents[task['id']]
        else:
            parent=next(k for k in task['dependencies'] if k.startswith('ode/'))
            cfg['dynamics']['checkpoint']=str(done[parent]/'last.pt')
            if task.get('inner_ids'):
                epoch,curve=select_epoch([done[k] for k in task['inner_ids']]);cfg['training']['selection_epoch']=epoch
                dest=root/'epoch_selection'/cfg['experiment']['name'];dest.mkdir(parents=True,exist_ok=True)
                curve.to_csv(dest/f'{cfg["experiment"]["fold"]}.csv')
                write_json(dest/f'{cfg["experiment"]["fold"]}.json',dict(selected_epoch=epoch,inner_runs=[str(done[k]) for k in task['inner_ids']]))
            directory=run_training(cfg)
        done[task['id']]=directory;e=cfg['experiment']
        rows.append(dict(task_id=task['id'],type=task['type'],experiment=e['name'],fold=e['fold'],role=e['role'],seed=e['seed'],status='completed',origin='reused' if task['type']=='pretrain' else 'new',run_dir=str(directory)))
        pd.DataFrame(rows).to_csv(root/'study_manifest.csv',index=False)
    report(root);write_json(root/'status.json',{'status':'completed','audit_passed':True})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write_json(OUTPUT/STUDY/'status.json',{'status':'failed','error':repr(error)})
        raise
