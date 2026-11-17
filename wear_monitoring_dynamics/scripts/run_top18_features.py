"""Four fixed pure monitors using the pre-existing descriptive top-18 list."""
import argparse
import copy
import json
from pathlib import Path
import pandas as pd
import torch
from ..config import PACKAGE, load_config, project_path, write_json, file_hash
from ..study import build_tasks, select_epoch, variants
from ..training.trainer import run_training
from ..models.registry import build_model
from ..evaluation.bootstrap import paired_case_bootstrap
from ..evaluation.monotonicity import epoch_monotonicity, per_case_monotonicity
from .run_fixed_conditioning import MODELS
from .run_monotonicity import verify_compatible_source

STUDY='top18_sensor_features_20260927'
OUTPUT=project_path('wear_monitoring_dynamics/outputs')
RANKING=OUTPUT/'sensor_feature_inspection_20260927/temporal_monotonicity_ranking.csv'


def specification():
    r=pd.read_csv(RANKING)
    names=r[r.view=='all_records'].sort_values('rank').head(18).feature.tolist()
    assert len(names)==18
    items=[]
    for name,_,architecture,inputs in MODELS[1:]:
        items.append(dict(experiment={'name':name+'_top18'},
            monitor=dict(architecture=architecture,inputs=inputs,selected_features=names),
            dynamics={'mode':'none'},loss=dict(derivative_weight=0.,forecasting_weight=0.)))
    return dict(study_id=STUDY,family='explicit',variants=items,folds=['fold_a','fold_b'],seeds=list(range(5)),inner_seed=0)


def report(root):
    from .analyze_completed import analyze
    manifest=pd.read_csv(root/'study_manifest.csv')
    # Reconstruct export metadata from the actual seven validation histories,
    # and verify it against every completed final checkpoint's chosen epoch.
    for (experiment,fold),group in manifest.groupby(['experiment','fold']):
        paths=[Path(p) for p in group[group.role=='inner'].run_dir]
        epoch,curve=select_epoch(paths)
        for directory in group[group.role=='final'].run_dir:
            metrics=json.loads((Path(directory)/'metrics.json').read_text())
            assert metrics['selected_epoch']==epoch
        dest=root/'epoch_selection'/experiment;dest.mkdir(parents=True,exist_ok=True)
        write_json(dest/f'{fold}.json',dict(selected_epoch=epoch,inner_runs=[str(p) for p in paths]))
    analyze(root)
    out=root/'summary';m=pd.read_csv(root/'study_manifest.csv')
    new=pd.read_csv(out/'per_case_metrics.csv')
    oldroot=OUTPUT/'fixed_conditioning_20260923'
    old=pd.read_csv(oldroot/'summary/per_case_metrics.csv')
    oldmanifest=pd.read_csv(oldroot/'study_manifest.csv')
    checked=set();checks=[]
    for run in m.itertuples():
        newidentity=json.loads((Path(run.run_dir)/'provenance.json').read_text())
        nc=copy.deepcopy(newidentity['config'])
        candidate=oldmanifest[(oldmanifest.experiment==run.experiment.removesuffix('_top18'))&(oldmanifest.fold==run.fold)&(oldmanifest.role==run.role)&(oldmanifest.seed==run.seed)]
        if run.role=='inner':
            candidate=candidate[candidate.task_id.str.endswith('/'+str(nc['experiment']['holdout']))]
        assert len(candidate)==1
        olddir=Path(candidate.iloc[0].run_dir);oi=json.loads((olddir/'provenance.json').read_text());oc=copy.deepcopy(oi['config'])
        assert json.loads((olddir/'status.json').read_text())['status']=='completed'
        for key in ('data_signature','torch','numpy','python','device_name'):assert oi[key]==newidentity[key]
        for cfg in (nc,oc):
            cfg.pop('study_id');cfg.pop('output_root');cfg['experiment'].pop('name');cfg['training'].pop('selection_epoch',None);cfg['monitor'].pop('selected_features',None)
        assert nc==oc
        source=oi['source_hash']
        if source not in checked:
            verify_compatible_source(OUTPUT/'sources'/f'{source}.zip');checked.add(source)
        checks.append(dict(new_run=run.run_dir,baseline_run=str(olddir)))
    write_json(out/'baseline_comparison_audit.json',dict(pairs=checks,baseline_sources=sorted(checked),config_changes='feature subset; input-layer dimensions/channel count; independently CV-selected epoch'))
    summaries=[];diags=[];epochframes=[]
    for name,label,_,_ in MODELS[1:]:
        a=new[new.experiment==name+'_top18'].groupby(['fold','case']).mae.mean()
        b=old[old.experiment==name].groupby(['fold','case']).mae.mean()
        row=dict(model=name,label=label,full54_mae=b.mean(),top18_mae=a.mean(),**paired_case_bootstrap(a,b))
        for fold in ('fold_a','fold_b'):
            row[fold+'_full54_mae']=b.loc[fold].mean();row[fold+'_top18_mae']=a.loc[fold].mean()
        for feature_set,manifest in [('top18',m[m.experiment==name+'_top18']),('full54',oldmanifest[oldmanifest.experiment==name])]:
            chosen=manifest[manifest.role=='final'];rates=[]
            for run in chosen.itertuples():
                directory=Path(run.run_dir)
                metrics=json.loads((directory/'metrics.json').read_text());assert metrics['reload_verified']
                diag=per_case_monotonicity(pd.read_csv(directory/'monitoring_predictions.csv'))
                diag['model']=name;diag['feature_set']=feature_set;diag['fold']=run.fold;diag['seed']=run.seed
                rates.append(diag);diags.append(diag)
                ep=epoch_monotonicity(directory/'epoch_predictions.csv.gz');ep['model']=name;ep['feature_set']=feature_set;ep['fold']=run.fold;ep['seed']=run.seed;epochframes.append(ep)
            d=pd.concat(rates)
            row[feature_set+'_violation_rate']=d.decreases.sum()/d.pairs.sum()
            row[feature_set+'_mean_drop_mm']=(d.mean_drop_mm*d.pairs).sum()/d.pairs.sum()
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out/'full54_vs_top18.csv',index=False)
    pd.concat(diags).to_csv(out/'monotonicity_per_case.csv',index=False)
    pd.concat(epochframes).to_csv(out/'final_epoch_monotonicity.csv.gz',index=False)
    newcurve=pd.read_csv(out/'final_epoch_curves_5seed.csv')
    oldcurve=pd.read_csv(oldroot/'summary/final_epoch_curves_5seed.csv')
    pd.concat([newcurve,oldcurve[oldcurve.experiment.isin([x[0] for x in MODELS[1:]])]]).query('epoch==500').groupby('experiment').mean_test_case_macro_mae.mean().to_csv(out/'fixed_epoch500.csv')
    audit=json.loads((out/'audit_summary.json').read_text())
    audit['training_seconds']=sum(json.loads((Path(p)/'metrics.json').read_text())['wall_seconds'] for p in m.run_dir)
    write_json(out/'completion.json',audit)
    print(pd.DataFrame(summaries).drop(columns='label').to_string(index=False),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--report-only',action='store_true');args=p.parse_args()
    if args.report_only:
        report(OUTPUT/STUDY);write_json(OUTPUT/STUDY/'status.json',{'status':'completed','audit_passed':True});return
    base=load_config(PACKAGE/'configs/time_conditioning.yaml');spec=specification()
    root=OUTPUT/STUDY;root.mkdir(exist_ok=True)
    write_json(root/'study_spec.json',spec)
    write_json(root/'feature_selection.json',dict(features=spec['variants'][0]['monitor']['selected_features'],
        ranking_path=str(RANKING),ranking_sha256=file_hash(RANKING),scope='all_records, 14 cases, globally selected before this study; exploratory, not nested feature selection',
        qc='unchanged, including suspicious case12/run1',attention='five nonempty channel tokens, nine statistic slots with unselected slots zero padded'))
    configs=variants(base,spec)
    write_json(root/'parameter_counts.json',{c['experiment']['name']:sum(p.numel() for p in build_model(c).parameters()) for c in configs})
    if args.smoke:
        for cfg in configs:
            cfg['study_id']=STUDY+'_smoke';cfg['training']['epochs']=10
            run_training(cfg)
        return
    tasks=build_tasks(base,spec);assert len(tasks)==96
    write_json(root/'planned_tasks.json',tasks)
    if not args.execute:return
    write_json(root/'status.json',{'status':'running'})
    rows=[];done={}
    for task in tasks:
        cfg=copy.deepcopy(task['config'])
        if task.get('inner_ids'):
            epoch,curve=select_epoch([done[k] for k in task['inner_ids']]);cfg['training']['selection_epoch']=epoch
            dest=root/'epoch_selection'/cfg['experiment']['name'];dest.mkdir(parents=True,exist_ok=True)
            curve.to_csv(dest/f'{cfg["experiment"]["fold"]}.csv')
            write_json(dest/f'{cfg["experiment"]["fold"]}.json',dict(selected_epoch=epoch,inner_runs=[str(done[k]) for k in task['inner_ids']]))
        directory=run_training(cfg);done[task['id']]=directory;e=cfg['experiment']
        rows.append(dict(task_id=task['id'],type='monitor',experiment=e['name'],fold=e['fold'],role=e['role'],seed=e['seed'],status='completed',run_dir=str(directory)))
        pd.DataFrame(rows).to_csv(root/'study_manifest.csv',index=False)
    report(root);write_json(root/'status.json',{'status':'completed','audit_passed':True})


if __name__=='__main__':
    try:main()
    except Exception as error:
        write_json(OUTPUT/STUDY/'status.json',{'status':'failed','error':str(error)})
        raise
