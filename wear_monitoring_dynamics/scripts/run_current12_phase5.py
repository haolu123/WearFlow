"""Nested, split-safe calibration training for frozen-model scalar EKF fusion."""
import argparse,copy,itertools,json,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import PACKAGE,project_path,write_json
from ..data.records import load_table
from ..data.feature_archive import attach_features
from ..data.sensor_quality import raw_quality_metrics,fit_rules,assess,apply_quality,VERSION
from ..data.splits import POOLS,make_split
from ..training.trainer import run_training
from .run_psd_sweep import rank

PARENT=project_path('wear_monitoring_dynamics/outputs/current12_fusion_20261013')
ROOT=PARENT.with_name('current12_phase5_20261013')


def plan(root):
    destination=root/'planned_tasks.json'
    if destination.exists():return json.loads(destination.read_text())
    root.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(PARENT/'study_manifest.csv');tasks=[];ranks=[]
    first=Path(old[old.type=='monitor'].iloc[0].run_dir)
    cfg=json.loads((first/'provenance.json').read_text())['config']
    raw=np.load(project_path(cfg['data']['path']),allow_pickle=True).tolist();qm=raw_quality_metrics(raw)
    table,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))
    table=attach_features(table,cfg['data']['feature_archive']);names=cfg['data']['feature_archive']['names']
    candidates=[i for i,n in enumerate(names) if n.startswith(('smcAC_','smcDC_'))]
    for fold,pool in POOLS.items():
        source=Path(old[(old.type=='monitor')&(old.fold==fold)&(old.role=='inner')].iloc[0].run_dir)
        base=json.loads((source/'provenance.json').read_text())['config']
        ctbase=json.loads((Path(base['monitor']['fallback_checkpoint']).parent/'provenance.json').read_text())['config']
        od=Path(old[(old.type=='ode')&(old.fold==fold)].iloc[0].run_dir)
        odebase=json.loads((od/'provenance.json').read_text())['config']
        for pair in itertools.combinations(pool,2):
            split=make_split(fold,'nested',list(pair));rules=fit_rules(qm,split['train']);decisions,_=assess(raw,qm,rules)
            quality=dict(version=VERSION,rules=rules,invalid_records=decisions[~decisions.sensor_valid][['case','run']].astype(int).values.tolist(),fallback='frozen_condition_time',retain_previous_hard_exclusions=True)
            scores,_=rank(apply_quality(table,quality),split['train']);ordered=sorted(candidates,key=lambda j:(-scores[j],names[j]));indices=ordered[:12]
            key=f'{fold}/{pair[0]}_{pair[1]}'
            for position,j in enumerate(ordered):ranks.append(dict(split=key,rank=position+1,index=j,feature=names[j],score=scores[j]))
            for kind,template in [('ct',ctbase),('sensor',base),('ode',odebase)]:
                c=copy.deepcopy(template);c['study_id']=root.name;c['experiment'].update(name=kind,role='nested',holdout=list(pair),fold=fold,seed=0);c['data']['sensor_quality']=quality
                if kind=='sensor':
                    c['monitor'].update(selected_indices=indices,selected_features=[names[j] for j in indices],fallback_checkpoint='RESOLVE_CT')
                    c['data']['sensor_preprocessing'].update(indices=indices)
                tasks.append(dict(id=key+'/'+kind,fold=fold,holdout=list(pair),type=kind,config=c))
    assert len(tasks)==126
    pd.DataFrame(ranks).to_csv(root/'feature_rankings.csv',index=False);write_json(destination,tasks)
    return tasks


def execute(root,smoke=False):
    tasks=plan(root);tasks=tasks[:3] if smoke else tasks
    path=root/'study_manifest.csv';rows=pd.read_csv(path).to_dict('records') if path.exists() else [];done={r['id']:r for r in rows}
    for task in tasks:
        cfg=copy.deepcopy(task['config'])
        if smoke:cfg['training'].update(epochs=3,selection_epoch=3);cfg['dynamics']['pretrain_epochs']=3
        if task['type']=='sensor':cfg['monitor']['fallback_checkpoint']=str(Path(done[task['id'].rsplit('/',1)[0]+'/ct']['run_dir'])/'last.pt')
        if task['id'] in done:
            d=Path(done[task['id']]['run_dir']);p=json.loads((d/'provenance.json').read_text());assert p['config']==cfg
            with zipfile.ZipFile(project_path(cfg['output_root'])/'sources'/(p['source_hash']+'.zip')) as z:
                for f in PACKAGE.rglob('*.py'):
                    if f.relative_to(PACKAGE).parts[0] not in ('scripts','tests','outputs'):assert z.read(f.relative_to(PACKAGE.parent).as_posix())==f.read_bytes(),'Execution source changed'
        else:
            d=run_training(cfg,pretrain=task['type']=='ode');row=dict(id=task['id'],fold=task['fold'],held_a=task['holdout'][0],held_b=task['holdout'][1],type=task['type'],run_dir=str(d));rows.append(row);done[task['id']]=row;pd.DataFrame(rows).to_csv(path,index=False)
        assert json.loads((d/'status.json').read_text())['status']=='completed'
        write_json(root/'status.json',dict(status='training',completed=len(rows),planned=len(tasks)))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--report-only',action='store_true');args=ap.parse_args()
    root=ROOT.with_name(ROOT.name+'_smoke') if args.smoke else ROOT
    try:
        if not args.report_only:
            plan(root)
            if not(args.execute or args.smoke):return
            execute(root,args.smoke)
        if args.smoke:
            from .report_current12_phase5 import smoke_check
            smoke_check(root)
        else:
            write_json(root/'status.json',dict(status='evaluating'))
            from .report_current12_phase5 import report
            report(root)
    except Exception as e:
        write_json(root/'failure.json',dict(error=str(e)));write_json(root/'status.json',dict(status='failed',error=str(e)));raise

if __name__=='__main__':main()

