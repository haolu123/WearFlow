"""Top12 monitoring, OOF measurement error, independent ODE, fixed-gain fusion."""
import argparse,copy,json,zipfile,time
from pathlib import Path
import numpy as np,pandas as pd
from ..config import PACKAGE,project_path,write_json
from ..training.trainer import run_training
ROOT=project_path('wear_monitoring_dynamics/outputs/current12_fusion_20261013')
PARENT=ROOT.with_name('current_top18_preprocessing_20261010')
ODE_PARENT=ROOT.with_name('history_quality_skip_20261008')

def plan(root):
    root.mkdir(parents=True,exist_ok=True);p=pd.read_csv(PARENT/'study_manifest.csv');p=p[p.name=='clip_zscore'];o=pd.read_csv(ODE_PARENT/'study_manifest.csv');o=o[o.type=='ode'];tasks=[]
    for r in p.itertuples():
        c=json.loads((Path(r.run_dir)/'provenance.json').read_text())['config'];c['study_id']=root.name;c['experiment']['name']='current12';c['monitor']['selected_indices']=c['monitor']['selected_indices'][:12];c['monitor']['selected_features']=c['monitor']['selected_features'][:12];c['data']['sensor_preprocessing']['indices']=c['monitor']['selected_indices']
        tasks.append(dict(id=f'{r.fold}/{r.role}/{r.holdout}/{r.seed}/monitor',type='monitor',fold=r.fold,role=r.role,holdout=None if pd.isna(r.holdout) else int(r.holdout),seed=r.seed,config=c))
        if r.role=='inner':
            od=Path(o[(o.fold==r.fold)&(o.seed==0)].iloc[0].run_dir);d=json.loads((od/'provenance.json').read_text())['config'];d['study_id']=root.name;d['experiment'].update(name='inner_ode',role='inner',holdout=int(r.holdout),seed=0);d['data']['sensor_quality']=copy.deepcopy(c['data']['sensor_quality'])
            tasks.append(dict(id=f'{r.fold}/inner/{int(r.holdout)}/0/ode',type='ode',fold=r.fold,role='inner',holdout=int(r.holdout),seed=0,config=d))
    assert len(tasks)==38
    p.to_csv(root/'parent_monitor_manifest.csv',index=False);o.to_csv(root/'reused_ode_manifest.csv',index=False);pd.read_csv(PARENT/'reused_baseline_manifest.csv').to_csv(root/'reused_ct_manifest.csv',index=False);write_json(root/'planned_tasks.json',tasks);return tasks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--report-only',action='store_true');args=ap.parse_args();root=ROOT.with_name(ROOT.name+'_smoke') if args.smoke else ROOT
    try:
        if not args.report_only:
            tasks=plan(root)
            if not(args.execute or args.smoke):return
            if args.smoke:tasks=[t for t in tasks if t['fold']=='fold_a' and t['seed']==0 and (t['holdout']==1 or t['role']=='final')]
            mf=root/'study_manifest.csv';old={r['id']:r for r in pd.read_csv(mf).to_dict('records')} if mf.exists() else {};rows=[]
            for task in tasks:
                c=copy.deepcopy(task['config'])
                if args.smoke:c['training'].update(epochs=3,selection_epoch=3);c['dynamics']['pretrain_epochs']=3
                if task['id'] in old:
                    d=Path(old[task['id']]['run_dir']);identity=json.loads((d/'provenance.json').read_text());assert identity['config']==c
                    with zipfile.ZipFile(project_path(c['output_root'])/'sources'/(identity['source_hash']+'.zip')) as z:
                        for p in PACKAGE.rglob('*.py'):
                            if p.relative_to(PACKAGE).parts[0] not in ('scripts','tests','outputs'):assert z.read(p.relative_to(PACKAGE.parent).as_posix())==p.read_bytes(),'Execution source changed'
                else:d=run_training(c,pretrain=task['type']=='ode')
                assert json.loads((d/'status.json').read_text())['status']=='completed';rows.append({**{k:v for k,v in task.items() if k!='config'},'run_dir':str(d)});pd.DataFrame(rows).to_csv(mf,index=False);write_json(root/'status.json',dict(status='training',completed=len(rows),planned=len(tasks)))
        write_json(root/'status.json',dict(status='evaluating'))
        from .report_current12_fusion import report
        report(root,3 if args.smoke else 500)
    except Exception as e:write_json(root/'failure.json',dict(error=str(e)));raise
if __name__=='__main__':main()

