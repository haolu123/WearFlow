"""Fixed ABCD study: no new hyperparameter or epoch selection."""
import argparse,json,time
from pathlib import Path
import pandas as pd
from ..config import write_json
from ..training.adaptation_from100 import execute
BASE=Path('wear_monitoring_dynamics/outputs/current12_fusion_20261013')
ROOT=BASE.with_name('current12_abcd_from100_20261019')

def plan(root):
    root.mkdir(exist_ok=True);m=pd.read_csv(BASE/'study_manifest.csv');odes=pd.read_csv(BASE/'reused_ode_manifest.csv');tasks=[]
    for r in m[(m.type=='monitor')&(m.role=='final')].itertuples():
        ode=odes[(odes.fold==r.fold)&(odes.seed==r.seed)].iloc[0].run_dir
        for group in ['B','C','D']:
            tasks.append(dict(id=f'{r.fold}/seed{r.seed}/{group}',fold=r.fold,seed=int(r.seed),group=group,
                alpha=1. if r.fold=='fold_a' else .25,sensor_dir=r.run_dir,ode_dir=ode,
                monitor100=str(Path(r.run_dir)/'epochs/epoch_0100.pt')))
    assert len(tasks)==30;write_json(root/'planned_tasks.json',tasks);return tasks

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--report-only',action='store_true');a=ap.parse_args()
    root=ROOT.with_name(ROOT.name+'_smoke') if a.smoke else ROOT
    try:
        if not a.report_only:
            tasks=plan(root)
            if not(a.execute or a.smoke):return
            if a.smoke:tasks=tasks[:3]
            rows=[]
            for task in tasks:
                d=root/'runs'/task['id'];execute(task,d,3 if a.smoke else 100)
                rows.append({**task,'run_dir':str(d)});pd.DataFrame(rows).to_csv(root/'study_manifest.csv',index=False)
                write_json(root/'status.json',dict(status='training',completed=len(rows),planned=len(tasks)))
        if a.smoke:write_json(root/'status.json',dict(status='smoke_completed',runs=3))
        else:
            from .report_abcd_from100 import report
            report(root)
    except Exception as e:
        write_json(root/'failure.json',dict(error=str(e)));write_json(root/'status.json',dict(status='failed',error=str(e)));raise
if __name__=='__main__':main()

