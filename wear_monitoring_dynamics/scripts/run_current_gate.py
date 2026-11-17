"""Current top18 soft-gate ablation. Reuse frozen input protocol and baselines."""
import argparse,copy,json,zipfile
from pathlib import Path
import numpy as np,pandas as pd,torch
from ..config import PACKAGE,project_path,write_json
from ..training.trainer import run_training
ROOT=project_path('wear_monitoring_dynamics/outputs/current_gate_20261012')
PARENT=ROOT.with_name('current_top18_preprocessing_20261010')
KINDS=['static','condition','time_condition']
LAMBDAS=[0.,.001,.01,.1]

def plan(root):
    root.mkdir(exist_ok=True,parents=True)
    base=pd.read_csv(PARENT/'study_manifest.csv');base=base[base.name=='clip_zscore'];assert len(base)==24
    base.to_csv(root/'reused_sensor_manifest.csv',index=False)
    pd.read_csv(PARENT/'reused_baseline_manifest.csv').to_csv(root/'reused_ct_manifest.csv',index=False)
    tasks=[]
    for row in base.itertuples():
        cfg=json.loads((Path(row.run_dir)/'provenance.json').read_text())['config']
        for kind in KINDS:
            for weight in LAMBDAS:
                c=copy.deepcopy(cfg);name=f'{kind}_l1_{weight:g}';c['study_id']=root.name;c['experiment']['name']=name
                c['monitor']['architecture']='gated_feature_mlp';c['monitor']['gate']=dict(kind=kind,l1=weight,hidden_dim=16)
                tasks.append(dict(id=f'{row.fold}/{row.role}/{row.holdout}/{row.seed}/{name}',name=name,kind=kind,l1=weight,fold=row.fold,role=row.role,holdout=None if pd.isna(row.holdout) else int(row.holdout),seed=row.seed,config=c))
    write_json(root/'planned_tasks.json',tasks);return tasks

def md(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def report(root,epochs=500):
    manifest=pd.read_csv(root/'study_manifest.csv');inner=[];pred=[];audit=[]
    assert len(manifest)==(288 if epochs==500 else 12)
    for row in manifest.itertuples():
        d=Path(row.run_dir);met=json.loads((d/'metrics.json').read_text());h=pd.read_csv(d/'history.csv')
        assert h.epoch.tolist()==list(range(1,epochs+1)) and met['selected_epoch']==epochs and met['reload_verified']
        assert h.filter(regex='^(train|validation|test_diagnostic|gate)_').notna().all().all()
        ep=pd.read_csv(d/'epoch_predictions.csv.gz');assert np.isfinite(ep[['predicted_vb_mm','target_vb_mm']]).all().all()
        for _,g in ep.groupby(['split','task']):assert g.epoch.nunique()==epochs and g.groupby('epoch').size().nunique()==1
        audit.append(dict(id=row.id,epochs=epochs,prediction_rows=len(ep),seconds=met['wall_seconds']))
        if row.role=='inner':inner.append(dict(fold=row.fold,kind=row.kind,l1=row.l1,name=row.name,holdout=row.holdout,mae=met['monitoring']['case_macro_mae']))
        else:pred.append(pd.read_csv(d/'monitoring_predictions.csv').assign(name=row.name,kind=row.kind,l1=row.l1,fold=row.fold,seed=row.seed))
    for fn,name in [('reused_sensor_manifest.csv','ungated'),('reused_ct_manifest.csv','ct_baseline')]:
        b=pd.read_csv(root/fn)
        if epochs!=500:b=b[(b.fold=='fold_a')&(b.seed==0)]
        for row in b.itertuples():
            met=json.loads((Path(row.run_dir)/'metrics.json').read_text())
            if row.role=='inner':inner.append(dict(fold=row.fold,kind=name,l1=0.,name=name,holdout=row.holdout,mae=met['monitoring']['case_macro_mae']))
            else:pred.append(pd.read_csv(Path(row.run_dir)/'monitoring_predictions.csv').assign(name=name,kind=name,l1=0.,fold=row.fold,seed=row.seed))
    pred=pd.concat(pred);pred['ae']=abs(pred.predicted_vb_mm-pred.vb_mm);pred.to_csv(root/'test_predictions.csv',index=False)
    scores=pd.DataFrame(inner).groupby(['fold','kind','l1','name']).mae.mean().reset_index();scores.to_csv(root/'inner_validation_scores.csv',index=False)
    selected=scores.sort_values(['mae','l1','name']).groupby(['fold','kind'],as_index=False).first();selected.to_csv(root/'selected_lambdas.csv',index=False)
    extra=[pred[(pred.fold==r.fold)&(pred.name==r.name)].assign(name='selected_'+r.kind) for r in selected.itertuples() if r.kind in KINDS]
    best=scores.sort_values(['mae','name']).groupby('fold',as_index=False).first();best.to_csv(root/'selected_family.csv',index=False)
    extra += [pred[(pred.fold==r.fold)&(pred.name==r.name)].assign(name='validation_selected_all') for r in best.itertuples()]
    q=pd.concat([pred]+extra);rows=[]
    for name,g in q.groupby('name'):
        subsets=dict(overall=np.ones(len(g),bool),fold_a=g.fold=='fold_a',fold_b=g.fold=='fold_b',fold_a_in=(g.fold=='fold_a')&(g.vb_mm<=.74),fold_a_out=(g.fold=='fold_a')&(g.vb_mm>.74))
        for subset,mask in subsets.items():
            x=g[mask]
            if len(x):rows.append(dict(name=name,subset=subset,mae=x.groupby(['fold','seed','case']).ae.mean().mean()))
    summary=pd.DataFrame(rows);summary.to_csv(root/'summary.csv',index=False)
    q.groupby(['name','fold','seed','case']).ae.mean().rename('mae').reset_index().to_csv(root/'case_metrics.csv',index=False)
    pd.DataFrame(audit).to_csv(root/'run_audit.csv',index=False)
    (root/'RESULTS.md').write_text('# 电流18维软门控实验\n\n固定500epoch；门控输出均值L1；每fold仅内层验证选择正则强度。静态18参数，condition gate386参数，time_condition gate402参数，叠加原预测器3649参数。均使用AdamW weight_decay=1e-4，gate初值0.5。动态结构参数更多，lambda=0为容量对照。\n\n'+md(summary.pivot(index='name',columns='subset',values='mae').reset_index())+'\n\n## 内层选择\n\n'+md(selected)+'\n\n跨结构含基线的选择：\n\n'+md(best)+'\n\n所有逐epoch指标和预测保留在manifest索引目录中；gate统计为训练后值，loss_gate_l1为更新前值。测试诊断不参与选择。软gate不等于硬选择，sum(g)仅是软放行量。五seed不作为独立case。此前已查看外层测试，属探索性研究。\n',encoding='utf-8')
    write_json(root/'status.json',dict(status='completed' if epochs==500 else 'smoke_completed',runs=len(manifest),epochs=epochs,training_seconds=sum(a['seconds'] for a in audit)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    root=ROOT.with_name(ROOT.name+'_smoke') if args.smoke else ROOT
    if args.report_only:report(root);return
    tasks=plan(root)
    if not(args.execute or args.smoke):return
    if args.smoke:tasks=[t for t in tasks if t['fold']=='fold_a' and t['seed']==0 and (t['role']=='final' or t['holdout']==1) and t['l1'] in (0.,.1)]
    mf=root/'study_manifest.csv';old={r['id']:r for r in pd.read_csv(mf).to_dict('records')} if mf.exists() else {};rows=[]
    try:
        for t in tasks:
            cfg=copy.deepcopy(t['config'])
            if args.smoke:cfg['training'].update(epochs=3,selection_epoch=3)
            if t['id'] in old:
                directory=Path(old[t['id']]['run_dir']);identity=json.loads((directory/'provenance.json').read_text());assert identity['config']==cfg
                with zipfile.ZipFile(project_path(cfg['output_root'])/'sources'/(identity['source_hash']+'.zip')) as z:
                    for p in PACKAGE.rglob('*.py'):
                        if p.relative_to(PACKAGE).parts[0] not in ('scripts','tests','outputs'):assert z.read(p.relative_to(PACKAGE.parent).as_posix())==p.read_bytes(),'Execution source changed'
            else:directory=run_training(cfg)
            assert json.loads((directory/'status.json').read_text())['status']=='completed'
            rows.append({**{k:v for k,v in t.items() if k!='config'},'run_dir':str(directory)});pd.DataFrame(rows).to_csv(mf,index=False);write_json(root/'status.json',dict(status='running',completed=len(rows),planned=len(tasks)))
        report(root,3 if args.smoke else 500)
    except Exception as e:write_json(root/'failure.json',dict(error=str(e)));raise

if __name__=='__main__':main()

