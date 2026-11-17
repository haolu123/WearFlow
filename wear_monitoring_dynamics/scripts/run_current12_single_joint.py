"""Single-shot end-to-end study; no filtering or sequence unrolling."""

import argparse,json,time

from pathlib import Path

import numpy as np

import pandas as pd

from ..config import write_json

from ..training.single_shot_adaptation import execute

from .report_current12_fusion import metrics,md



PARENT=Path('wear_monitoring_dynamics/outputs/current12_fusion_20261013')

ROOT=PARENT.with_name('current12_single_joint_20261014')

DEFINITIONS=[('continue_current','continue',0.),('frozen_l01','frozen',.1),('frozen_l1','frozen',1.),('joint_l01','joint',.1),('joint_l1','joint',1.)]





def plan(root):

    root.mkdir(parents=True,exist_ok=True);old=pd.read_csv(PARENT/'study_manifest.csv');odes=pd.read_csv(PARENT/'reused_ode_manifest.csv');tasks=[]

    for r in old[old.type=='monitor'].itertuples():

        hold=None if r.role=='final' else int(r.holdout)

        od=odes[(odes.fold==r.fold)&(odes.seed==r.seed)] if hold is None else old[(old.type=='ode')&(old.fold==r.fold)&(old.holdout==hold)]

        for name,mode,weight in DEFINITIONS:

            tasks.append(dict(id=f'{r.fold}/{r.role}_{hold}/{r.seed}/{name}',fold=r.fold,role=r.role,holdout=hold,seed=int(r.seed),name=name,mode=mode,lambda_future=weight,sensor_dir=r.run_dir,ode_dir=od.iloc[0].run_dir))

    assert len(tasks)==120;write_json(root/'planned_tasks.json',tasks);return tasks





def report(root,epochs=500):

    start=time.perf_counter();m=pd.read_csv(root/'study_manifest.csv');records=[];audit=[];baseline_seen=set();baseline_frames={}

    for r in m.itertuples():

        d=Path(r.run_dir);met=json.loads((d/'metrics.json').read_text());h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,epochs+1)) and met['reload_verified']

        parts=[];counts=np.zeros(epochs+1,dtype=int);groups=set()

        for c in pd.read_csv(d/'epoch_predictions.csv.gz',chunksize=200000):

            assert np.isfinite(c.predicted_vb_mm).all();counts+=np.bincount(c.epoch,minlength=epochs+1);groups.update(c.split.unique());parts.append(c[c.epoch.isin([0,epochs])])

        assert np.all(counts>0) and len(np.unique(counts))==1;assert {'train','test_diagnostic'}.issubset(groups)

        q=pd.concat(parts);g=q[(q.epoch==epochs)&(q.split==('validation' if r.role=='inner' else 'test_diagnostic'))].copy()

        records.append(g.assign(method=r.name,fold=r.fold,role=r.role,holdout=r.holdout,seed=r.seed))

        key=(r.fold,r.role,r.holdout if r.role=='inner' else None,r.seed)

        zero=q[(q.epoch==0)&(q.split==('validation' if r.role=='inner' else 'test_diagnostic'))].copy()

        if key not in baseline_seen:

            records.append(zero.assign(method='independent_500',fold=r.fold,role=r.role,holdout=r.holdout,seed=r.seed));baseline_seen.add(key);baseline_frames[key]=zero.reset_index(drop=True)

        else:

            pd.testing.assert_frame_equal(baseline_frames[key],zero.reset_index(drop=True),check_exact=True)

        audit.append(dict(id=r.id,epochs=epochs,prediction_rows=int(counts.sum()),seconds=met['seconds'],monitor_change=met['monitor_max_change'],ode_change=met['ode_max_change']))

    p=pd.concat(records);p.to_csv(root/'predictions.csv.gz',index=False);selection=[]

    for (fold,family),g in p[(p.role=='inner')&(p.task=='monitoring')&p.method.str.startswith(('frozen','joint'))].assign(family=lambda q:q.method.str.split('_').str[0]).groupby(['fold','family']):

        score=g.assign(ae=abs(g.predicted_vb_mm-g.target_vb_mm)).groupby(['method','case']).ae.mean().groupby('method').mean();winner=score.sort_index().idxmin()

        for name,value in score.items():selection.append(dict(fold=fold,family=family,method=name,validation_mae=value,selected=name==winner))

    sel=pd.DataFrame(selection);sel.to_csv(root/'inner_selection.csv',index=False)

    for task in ['monitoring','forecasting']:

        f=p[(p.task==task)&(p.role=='final')];extras=[]

        for r in sel[sel.selected].itertuples():extras.append(f[(f.fold==r.fold)&(f.method==r.method)].assign(method='selected_'+r.family))

        keys=['method']+(['horizon'] if task=='forecasting' else [])

        if f.empty:continue

        per,s=metrics(pd.concat([f]+extras),keys);per.to_csv(root/f'{task}_case_metrics.csv',index=False);s.to_csv(root/f'{task}_summary.csv',index=False)

        keycols=['case','origin_run','target_run','horizon']

        for _,g in f.groupby(['fold','seed']):

            sets=[set(map(tuple,x[keycols].to_numpy())) for _,x in g.groupby('method')];assert all(x==sets[0] for x in sets)

    pd.DataFrame(audit).to_csv(root/'training_audit.csv',index=False)

    if epochs==500:

        assert len(m)==120

        summary=pd.read_csv(root/'monitoring_summary.csv');main=summary[summary.subset.isin(['overall','fold_a','fold_b'])].pivot(index='method',columns='subset',values='mae').reset_index()

        future=pd.read_csv(root/'forecasting_summary.csv');fs=future[future.subset=='overall'].pivot(index='method',columns='horizon',values='mae').reset_index();fs.columns=fs.columns.map(str)

        boot=[];rng=np.random.default_rng(20261014)

        for task in ['monitoring','forecasting']:

            a=pd.read_csv(root/f'{task}_case_metrics.csv')

            for method in ['selected_frozen','selected_joint']:

                for base in ['independent_500','continue_current']:

                    for horizon in ([None] if task=='monitoring' else [1,2,3]):

                        q=a if horizon is None else a[a.horizon==horizon];x=q[q.method==method].groupby(['fold','case']).mae.mean();y=q[q.method==base].groupby(['fold','case']).mae.mean();delta=x-y;draw=[]

                        for fold in ['fold_a','fold_b']:

                            vals=delta.loc[fold].to_numpy();draw.append(rng.choice(vals,(5000,len(vals))).mean(axis=1))

                        low,high=np.quantile(np.mean(draw,axis=0),[.025,.975]);boot.append(dict(task=task,method=method,baseline=base,horizon=horizon,delta=delta.mean(),low=low,high=high))

        pd.DataFrame(boot).to_csv(root/'case_bootstrap.csv',index=False)

        text='# 单次传播联合微调\n\n'+md(main)+'\n\n## 未来VB\n\n'+md(fs)+'\n\n## 内层当前VB选择未来损失权重\n\n'+md(sel[sel.selected])+'\n\n独立500轮基线复用；其他配置从同一500轮预训练monitor与独立ODE出发，再训练500轮。continue_current为相同优化预算的当前监督续训对照。frozen更新monitor且未来损失穿过冻结ODE；joint同时更新两分支。所有方法保留当前VB监督，无derivative loss、无递推校正、无EKF和学习alpha。未来h1/2/3使用实际dt，预测起点是monitor输出而非真实VB，不使用未来sensor。\n\n归一化wear/condition/time尺度逐项一致，sensor scaler独立按原split保留。完整逐epoch含epoch0的训练/验证/测试当前与未来预测保存。固定最后微调500轮，不按测试选epoch；两个未来权重只按内层当前MAE选择。外层138当前目标，未来目标与独立基线成对一致；本批单次传播包括全部有标签起点，不能直接与排除初始化点的递推融合MAE对比。五seed不作独立case，研究探索性。\n'

        (root/'RESULTS.md').write_text(text,encoding='utf-8')

    write_json(root/'completion_audit.json',dict(status='passed',runs=len(m),epochs=len(m)*epochs,prediction_rows=sum(x['prediction_rows'] for x in audit)))

    write_json(root/'status.json',dict(status='completed' if epochs==500 else 'smoke_completed',runs=len(m),training_seconds=sum(x['seconds'] for x in audit),report_seconds=time.perf_counter()-start))





def main():

    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--report-only',action='store_true');args=ap.parse_args();root=ROOT.with_name(ROOT.name+'_smoke') if args.smoke else ROOT

    try:

        if not args.report_only:

            tasks=plan(root)

            if not(args.execute or args.smoke):return

            if args.smoke:tasks=[t for t in tasks if t['fold']=='fold_a' and t['role']=='inner' and t['holdout']==1 and t['name'] in ['continue_current','frozen_l1','joint_l1']]

            rows=[]

            for task in tasks:

                d=root/'runs'/task['id'];execute(task,d,3 if args.smoke else 500);rows.append({**{k:v for k,v in task.items() if k not in ['sensor_dir','ode_dir']},'run_dir':str(d)})

                pd.DataFrame(rows).to_csv(root/'study_manifest.csv',index=False);write_json(root/'status.json',dict(status='training',completed=len(rows),planned=len(tasks)))

        report(root,3 if args.smoke else 500)

    except Exception as e:write_json(root/'failure.json',dict(error=str(e)));write_json(root/'status.json',dict(status='failed',error=str(e)));raise

if __name__=='__main__':main()

