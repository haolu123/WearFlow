"""Fixed-alpha case-recursive joint training with matched single-shot control."""

import argparse,json,time

from pathlib import Path

import numpy as np

import pandas as pd

from ..config import write_json

from ..training.recursive_adaptation import CONFIGS,execute

from .report_current12_fusion import metrics,md



PARENT=Path('wear_monitoring_dynamics/outputs/current12_single_joint_20261014')

ROOT=PARENT.with_name('current12_recursive_joint_20261014')



def plan(root):

    root.mkdir(parents=True,exist_ok=True);m=pd.read_csv(PARENT/'study_manifest.csv');tasks=[]

    for r in m[m.name=='joint_l1'].itertuples():

        parent=json.loads((Path(r.run_dir)/'identity.json').read_text())['task']

        for name,mode,alpha,clamp in [('single','single',None,False)]+[(f'recursive_a{a:g}_m{int(c)}','recursive',a,c) for a,c in CONFIGS]:

            tasks.append(dict(id=f'{r.fold}/{r.role}_{parent["holdout"]}/{r.seed}/{name}',fold=r.fold,role=r.role,holdout=parent['holdout'],seed=int(r.seed),name=name,mode=mode,alpha=alpha,clamp=clamp,device='cpu',start_checkpoint=str(Path(r.run_dir)/'last.pt'),sensor_dir=parent['sensor_dir'],ode_dir=parent['ode_dir']))

    assert len(tasks)==168;write_json(root/'planned_tasks.json',tasks);return tasks





def report(root,epochs):

    start=time.perf_counter();m=pd.read_csv(root/'study_manifest.csv');records=[];audit=[]

    for r in m.itertuples():

        d=Path(r.run_dir);h=pd.read_csv(d/'history.csv');met=json.loads((d/'metrics.json').read_text());assert h.epoch.tolist()==list(range(1,epochs+1)) and met['reload_verified'];parts=[];counts=np.zeros(epochs+1,dtype=int)

        for chunk in pd.read_csv(d/'epoch_predictions.csv.gz',chunksize=200000):

            assert np.isfinite(chunk.predicted_vb_mm).all();counts+=np.bincount(chunk.epoch,minlength=epochs+1);parts.append(chunk[chunk.epoch.isin([0,epochs])])

        assert np.all(counts>0) and np.unique(counts).size==1

        q=pd.concat(parts);q=q[q.split==('validation' if r.role=='inner' else 'test_diagnostic')]

        last=q[q.epoch==epochs].assign(family=r.mode);records.append(last.assign(fold=r.fold,role=r.role,seed=r.seed,holdout=r.holdout))

        if r.mode=='single':records.append(q[q.epoch==0].assign(family='frozen',fold=r.fold,role=r.role,seed=r.seed,holdout=r.holdout))

        audit.append(dict(id=r.id,epochs=epochs,rows=int(counts.sum()),seconds=met['seconds']))

    p=pd.concat(records);p['method']=p.family+'_'+p.fusion;p.to_csv(root/'predictions.csv.gz',index=False);selections=[]

    for (fold,family),g in p[(p.role=='inner')&(p.task=='current')].groupby(['fold','family']):

        scores=g.assign(ae=abs(g.predicted_vb_mm-g.target_vb_mm)).groupby(['method','case']).ae.mean().groupby('method').mean()

        for method,score in scores.items():selections.append(dict(fold=fold,family=family,method=method,validation_mae=score,selected=method==scores.sort_index().idxmin()))

    sel=pd.DataFrame(selections);sel.to_csv(root/'inner_selection.csv',index=False)

    for task in ['monitor','current','future']:

        f=p[(p.role=='final')&(p.task==task)];extras=[]

        if f.empty:continue

        for r in sel[sel.selected].itertuples():extras.append(f[(f.fold==r.fold)&(f.method==r.method)].assign(method='selected_'+r.family))

        keys=['method']+(['horizon'] if task=='future' else []);per,s=metrics(pd.concat([f]+extras),keys);per.to_csv(root/f'{task}_case_metrics.csv',index=False);s.to_csv(root/f'{task}_summary.csv',index=False)

        for _,g in f.groupby(['fold','seed']):

            sets=[set(map(tuple,x[['case','origin_run','target_run','horizon']].to_numpy())) for _,x in g.groupby('method')];assert all(x==sets[0] for x in sets)

    pd.DataFrame(audit).to_csv(root/'training_audit.csv',index=False)

    if epochs==500:

        assert len(m)==168

        summary=pd.read_csv(root/'current_summary.csv');s=summary[summary.method.str.startswith('selected_')];future=pd.read_csv(root/'future_summary.csv');future=future[future.method.str.startswith('selected_')]

        text='# 完整case递推联合训练\n\n'+md(s)+'\n\n## 未来VB\n\n'+md(future)+'\n\n## 内层选择\n\n'+md(sel[sel.selected])+'\n\n168次微调均从上一批固定joint_l1（非测试选出）同split/seed模型出发，另500epoch。24个single对照仅训练一次，推理评估六组alpha/截断；144个recursive分别训练；冻结模型以epoch0复用。三组各自内层当前融合MAE选配置，全部固定配置单列。未来不挑参。\n\nL为case等权原始monitor当前监督＋输出当前监督＋未来h1/2/3监督，权重均1。single组用原始monitor替代融合输出，控制当前监督的总权重和标签池；不同于上批M+F的原始损失，不与其数值简单归因于训练方式。训练更新次数相同但计算量不同。\n\nfirst_record初始化，当前/未来起点剔除初始记录，raw monitor单列含全部标签。不用真实VB重置状态。跳过坏sensor校正但保留时间传播；初值无效仍用同split/seed冻结CT。全case无截断反传，没有跨case状态。单调max可能截断部分梯度，审计保留梯度值；不宣称所有路径总有非零梯度。所有逐轮训练/验证/测试原始预测保存，外层每轮checkpoint。已见过测试，研究探索性，五seed不当独立case。\n'

        (root/'RESULTS.md').write_text(text,encoding='utf-8')

    write_json(root/'status.json',dict(status='completed' if epochs==500 else 'smoke_completed',runs=len(m),training_seconds=sum(x['seconds'] for x in audit),report_seconds=time.perf_counter()-start))





def main():

    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--report-only',action='store_true');a=ap.parse_args();root=ROOT.with_name(ROOT.name+'_smoke') if a.smoke else ROOT

    try:

        if not a.report_only:

            tasks=plan(root)

            if not(a.execute or a.smoke):return

            if a.smoke:tasks=[t for t in tasks if t['fold']=='fold_a' and t['role']=='inner' and t['holdout']==1 and (t['mode']=='single' or t['alpha']==.5)]

            rows=[]

            for task in tasks:

                d=root/'runs'/task['id'];execute(task,d,3 if a.smoke else 500);rows.append({**{k:v for k,v in task.items() if k not in ['sensor_dir','ode_dir','start_checkpoint']},'run_dir':str(d)})

                pd.DataFrame(rows).to_csv(root/'study_manifest.csv',index=False);write_json(root/'status.json',dict(status='training',completed=len(rows),planned=len(tasks)))

        report(root,3 if a.smoke else 500)

    except Exception as e:write_json(root/'failure.json',dict(error=str(e)));write_json(root/'status.json',dict(status='failed',error=str(e)));raise

if __name__=='__main__':main()

