"""Phase6: evaluate a preregistered Q multiplier grid, without training."""

import json,time

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from ..config import write_json,file_hash

from .report_current12_phase5 import evaluate

from .report_current12_fusion import metrics,md



BASE=Path('wear_monitoring_dynamics/outputs/current12_fusion_20261013')

P5=BASE.with_name('current12_phase5_20261013')

ROOT=BASE.with_name('current12_phase6_q_20261013')

BETAS=[0.,.1,1.,10.,100.]





def diagnostics(frame):

    q=frame[(frame.role=='final')&~frame.is_initial].copy()

    y=q.target_vb_mm;ep=q.prior_vb_mm-y;eo=q.observation_vb_mm-y;ef=q.predicted_vb_mm-y

    q['prior_mae']=abs(ep);q['observation_mae']=abs(eo);q['posterior_mae']=abs(ef)

    for name,v in [('same_sign',(ep*eo)>0),('prior_worse',abs(ep)>abs(eo)),('fusion_worse_than_observation',abs(ef)>abs(eo)),('covered95',abs(ef)<=1.96*np.sqrt(q.P_posterior))]:q[name]=np.where(y.notna(),v,np.nan)

    q['gain_when_prior_worse']=np.where(y.notna()&(abs(ep)>abs(eo)),q.K,np.nan)

    q['drop']=q.delta < -1e-8;q['drop_mm']=np.maximum(-q.delta,0);q['near_zero_gain']=q.K<.05;q['near_one_gain']=q.K>.95

    cols=['prior_mae','observation_mae','posterior_mae','same_sign','prior_worse','fusion_worse_than_observation','gain_when_prior_worse','covered95','K','drop','drop_mm','clipped','near_zero_gain','near_one_gain']

    cases=q.groupby(['fold','seed','case','method','start_kind'])[cols].mean().reset_index()

    return cases,cases.groupby(['method','start_kind'])[cols].mean().reset_index()





def main():

    start=time.perf_counter();torch.set_num_threads(1);ROOT.mkdir(exist_ok=True);cache=ROOT/'evaluation';cache.mkdir(exist_ok=True)

    protocol=dict(betas=BETAS,bias_correction=False,selection='within source/fold/start, inner current case MAE',calibration_hash=file_hash(P5/'calibration_parameters.csv'),models_manifest_hash=file_hash(BASE/'study_manifest.csv'))

    pp=ROOT/'protocol.json'

    if pp.exists():assert json.loads(pp.read_text())==protocol

    else:write_json(pp,protocol)

    old=pd.read_csv(BASE/'study_manifest.csv');ode=pd.read_csv(BASE/'reused_ode_manifest.csv');params=pd.read_csv(P5/'calibration_parameters.csv');chunks=[[],[]];tasks=[]

    for r in old[old.type=='monitor'].itertuples():

        hold=None if r.role=='final' else int(r.holdout);p=params[(params.fold==r.fold)&(params.holdout.isna() if hold is None else params.holdout==hold)];configs=[]

        for source in ['ct','sensor']:

            a=p[p.source==source].iloc[0]

            assert hold is None or hold not in json.loads(a.calibration_cases)

            for beta in BETAS:

                for clamp in [False,True]:configs.append(dict(method=f'{source}_b{beta:g}_m{int(clamp)}',source=source,bias=0.,R=float(a.mse),P0=float(a.P0),q=float(a.q)*beta,clamp=clamp))

        o=ode[(ode.fold==r.fold)&(ode.seed==r.seed)] if hold is None else old[(old.fold==r.fold)&(old.type=='ode')&(old.holdout==hold)]

        prefix=f'{r.fold}_{r.role}_{hold}_{r.seed}';files=[cache/f'{prefix}_{name}.csv.gz' for name in ['current','future']]

        if all(f.exists() for f in files):result=[pd.read_csv(f) for f in files]

        else:

            result=evaluate(r.run_dir,o.iloc[0].run_dir,p,r.fold,r.role,r.seed,hold,configs)

            for df,path in zip(result,files):df.to_csv(path,index=False)

        for target,df in zip(chunks,result):target.append(df)

        tasks.append(dict(fold=r.fold,role=r.role,holdout=hold,seed=r.seed,monitor=r.run_dir,ode=o.iloc[0].run_dir));pd.DataFrame(tasks).to_csv(ROOT/'study_manifest.csv',index=False)

        write_json(ROOT/'status.json',dict(status='evaluating',completed=len(tasks),planned=24));print(prefix,flush=True)

    rawcur,rawfuture=[pd.concat(x,ignore_index=True) for x in chunks]

    rawcur.to_csv(ROOT/'current_predictions.csv.gz',index=False);rawfuture.to_csv(ROOT/'future_predictions.csv.gz',index=False)

    cur=rawcur[rawcur.target_vb_mm.notna()&~rawcur.is_initial];future=rawfuture[~rawfuture.origin_is_initial]

    selections=[]

    for (fold,source,startkind),g in cur[cur.role=='inner'].groupby(['fold','source','start_kind']):

        scores=g.assign(ae=abs(g.predicted_vb_mm-g.target_vb_mm)).groupby(['method','case']).ae.mean().groupby('method').mean()

        # Fixed tie policy: ascending beta, then no clamp before clamp.

        winner=min(scores.index,key=lambda x:(scores[x],float(x.split('_b')[1].split('_m')[0]),int(x[-1])))

        for method,score in scores.items():selections.append(dict(fold=fold,source=source,start_kind=startkind,method=method,validation_mae=score,selected=method==winner))

    sel=pd.DataFrame(selections);sel.to_csv(ROOT/'inner_selection.csv',index=False);audit=[]

    for df,name in [(cur,'current'),(future,'future')]:

        final=df[df.role=='final'];extras=[]

        for r in sel[sel.selected].itertuples():extras.append(final[(final.fold==r.fold)&(final.start_kind==r.start_kind)&(final.method==r.method)].assign(method='selected_'+r.source+'_q'))

        keys=['method','start_kind']+(['horizon'] if name=='future' else [])

        per,summary=metrics(pd.concat([final]+extras),keys);per.to_csv(ROOT/f'{name}_case_metrics.csv',index=False);summary.to_csv(ROOT/f'{name}_summary.csv',index=False)

        ref=pd.read_csv(P5/f'{name}_predictions.csv.gz');ref=ref[ref.target_vb_mm.notna()&~ref.is_initial] if name=='current' else ref[~ref.origin_is_initial]

        join=['fold','role','seed','case','start_kind']+(['run'] if name=='current' else ['origin_run','target_run','horizon'])

        for source in ['ct','sensor']:

            for clamp in [0,1]:

                refpart=ref[ref.method==f'{source}_bias0_m{clamp}'];part=df[df.method==f'{source}_b1_m{clamp}']

                both=part.merge(refpart,on=join,suffixes=('_new','_old'),validate='one_to_one');assert len(both)==len(part)==len(refpart)

                err=float(abs(both.predicted_vb_mm_new-both.predicted_vb_mm_old).max());assert err<1e-5;np.testing.assert_allclose(both.target_vb_mm_new,both.target_vb_mm_old)

                audit.append(dict(task=name,source=source,clamp=clamp,beta1_max_error=err))

        for key,g in final.groupby(['fold','seed','start_kind']):

            sets=[set(map(tuple,x[join].to_numpy())) for _,x in g.groupby('method')];assert len(sets)==20 and all(s==sets[0] for s in sets)

        prev=pd.read_csv(BASE/f'{name}_summary.csv');names=['selected_ct_m0','selected_ct_m1','selected_sensor_m0','selected_sensor_m1','sensor_a0_m0','sensor_a1_m0','ct_a1_m0']

        pd.concat([summary,prev[prev.method.isin(names)]]).to_csv(ROOT/f'{name}_comparison.csv',index=False)

    dc,ds=diagnostics(rawcur);dc.to_csv(ROOT/'diagnostic_cases.csv',index=False);ds.to_csv(ROOT/'diagnostics.csv',index=False)

    _,prior_diag=diagnostics(pd.read_csv(P5/'current_predictions.csv.gz'));prior_diag.to_csv(ROOT/'phase5_error_diagnosis.csv',index=False)

    rng=np.random.default_rng(20261013);boot=[]

    for task in ['current','future']:

        new=pd.read_csv(ROOT/f'{task}_case_metrics.csv');previous=pd.read_csv(BASE/f'{task}_case_metrics.csv')

        for source in ['ct','sensor']:

            for horizon in ([None] if task=='current' else [1,2,3]):

                a=new[(new.method=='selected_'+source+'_q')&(new.start_kind=='first_record')];b=previous[(previous.method=='selected_'+source+'_m1')&(previous.start_kind=='first_record')]

                if horizon is not None:a=a[a.horizon==horizon];b=b[b.horizon==horizon]

                delta=a.groupby(['fold','case']).mae.mean()-b.groupby(['fold','case']).mae.mean();draw=[]

                for fold in ['fold_a','fold_b']:

                    v=delta.loc[fold].to_numpy();draw.append(rng.choice(v,(5000,len(v))).mean(axis=1))

                lo,hi=np.quantile(np.mean(draw,axis=0),[.025,.975]);boot.append(dict(task=task,source=source,horizon=horizon,delta=delta.mean(),low=lo,high=hi))

    pd.DataFrame(boot).to_csv(ROOT/'case_bootstrap.csv',index=False)

    import matplotlib

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    s=pd.read_csv(ROOT/'current_summary.csv');fig,axes=plt.subplots(2,2,figsize=(11,8))

    for row,source in enumerate(['ct','sensor']):

        for col,fold in enumerate(['fold_a','fold_b']):

            ax=axes[row,col]

            for clamp in [0,1]:

                vals=[s[(s.method==f'{source}_b{b:g}_m{clamp}')&(s.start_kind=='first_record')&(s.subset==fold)].mae.iloc[0] for b in BETAS]

                ax.plot(range(5),vals,'o-',label=f'monotone={clamp}')

            ax.set_xticks(range(5),[str(b) for b in BETAS]);ax.set(title=f'{source} / {fold}',xlabel='Q multiplier beta',ylabel='Outer-test case MAE (mm)');ax.legend();ax.grid(alpha=.2)

    fig.suptitle('Fixed grid diagnostic; outer-test minima are NOT used for selection');fig.tight_layout();fig.savefig(ROOT/'q_grid.png',dpi=150);plt.close(fig)

    main=s[(s.start_kind=='first_record')&s.method.str.startswith('selected_')];text='# Phase6：Q倍率扫描\n\n'+md(main)+'\n\n## 内层选择（CT与sensor分开）\n\n'+md(sel[sel.selected])+'\n\n## 相对同来源固定alpha＋截断的case配对区间\n\n'+md(pd.DataFrame(boot))+'\n\n20固定配置×2fold×5seed=200条主外层评估，无新训练。另有一级内层验证与first_labeled诊断。所有模型500epoch，R/P0及初值复用Phase5，只改变q倍率；不去偏。当前MAE选择参数，未来不再选。固定网格外层曲线只描述，不用测试挑beta。\n\nbeta=1已逐点复现Phase5无去偏配置，全部配置目标键一致。单调、无效sensor、Jacobian核查沿用共用评估器。嵌套参数沿用已审计的OOF，不含当前验证case。case bootstrap先平均seed、按fold分层抽case；区间探索性，非五seed独立样本。先验/观测相关、first_labeled P0代理及q近似仍存在，截断只投影均值，区间非严格截断分布。未来无未来sensor及VB重置。\n'

    (ROOT/'RESULTS.md').write_text(text,encoding='utf-8');write_json(ROOT/'audit.json',dict(status='passed',beta1_checks=audit,training_runs=0,outer_configs=200));write_json(ROOT/'status.json',dict(status='completed',evaluation_seconds=time.perf_counter()-start))



if __name__=='__main__':

    try:main()

    except Exception as e:write_json(ROOT/'failure.json',dict(error=str(e)));raise

