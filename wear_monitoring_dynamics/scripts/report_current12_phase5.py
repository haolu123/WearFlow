"""Frozen scalar EKF evaluation; calibration uses case-held-out predictions only."""

import json,time

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from ..config import write_json

from ..models.registry import build_dynamics

from ..evaluation.kalman_fusion import update

from .report_current12_fusion import monitor,metrics,md

from .run_current12_phase5 import PARENT



FLOOR=1e-10  # mm^2, numerical only



class PhysicalODE:

    def __init__(self,directory,table):

        d=Path(directory);self.sc=json.loads((d/'scalers.json').read_text());cfg=json.loads((d/'provenance.json').read_text())['config']

        self.model=build_dynamics(cfg['dynamics']).eval();self.model.load_state_dict(torch.load(d/'last.pt',map_location='cpu',weights_only=True)['model']);self.model.requires_grad_(False)

        self.conditions=torch.tensor(np.column_stack([(np.column_stack([table['DOC'],table['feed']])-np.asarray(self.sc['condition'][0]))/np.asarray(self.sc['condition'][1]),table['material']==1,table['material']==2]),dtype=torch.float32)

    def propagate(self,values,index,dt,jacobian=False):

        v=torch.tensor(np.asarray(values).reshape(-1,1),dtype=torch.float32,requires_grad=jacobian);c=self.conditions[index:index+1].expand(len(v),-1)

        delta=torch.full_like(v,float(dt)/self.sc['time_scale'])

        with torch.set_grad_enabled(jacobian):

            y=self.model.propagate((v-self.sc['wear'][0])/self.sc['wear'][1],c,delta)*self.sc['wear'][1]+self.sc['wear'][0]

            f=torch.autograd.grad(y.sum(),v)[0].detach().numpy().ravel() if jacobian else None

        return y.detach().numpy().ravel(),f





def extract_calibration(sensor_dir,ode_dir,cases):

    cfg,t,sensor=monitor(sensor_dir);_,tc,ct=monitor(Path(cfg['monitor']['fallback_checkpoint']).parent)

    assert np.array_equal(t['case'],tc['case']) and np.array_equal(t['run'],tc['run'])

    sp=json.loads((Path(sensor_dir)/'split.json').read_text());osp=json.loads((Path(ode_dir)/'split.json').read_text())

    assert sp==osp and not set(cases)&set(sp['train'])

    ode=PhysicalODE(ode_dir,t);rows=[]

    for case in cases:

        ids=np.flatnonzero((t['case']==case)&np.isfinite(t['VB']));ids=ids[np.argsort(t['time'][ids])]

        for pos,i in enumerate(ids):

            for source,pred in [('sensor',sensor),('ct',ct)]:

                rows.append(dict(case=case,run=t['run'][i],kind=source,error=pred[i]-t['VB'][i],valid=bool(t['sensor_valid'][i]) if source=='sensor' else True,first_labeled=pos==0,dt=np.nan))

            if pos:

                j=ids[pos-1];dt=t['time'][i]-t['time'][j];p,_=ode.propagate([t['VB'][j]],j,dt)

                rows.append(dict(case=case,run=t['run'][i],kind='ode',error=p[0]-t['VB'][i],valid=True,first_labeled=False,dt=dt))

    return pd.DataFrame(rows).assign(sensor_dir=str(sensor_dir),ode_dir=str(ode_dir),train_cases=json.dumps(sp['train']))





def calibration(root):

    old=pd.read_csv(PARENT/'study_manifest.csv');nested=pd.read_csv(root/'study_manifest.csv');allrows=[];parameters=[]

    for fold in ['fold_a','fold_b']:

        pool=sorted(map(int,old[(old.fold==fold)&(old.role=='inner')].holdout.dropna().astype(int).unique()))

        for holdout in [None]+pool:

            cache=root/'calibration'/f'{fold}_{holdout}.csv';cache.parent.mkdir(exist_ok=True)

            if cache.exists():res=pd.read_csv(cache)

            else:

                pieces=[]

                for case in pool:

                    if case==holdout:continue

                    if holdout is None:

                        q=old[(old.fold==fold)&(old.role=='inner')&(old.holdout==case)]

                        sd=q[q.type=='monitor'].iloc[0].run_dir;od=q[q.type=='ode'].iloc[0].run_dir

                    else:

                        a,b=sorted([case,holdout]);q=nested[(nested.fold==fold)&(nested.held_a==a)&(nested.held_b==b)]

                        sd=q[q.type=='sensor'].iloc[0].run_dir;od=q[q.type=='ode'].iloc[0].run_dir

                    pieces.append(extract_calibration(sd,od,[case]))

                res=pd.concat(pieces,ignore_index=True);res.to_csv(cache,index=False)

            allowed=set(pool)-({holdout} if holdout is not None else set());assert set(res.case)==allowed

            for r in res.itertuples():assert r.case not in json.loads(r.train_cases) and (holdout is None or holdout not in json.loads(r.train_cases))

            initial=res[(res.kind=='sensor')&res.first_labeled];p0=float(initial.error.pow(2).mean())

            o=res[res.kind=='ode'].copy();o['rate_mse']=o.error.pow(2)/o.dt

            q=float(o.groupby('case').rate_mse.mean().mean())

            for source in ['ct','sensor']:

                g=res[(res.kind==source)&res.valid];bias=float(g.groupby('case').error.mean().mean());mse=float(g.assign(se=g.error**2).groupby('case').se.mean().mean())

                var=max(FLOOR,mse-bias*bias);lags=[]

                for _,c in g.groupby('case'):

                    e=c.sort_values('run').error.to_numpy()

                    if len(e)>3 and np.std(e[:-1])>0 and np.std(e[1:])>0:lags.append(float(np.corrcoef(e[:-1],e[1:])[0,1]))

                parameters.append(dict(fold=fold,holdout=holdout,source=source,bias=bias,mse=max(FLOOR,mse),variance=var,P0=max(FLOOR,p0),q=max(0.,q),lag1=float(np.mean(lags)),calibration_cases=json.dumps(sorted(allowed)),initial_proxy='first_labeled_sensor_MSE'))

            allrows.append(res.assign(fold=fold,holdout=holdout))

    pd.concat(allrows).to_csv(root/'calibration_residuals.csv.gz',index=False)

    params=pd.DataFrame(parameters);params.to_csv(root/'calibration_parameters.csv',index=False);return params





def evaluate(sensor_dir,ode_dir,params,fold,role,seed,holdout,configurations=None):

    cfg,t,sensor=monitor(sensor_dir);_,_,ct=monitor(Path(cfg['monitor']['fallback_checkpoint']).parent)

    sp=json.loads((Path(sensor_dir)/'split.json').read_text());ode=PhysicalODE(ode_dir,t)

    assert json.loads((Path(ode_dir)/'split.json').read_text())==sp

    cases=sp['validation'] if role=='inner' else sp['test'];configs=[]

    for source in ['ct','sensor']:

        p=params[params.source==source].iloc[0]

        for bias in [False,True]:

            for clamp in [False,True]:configs.append(dict(method=f'{source}_bias{int(bias)}_m{int(clamp)}',source=source,bias=float(p.bias) if bias else 0.,R=float(p.variance if bias else p.mse),P0=float(p.P0),q=float(p.q),clamp=clamp))

    if configurations is not None:configs=configurations

    nconfigs=len(configs)

    current=[];future=[];checks=[]

    for case in cases:

        ids=np.flatnonzero(t['case']==case);ids=ids[np.argsort(t['time'][ids])];labeled=ids[np.isfinite(t['VB'][ids])]

        for start_kind in ['first_record','first_labeled']:

            seq=ids if start_kind=='first_record' else ids[np.flatnonzero(ids==labeled[0])[0]:]

            state=np.full(nconfigs,sensor[seq[0]]);cov=np.array([c['P0'] for c in configs])

            for pos,i in enumerate(seq):

                previous=state.copy()

                if pos:

                    dt=t['time'][i]-t['time'][seq[pos-1]];prior,F=ode.propagate(previous,seq[pos-1],dt,True)

                    pcov=F**2*cov+np.array([c['q'] for c in configs])*dt

                else:prior=state.copy();pcov=cov.copy();F=np.ones(nconfigs)

                for k,c in enumerate(configs):

                    obs=(sensor[i] if c['source']=='sensor' else ct[i])-c['bias'];valid=bool(t['sensor_valid'][i]) if c['source']=='sensor' else True

                    if pos:state[k],cov[k],gain,clipped=update(prior[k],pcov[k],obs,c['R'],previous[k],valid,c['clamp'])

                    else:gain=0.;clipped=False

                    assert np.isfinite(state[k]) and np.isfinite(cov[k]) and cov[k]>=0

                    if pos and c['clamp']:assert state[k]>=previous[k]

                    if pos and not valid:assert abs(state[k]-prior[k])<1e-6 and gain==0

                    current.append(dict(case=int(case),run=int(t['run'][i]),time=t['time'][i],start_kind=start_kind,is_initial=pos==0,method=c['method'],source=c['source'],monotone=c['clamp'],target_vb_mm=t['VB'][i],predicted_vb_mm=state[k],prior_vb_mm=prior[k],observation_vb_mm=obs,sensor_valid=bool(t['sensor_valid'][i]),K=gain,P_prior=pcov[k],P_posterior=cov[k],F=F[k],R=c['R'],q=c['q'],clipped=clipped,delta=state[k]-previous[k]))

                if np.isfinite(t['VB'][i]):

                    loc=np.flatnonzero(labeled==i)[0]

                    for h,j in enumerate(labeled[loc+1:loc+4],1):

                        prediction,_=ode.propagate(state,i,t['time'][j]-t['time'][i])

                        for k,c in enumerate(configs):future.append(dict(case=int(case),origin_run=int(t['run'][i]),target_run=int(t['run'][j]),horizon=h,start_kind=start_kind,origin_is_initial=pos==0,method=c['method'],target_vb_mm=t['VB'][j],predicted_vb_mm=prediction[k]))

            # Independent finite-difference check of physical-unit Jacobian.

            if len(seq)>1:

                value=float(sensor[seq[0]]);dt=t['time'][seq[1]]-t['time'][seq[0]];eps=.001

                _,jac=ode.propagate([value],seq[0],dt,True);fd,_=ode.propagate([value+eps,value-eps],seq[0],dt)

                assert np.isclose(jac[0],(fd[0]-fd[1])/(2*eps),rtol=.01,atol=.002)

    return [pd.DataFrame(x).assign(fold=fold,role=role,seed=seed,holdout=holdout) for x in [current,future]]





def smoke_check(root):

    torch.set_num_threads(1);m=pd.read_csv(root/'study_manifest.csv');sd=m[m.type=='sensor'].iloc[0].run_dir;od=m[m.type=='ode'].iloc[0].run_dir

    residual=extract_calibration(sd,od,[1,2]);assert set(residual.case)=={1,2}

    params=pd.DataFrame([dict(source=s,bias=.01,mse=.01,variance=.0099,P0=.01,q=.0001) for s in ['ct','sensor']])

    a,b=evaluate(sd,od,params,'fold_a','inner',0,None);a.to_csv(root/'smoke_current.csv',index=False);b.to_csv(root/'smoke_future.csv',index=False)

    write_json(root/'status.json',dict(status='smoke_completed',runs=3,current_rows=len(a),future_rows=len(b)))





def audit_training(root):

    from ..data.records import load_table

    from ..data.feature_archive import attach_features

    from ..data.sensor_quality import apply_quality

    from ..data.preprocessing import fit_scalers

    from ..config import project_path

    from .run_psd_sweep import rank

    m=pd.read_csv(root/'study_manifest.csv');assert len(m)==126;rows=[]

    for r in m.itertuples():

        d=Path(r.run_dir);met=json.loads((d/'metrics.json').read_text());h=pd.read_csv(d/'history.csv');sp=json.loads((d/'split.json').read_text());cfg=json.loads((d/'provenance.json').read_text())['config']

        assert h.epoch.tolist()==list(range(1,501)) and met['epochs']==500 and met['reload_verified']

        assert len(sp['train'])==5 and set(sp['validation'])=={r.held_a,r.held_b}

        assert cfg['data']['sensor_quality']['rules']['train_cases']==sp['train']

        t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']));t=apply_quality(t,cfg['data']['sensor_quality'])

        if cfg['data'].get('feature_archive'):t=attach_features(t,cfg['data']['feature_archive'])

        sc=fit_scalers(t,sp['train'],cfg['data'].get('sensor_preprocessing'));assert json.loads(json.dumps(sc))==json.loads((d/'scalers.json').read_text())

        if r.type=='sensor':

            scores,_=rank(t,sp['train']);names=cfg['data']['feature_archive']['names'];ordered=sorted([j for j,n in enumerate(names) if n.startswith(('smcAC_','smcDC_'))],key=lambda j:(-scores[j],names[j]))

            assert cfg['monitor']['selected_indices']==ordered[:12]

            fallback=torch.load(cfg['monitor']['fallback_checkpoint'],map_location='cpu',weights_only=True);assert fallback['split']==sp

        counts=np.zeros(501,dtype=int);groups=set();prediction_rows=0

        for chunk in pd.read_csv(d/'epoch_predictions.csv.gz',chunksize=200000):

            assert np.isfinite(chunk.predicted_vb_mm).all();counts+=np.bincount(chunk.epoch,minlength=501);groups.update(chunk.split.unique());prediction_rows+=len(chunk)

        assert counts[0]==0 and np.all(counts[1:]>0) and np.unique(counts[1:]).size==1 and groups=={'train','validation','test_diagnostic'}

        rows.append(dict(id=r.id,seconds=met['wall_seconds'],prediction_rows=prediction_rows,epochs=500))

    pd.DataFrame(rows).to_csv(root/'training_audit.csv',index=False);return pd.DataFrame(rows)





def report(root):

    started=time.perf_counter();torch.set_num_threads(1);audit=audit_training(root);params=calibration(root)

    old=pd.read_csv(PARENT/'study_manifest.csv');reused=pd.read_csv(PARENT/'reused_ode_manifest.csv');chunks=[[],[]];cache=root/'evaluation';cache.mkdir(exist_ok=True)

    for r in old[old.type=='monitor'].itertuples():

        hold=None if r.role=='final' else int(r.holdout);p=params[(params.fold==r.fold)&(params.holdout.isna() if hold is None else params.holdout==hold)]

        prefix=f'{r.fold}_{r.role}_{hold}_{r.seed}';files=[cache/f'{prefix}_{name}.csv.gz' for name in ['current','future']]

        if all(f.exists() for f in files):result=[pd.read_csv(f) for f in files]

        else:

            o=reused[(reused.fold==r.fold)&(reused.seed==r.seed)] if hold is None else old[(old.fold==r.fold)&(old.type=='ode')&(old.holdout==hold)]

            result=evaluate(r.run_dir,o.iloc[0].run_dir,p,r.fold,r.role,r.seed,hold)

            for frame,path in zip(result,files):frame.to_csv(path,index=False)

        for target,frame in zip(chunks,result):target.append(frame)

        print('Evaluated '+prefix,flush=True)

    cur,fut=[pd.concat(x,ignore_index=True) for x in chunks]

    cur.to_csv(root/'current_predictions.csv.gz',index=False);fut.to_csv(root/'future_predictions.csv.gz',index=False)

    all_current=cur.copy()

    cur=cur[cur.target_vb_mm.notna()&~cur.is_initial];fut=fut[~fut.origin_is_initial]

    chosen=[]

    for (fold,start),g in cur[cur.role=='inner'].groupby(['fold','start_kind']):

        score=g.assign(ae=abs(g.predicted_vb_mm-g.target_vb_mm)).groupby(['method','case']).ae.mean().groupby('method').mean()

        for method,value in score.items():chosen.append(dict(fold=fold,start_kind=start,method=method,validation_mae=value,selected=method==score.sort_index().idxmin()))

    choices=pd.DataFrame(chosen);choices.to_csv(root/'inner_selection.csv',index=False)

    for frame,name in [(cur,'current'),(fut,'future')]:

        outer=frame[frame.role=='final'];extras=[]

        keycols=['case','run'] if name=='current' else ['case','origin_run','target_run','horizon']

        reference=pd.read_csv(PARENT/f'{name}_predictions.csv.gz')

        reference=reference[(reference.role=='final')&(reference.method=='sensor_a0_m0')]

        reference=reference[reference.target_vb_mm.notna()&~reference.is_initial] if name=='current' else reference[~reference.origin_is_initial]

        for (fold,start,seed),g in outer.groupby(['fold','start_kind','seed']):

            keys=[set(map(tuple,q[keycols].to_numpy())) for _,q in g.groupby('method')];assert len(keys)==8 and all(x==keys[0] for x in keys)

            ref=reference[(reference.fold==fold)&(reference.start_kind==start)&(reference.seed==seed)]

            assert keys[0]==set(map(tuple,ref[keycols].to_numpy())), 'Phase4/5 target mismatch'

            truth=g.drop_duplicates(keycols).merge(ref,on=keycols,suffixes=('_new','_old'),validate='one_to_one')

            np.testing.assert_allclose(truth.target_vb_mm_new,truth.target_vb_mm_old,rtol=0,atol=1e-12)

        for r in choices[choices.selected].itertuples():extras.append(outer[(outer.fold==r.fold)&(outer.start_kind==r.start_kind)&(outer.method==r.method)].assign(method='validation_selected'))

        keys=['method','start_kind']+(['horizon'] if name=='future' else [])

        per,summary=metrics(pd.concat([outer]+extras),keys);per.to_csv(root/f'{name}_case_metrics.csv',index=False);summary.to_csv(root/f'{name}_summary.csv',index=False)

        prior=pd.read_csv(PARENT/f'{name}_summary.csv');keep=['sensor_a0_m0','sensor_a1_m0','ct_a1_m0','selected_ct_m0','selected_ct_m1','selected_sensor_m0','selected_sensor_m1']

        pd.concat([summary,prior[prior.method.isin(keep)]]).to_csv(root/f'{name}_comparison.csv',index=False)

    diagnostic=all_current[(all_current.role=='final')&~all_current.is_initial].copy();diagnostic['covered95']=np.where(diagnostic.target_vb_mm.notna(),abs(diagnostic.predicted_vb_mm-diagnostic.target_vb_mm)<=1.96*np.sqrt(diagnostic.P_posterior),np.nan);diagnostic['drop']=diagnostic.delta < -1e-8;diagnostic['drop_mm']=np.maximum(-diagnostic.delta,0);diagnostic['near_zero_gain']=diagnostic.K<.05;diagnostic['near_one_gain']=diagnostic.K>.95

    dc=diagnostic.groupby(['fold','seed','case','method','start_kind'])[['covered95','drop','drop_mm','clipped','K','near_zero_gain','near_one_gain']].mean().reset_index();dc.to_csv(root/'case_filter_diagnostics.csv',index=False);dc.groupby(['method','start_kind'])[['covered95','drop','drop_mm','clipped','K','near_zero_gain','near_one_gain']].mean().to_csv(root/'filter_diagnostics.csv')

    # Bootstrap paired case means after averaging seeds, stratified by outer fold.

    oldcases=pd.read_csv(PARENT/'current_case_metrics.csv');newcases=pd.read_csv(root/'current_case_metrics.csv');boot=[];rng=np.random.default_rng(20261013)

    for method in newcases.method.unique():

        a=newcases[(newcases.method==method)&(newcases.start_kind=='first_record')].groupby(['fold','case']).mae.mean()

        for baseline in ['selected_ct_m1','selected_sensor_m1']:

            b=oldcases[(oldcases.method==baseline)&(oldcases.start_kind=='first_record')].groupby(['fold','case']).mae.mean();delta=(a-b).dropna();draw=[]

            for fold in ['fold_a','fold_b']:

                values=delta.loc[fold].to_numpy();draw.append(rng.choice(values,(5000,len(values)),replace=True).mean(axis=1))

            lo,hi=np.quantile(np.mean(draw,axis=0),[.025,.975]);boot.append(dict(method=method,baseline=baseline,delta_mae=delta.mean(),ci_low=lo,ci_high=hi))

    pd.DataFrame(boot).to_csv(root/'case_bootstrap.csv',index=False)

    plots(root,cur)

    s=pd.read_csv(root/'current_comparison.csv');main=s[(s.start_kind=='first_record')&s.subset.isin(['overall','fold_a','fold_b'])].pivot(index='method',columns='subset',values='mae').reset_index()

    fs=pd.read_csv(root/'future_comparison.csv');fs=fs[(fs.start_kind=='first_record')&(fs.subset=='overall')].pivot(index='method',columns='horizon',values='mae').reset_index();fs.columns=fs.columns.map(str)

    report_text='# Phase 5：固定R的标量EKF融合\n\n'+md(main)+'\n\n## 未来VB\n\n'+md(fs)+'\n\n## 内层选择\n\n'+md(choices[choices.selected])+'\n\n## 解释与限制\n\n所有模型固定500epoch；本次新增126次嵌套校准训练，42个五case子集，各ct/sensor/ODE一次seed0。外层和一级内层模型复用Phase4。八种固定配置与验证选择单独报告，未用测试选择方法或Q。\n\n校准case等权，sensor R排除无效sensor。未去偏使用OOF MSE作有效噪声尺度；去偏用全局case等权偏差及中心方差。P0使用first_labeled处sensor OOF MSE，因first_record可能没有VB，这只是共享初值误差代理；初值本身不去偏，所有配置相同。q=case等权mean(ODE相邻有标签记录传播残差平方/实际dt)，单位mm²/时间单位，不是直接测得的物理过程噪声。\n\n标量EKF采用自动微分状态转移Jacobian，Joseph方差更新。单调截断仅投影均值，没有推导截断分布；带截断的区间覆盖仅作诊断，不能宣称严格概率校准。观测误差有时间相关性，且先验/观测可能相关；当前滤波近似忽略这些相关项。\n\n当前VB去除初始化点，未来h1/2/3为后续有标签记录步数，真实时间间隔，无未来sensor输入或真实VB重置。first_record主协议，first_labeled诊断。固定alpha旧对照使用相同case/run目标；逐case配对bootstrap先平均seed再按case重抽样，五seed不当独立case。研究为探索性；无优势时不得称证明滤波收益。\n\n详见README.md、case_bootstrap.csv、filter_diagnostics.csv及逐记录预测。\n'

    (root/'RESULTS.md').write_text(report_text,encoding='utf-8')

    write_json(root/'completion_audit.json',dict(status='passed',runs=126,epochs=63000,prediction_rows=int(audit.prediction_rows.sum()),outer_configurations=80,calibration_groups=16))

    write_json(root/'status.json',dict(status='completed',runs=126,training_seconds=float(audit.seconds.sum()),evaluation_seconds=time.perf_counter()-started))





def plots(root,cur):

    import matplotlib

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    choices=pd.read_csv(root/'inner_selection.csv');p=cur[(cur.role=='final')&(cur.start_kind=='first_record')&(cur.seed==0)]

    fig,axes=plt.subplots(4,4,figsize=(17,13));used=0

    for r in choices[choices.selected&(choices.start_kind=='first_record')].itertuples():

        for case,g in p[(p.fold==r.fold)&(p.method==r.method)].groupby('case'):

            ax=axes.ravel()[used];used+=1;g=g.sort_values('time');ax.plot(g.time,g.target_vb_mm,'ko-',label='VB');ax.plot(g.time,g.predicted_vb_mm,label='Posterior');ax.plot(g.time,g.observation_vb_mm,alpha=.6,label='Observation');ax.set_title(f'case {case}, {r.method}',fontsize=9);ax.legend(fontsize=7)

    for ax in axes.ravel()[used:]:ax.set_visible(False)

    fig.tight_layout();fig.savefig(root/'selected_trajectories.png',dpi=150);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12,4))

    for ax,fold in zip(axes,['fold_a','fold_b']):

        for method,g in p[p.fold==fold].groupby('method'):ax.plot(g.groupby('time').K.mean(),label=method)

        ax.set_title(fold);ax.set_xlabel('Time');ax.set_ylabel('Mean gain across available cases');ax.legend(fontsize=7)

    fig.tight_layout();fig.savefig(root/'gain_curves.png',dpi=150);plt.close(fig)

