"""Evaluation-only phases 2/3/4, with fold-internal fixed-gain selection."""

import json,time

from pathlib import Path

import numpy as np,pandas as pd,torch

from ..config import project_path,write_json,file_hash

from ..data.records import load_table

from ..data.feature_archive import attach_features

from ..data.sensor_quality import apply_quality

from ..data.monitor_batch import make_monitor_batch

from ..data.datasets import tensors

from ..models.registry import build_model,build_dynamics

from ..evaluation.fixed_fusion import rollout



ALPHAS=[0.,.25,.5,.75,1.]



def monitor(directory):

    d=Path(directory);cfg=json.loads((d/'provenance.json').read_text())['config'];sc=json.loads((d/'scalers.json').read_text());t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']));t=apply_quality(t,cfg['data'].get('sensor_quality'))

    if cfg['data'].get('feature_archive'):t=attach_features(t,cfg['data']['feature_archive'])

    b=make_monitor_batch(cfg,t,sc,'cpu');model=build_model(cfg);model.load_state_dict(torch.load(d/'last.pt',map_location='cpu',weights_only=True)['model']);model.eval()

    with torch.no_grad():z=model.estimate(b,np.arange(len(t['case']))).numpy().ravel()*sc['wear'][1]+sc['wear'][0]

    return cfg,t,z



def evaluate_one(r,ode_dir,root):

    cfg,t,sensor=monitor(r.run_dir);_,cttable,ct=monitor(Path(cfg['monitor']['fallback_checkpoint']).parent);assert np.array_equal(t['case'],cttable['case']) and np.array_equal(t['run'],cttable['run'])

    od=Path(ode_dir);oc=json.loads((od/'provenance.json').read_text())['config'];sc=json.loads((od/'scalers.json').read_text());sp=json.loads((Path(r.run_dir)/'split.json').read_text());osp=json.loads((od/'split.json').read_text());assert sp==osp

    model=build_dynamics(oc['dynamics']);model.load_state_dict(torch.load(od/'last.pt',map_location='cpu',weights_only=True)['model']);model.eval();condition=np.column_stack([(np.column_stack([t['DOC'],t['feed']])-np.asarray(sc['condition'][0]))/np.asarray(sc['condition'][1]),t['material']==1,t['material']==2]);batch={'condition':torch.tensor(condition,dtype=torch.float32)};cases=sp['validation'] or sp['test'];current=[];future=[];p3=[]

    @torch.no_grad()

    def prop(values,idx,delta):

        v=torch.as_tensor(np.asarray(values).reshape(-1,1),dtype=torch.float32);v=(v-sc['wear'][0])/sc['wear'][1]

        condition=batch['condition'][idx:idx+1].expand(len(v),-1);dt=torch.as_tensor(np.broadcast_to(delta,(len(v),)).copy().reshape(-1,1),dtype=torch.float32)/sc['time_scale']

        out=model.propagate(v,condition,dt);return out.numpy().ravel()*sc['wear'][1]+sc['wear'][0]

    for case in cases:

        ids=np.flatnonzero(t['case']==case);ids=ids[np.argsort(t['time'][ids])];labeled=ids[np.isfinite(t['VB'][ids])]

        for originpos,i in enumerate(labeled):

            targets=labeled[originpos+1:originpos+4]

            for h,j in enumerate(targets,1):

                initial=[sensor[i],ct[i],t['VB'][i]];f=prop(initial,i,t['time'][j]-t['time'][i])

                for k,name in enumerate(['sensor_initial','ct_initial','oracle_initial']):p3.append(dict(case=int(case),origin_run=int(t['run'][i]),target_run=int(t['run'][j]),horizon=h,method=name,target_vb_mm=t['VB'][j],predicted_vb_mm=f[k]))

        for start_kind in ['first_record','first_labeled']:

            seq=ids if start_kind=='first_record' else ids[np.flatnonzero(ids==labeled[0])[0]:]

            configs=[dict(name=f'{source}_a{a:g}_m{int(clamp)}',source=source,alpha=a,monotone=clamp) for source in ['sensor','ct'] for a in ALPHAS for clamp in [False,True]]

            configs.append(dict(name='ode_ct_initial',source='ct',alpha=0.,monotone=False))

            if start_kind=='first_labeled':configs.append(dict(name='ode_oracle_initial',source='ct',alpha=0.,monotone=False))

            obs=np.column_stack([sensor[seq] if q['source']=='sensor' else ct[seq] for q in configs]);valid=np.column_stack([t['sensor_valid'][seq] if q['source']=='sensor' else np.ones(len(seq),bool) for q in configs]);initial=np.full(len(configs),sensor[seq[0]])

            initial[20]=ct[seq[0]]

            if len(configs)>21:initial[21]=t['VB'][seq[0]]

            prior,post=rollout(t['time'][seq],obs,valid,[q['alpha'] for q in configs],[q['monotone'] for q in configs],initial,lambda v,dt:prop(v,seq[0],dt))

            assert np.all(t['DOC'][seq]==t['DOC'][seq[0]]) and np.all(t['feed'][seq]==t['feed'][seq[0]]) and np.all(t['material'][seq]==t['material'][seq[0]])

            for pos,i in enumerate(seq):

                for k,q in enumerate(configs):current.append(dict(case=int(case),run=int(t['run'][i]),time=t['time'][i],start_kind=start_kind,is_initial=pos==0,sensor_valid=bool(t['sensor_valid'][i]),method=q['name'],source=q['source'],alpha=q['alpha'],monotone=q['monotone'],prior_vb_mm=prior[pos,k],predicted_vb_mm=post[pos,k],target_vb_mm=t['VB'][i],sensor_vb_mm=sensor[i],ct_vb_mm=ct[i]))

                if not np.isfinite(t['VB'][i]):continue

                loc=np.flatnonzero(labeled==i)[0]

                for h,j in enumerate(labeled[loc+1:loc+4],1):

                    pred=prop(post[pos],i,t['time'][j]-t['time'][i])

                    for k,q in enumerate(configs):future.append(dict(case=int(case),origin_run=int(t['run'][i]),target_run=int(t['run'][j]),horizon=h,delta_time=t['time'][j]-t['time'][i],start_kind=start_kind,origin_is_initial=pos==0,method=q['name'],source=q['source'],alpha=q['alpha'],monotone=q['monotone'],predicted_vb_mm=pred[k],target_vb_mm=t['VB'][j]))

    return [pd.DataFrame(x).assign(fold=r.fold,role=r.role,holdout=r.holdout,seed=r.seed) for x in [current,future,p3]]



def metrics(q,keys):

    q=q.copy();q['error']=q.predicted_vb_mm-q.target_vb_mm;q['ae']=abs(q.error);q['se']=q.error**2

    per=q.groupby(keys+['fold','seed','case'],dropna=False).agg(mae=('ae','mean'),mse=('se','mean'),bias=('error','mean')).reset_index()

    rows=[]

    for subset,mask in [('overall',np.ones(len(q),bool)),('fold_a',q.fold=='fold_a'),('fold_b',q.fold=='fold_b'),('fold_a_in',(q.fold=='fold_a')&(q.target_vb_mm<=.74)),('fold_a_out',(q.fold=='fold_a')&(q.target_vb_mm>.74))]:

        a=q[mask]

        if a.empty:continue

        c=a.groupby(keys+['fold','seed','case'],dropna=False).agg(mae=('ae','mean'),mse=('se','mean'),bias=('error','mean')).reset_index();rows.append(c.groupby(keys,dropna=False)[['mae','mse','bias']].mean().reset_index().assign(subset=subset))

    return per,pd.concat(rows,ignore_index=True)



def md(df):

    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])



def report(root,epochs=500):

    started=time.perf_counter();torch.set_num_threads(1);m=pd.read_csv(root/'study_manifest.csv');reused=pd.read_csv(root/'reused_ode_manifest.csv');assert len(m)==(38 if epochs==500 else 3)

    audit=[];inner=[];baseline=[]

    for r in m.itertuples():

        d=Path(r.run_dir);met=json.loads((d/'metrics.json').read_text());h=pd.read_csv(d/'history.csv');assert met['epochs']==epochs and h.epoch.tolist()==list(range(1,epochs+1)) and met['reload_verified']

        audit.append(dict(id=r.id,type=r.type,seconds=met['wall_seconds'],epochs=epochs))

        if r.type=='monitor':

            pred=pd.read_csv(d/'monitoring_predictions.csv').assign(fold=r.fold,seed=r.seed,case_role=r.role)

            if r.role=='inner':inner.append(pred)

            else:

                baseline.append(pred.assign(method='sensor12'));ct=pd.read_csv(Path(json.loads((d/'provenance.json').read_text())['config']['monitor']['fallback_checkpoint']).parent/'monitoring_predictions.csv');baseline.append(ct.assign(fold=r.fold,seed=r.seed,case_role='final',method='ct'))

    oo=pd.concat(inner);oo['residual']=oo.predicted_vb_mm-oo.vb_mm;oo.to_csv(root/'phase2_oof_residuals.csv',index=False);cal=[]

    for fold,g in oo.groupby('fold'):

        case=g.groupby('case').residual.agg(['mean',lambda x:(x*x).mean()]);bias=case['mean'].mean();mse=case.iloc[:,1].mean();lags=[]

        for caseid,c in g.groupby('case'):

            v=c.sort_values('time').residual.to_numpy();lags.append(dict(fold=fold,case=caseid,bias=v.mean(),mae=abs(v).mean(),lag1_correlation=float(np.corrcoef(v[:-1],v[1:])[0,1]) if len(v)>3 and np.std(v[:-1])>0 and np.std(v[1:])>0 else np.nan))

        cal.append(dict(fold=fold,cases=g.case.nunique(),points=len(g),bias_mm=bias,R_variance_mm2=max(0.,mse-bias*bias),MSE_mm2=mse,lag1_case_mean=pd.DataFrame(lags).lag1_correlation.mean()));pd.DataFrame(lags).to_csv(root/f'phase2_cases_{fold}.csv',index=False)

    pd.DataFrame(cal).to_csv(root/'phase2_calibration.csv',index=False)

    b=pd.concat(baseline).rename(columns={'vb_mm':'target_vb_mm'});b.to_csv(root/'phase1_predictions.csv',index=False);_,s=metrics(b,['method']);s.to_csv(root/'phase1_summary.csv',index=False)

    chunks=[[],[],[]];cache=root/'evaluation';cache.mkdir(exist_ok=True)

    for r in m[m.type=='monitor'].itertuples():

        matches=m[(m.type=='ode')&(m.fold==r.fold)&(m.holdout==r.holdout)] if r.role=='inner' else reused[(reused.fold==r.fold)&(reused.seed==r.seed)]

        assert len(matches)==1;od=Path(matches.iloc[0].run_dir);cfg=json.loads((od/'provenance.json').read_text())['config'];assert cfg['dynamics']['pretrain_seed']==r.seed

        # Existing outer ODE checkpoints are independent of the new sensor features.

        if r.role=='final':assert json.loads((od/'metrics.json').read_text())['epochs']==500

        result=evaluate_one(r,od,root)

        for target,frame,name in zip(chunks,result,['current','future','phase3']):

            frame.to_csv(cache/f'{r.fold}_{r.role}_{r.holdout}_{r.seed}_{name}.csv.gz',index=False);target.append(frame)

    current,future,p3=[pd.concat(x,ignore_index=True) for x in chunks]

    for x,name in [(current,'current'),(future,'future'),(p3,'phase3')]:assert np.isfinite(x.predicted_vb_mm).all();x.to_csv(root/f'{name}_predictions.csv.gz',index=False)

    cur=current[current.target_vb_mm.notna()&~current.is_initial];fut=future[~future.origin_is_initial]

    selection=[]

    for (fold,start,source,clamp),q in cur[(cur.role=='inner')&~cur.method.str.startswith('ode_')].groupby(['fold','start_kind','source','monotone']):

        scores=q.assign(ae=abs(q.predicted_vb_mm-q.target_vb_mm)).groupby(['alpha','case']).ae.mean().groupby('alpha').mean();alpha=float(scores.sort_index().idxmin());selection.append(dict(fold=fold,start_kind=start,source=source,monotone=clamp,alpha=alpha,validation_mae=float(scores[alpha]),method=f'{source}_a{alpha:g}_m{int(clamp)}'))

    selected=pd.DataFrame(selection);selected.to_csv(root/'selected_alpha.csv',index=False)

    for frame,name in [(cur,'current'),(fut,'future')]:

        out=frame[frame.role=='final'];extras=[]

        for r in selected.itertuples():extras.append(out[(out.fold==r.fold)&(out.start_kind==r.start_kind)&(out.method==r.method)].assign(method=f'selected_{r.source}_m{int(r.monotone)}'))

        keys=['method','start_kind']+(['horizon'] if name=='future' else []);per,summary=metrics(pd.concat([out]+extras),keys);per.to_csv(root/f'{name}_case_metrics.csv',index=False);summary.to_csv(root/f'{name}_summary.csv',index=False)

    per,summary=metrics(p3[p3.role=='final'],['method','horizon']);per.to_csv(root/'phase3_case_metrics.csv',index=False);summary.to_csv(root/'phase3_summary.csv',index=False)

    primary=pd.read_csv(root/'current_summary.csv');primary=primary[(primary.start_kind=='first_record')&(primary.subset.isin(['overall','fold_a','fold_b']))]

    text='# 12维电流观测与ODE固定权重融合\n\nPhase1/2/3/4已运行；不执行自适应滤波或联合训练。\n\n## 当前VB（剔除共同初始化点）\n\n'+md(primary)+'\n\n## 训练内OOF观测误差\n\n'+md(pd.DataFrame(cal))+'\n\nR仅用于描述，不参与本批固定alpha融合；bias单独报告，未用测试残差修正。未来滤波内层评估须在其训练case内部再次交叉校准。相关残差不能直接解释为独立测量噪声。\n\n## 内层选择alpha\n\n'+md(selected)+'\n\n主协议first_record，first_labeled是与oracle可比的独立诊断。所有融合alpha及ct观测对照共享sensor12初值；ode_ct_initial与ode_oracle_initial另列，不混入公平融合比较。无效sensor跳过校正，初次无效用冻结ct后备。无单独VB非负截断；单调版本仅max(上一后验,当前更新)，未来积分无sensor修正。h1/2/3表示后续有VB标注的记录步数，使用真实时间间隔，不等距。phase3为各有标签起点的独立短期预测。\n\n五seed不作独立case；外层此前已被查看，本研究为探索性。完整未来结果见future_summary.csv及phase3_summary.csv。所有逐epoch记录保留在manifest引用目录；重用模型不改写。\n'

    (root/'RESULTS.md').write_text(text,encoding='utf-8');pd.DataFrame(audit).to_csv(root/'run_audit.csv',index=False);write_json(root/'status.json',dict(status='completed' if epochs==500 else 'smoke_completed',runs=len(m),epochs=epochs,training_seconds=sum(x['seconds'] for x in audit),evaluation_seconds=time.perf_counter()-started))

