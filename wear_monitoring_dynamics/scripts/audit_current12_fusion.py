"""Independent completion checks and readout for fixed-gain fusion."""

import json

from pathlib import Path

import numpy as np,pandas as pd,torch

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from ..config import project_path,digest,write_json

from ..data.records import load_table

from ..data.feature_archive import attach_features

from ..data.sensor_quality import apply_quality

from ..data.preprocessing import fit_scalers,forecasting_pairs

from ..data.splits import make_split

from ..models.registry import build_dynamics

from .run_psd_sweep import rank

R=project_path('wear_monitoring_dynamics/outputs/current12_fusion_20261013')



def md(df):

 return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])



def main():

 torch.set_num_threads(1);m=pd.read_csv(R/'study_manifest.csv');assert len(m)==38;rows=0

 for r in m.itertuples():

  d=Path(r.run_dir);c=json.loads((d/'provenance.json').read_text())['config'];s=json.loads((d/'scalers.json').read_text());sp=make_split(r.fold,r.role,None if pd.isna(r.holdout) else int(r.holdout));assert sp==json.loads((d/'split.json').read_text())

  t,_=load_table(project_path(c['data']['path']),project_path(c['data']['cache']));t=apply_quality(t,c['data']['sensor_quality'])

  if r.type=='monitor':

   t=attach_features(t,c['data']['feature_archive']);names=c['data']['feature_archive']['names'];scores,_=rank(t,sp['train']);ix=sorted([j for j,n in enumerate(names) if n.startswith(('smcAC_','smcDC_'))],key=lambda j:(-scores[j],names[j]))[:12];assert ix==c['monitor']['selected_indices'];assert c['data']['sensor_preprocessing']['indices']==ix

   parent=Path(c['monitor']['fallback_checkpoint']).parent;pc=json.loads((parent/'provenance.json').read_text())['config'];assert pc['experiment']['seed']==r.seed and json.loads((parent/'split.json').read_text())==sp

  assert digest(fit_scalers(t,sp['train'],c['data'].get('sensor_preprocessing')))==digest(s)

  met=json.loads((d/'metrics.json').read_text());h=pd.read_csv(d/'history.csv');assert met['epochs']==500 and met['reload_verified'] and h.epoch.tolist()==list(range(1,501))

  e=pd.read_csv(d/'epoch_predictions.csv.gz');rows+=len(e);assert np.isfinite(e[['target_vb_mm','predicted_vb_mm']]).all().all()

  for (split,task),g in e.groupby(['split','task']):

   cases=sp['outer_test_reserved'] if split=='test_diagnostic' else sp[split];keys=['case','origin_run','target_run','horizon'];first=g[g.epoch==1]

   if task=='monitoring':

    ids=np.flatnonzero(np.isin(t['case'],cases)&np.isfinite(t['VB']));expected={(int(t['case'][i]),int(t['run'][i]),int(t['run'][i]),0) for i in ids}

   else:expected={(int(t['case'][i]),int(t['run'][i]),int(t['run'][j]),int(k)) for i,j,k in forecasting_pairs(t,cases,[1,2,3])}

   assert set(map(tuple,first[keys].values))==expected;assert not g.duplicated(['epoch']+keys).any();assert g.groupby(keys).size().eq(500).all();assert g.groupby(keys).target_vb_mm.nunique().eq(1).all()

 current=pd.read_csv(R/'current_predictions.csv.gz');future=pd.read_csv(R/'future_predictions.csv.gz');p3=pd.read_csv(R/'phase3_predictions.csv.gz');p1=pd.read_csv(R/'phase1_predictions.csv');ref=p1[(p1.method=='ct')&(p1.seed==0)].set_index(['case','run']).target_vb_mm.to_dict();assert len(ref)==138

 for (method,seed),q in p1.groupby(['method','seed']):assert len(q)==138 and q.set_index(['case','run']).target_vb_mm.to_dict()==ref

 # Trace endpoints and common evaluation targets, no labels used in update equations.

 tracekeys=['fold','role','holdout','seed','case','start_kind','method'];mon=[]

 for key,q in current.groupby(tracekeys,dropna=False):

  q=q.sort_values('time');assert q.is_initial.sum()==1 and q.is_initial.iloc[0];assert (q.time.diff().dropna()>0).all();prior=q.prior_vb_mm.to_numpy();post=q.predicted_vb_mm.to_numpy();a=q.alpha.iloc[0];clamp=q.monotone.iloc[0];source=q.source.iloc[0];z=q.sensor_vb_mm.to_numpy() if source=='sensor' else q.ct_vb_mm.to_numpy();v=q.sensor_valid.to_numpy() if source=='sensor' else np.ones(len(q),bool);gain=np.where(v[1:],a,0);expected=prior[1:]+gain*(z[1:]-prior[1:]);expected=np.maximum(expected,post[:-1]) if clamp else expected;np.testing.assert_allclose(post[1:],expected,atol=1e-6,rtol=1e-6)

  if clamp:assert np.all(np.diff(post)>=-1e-7)

  if not q.method.iloc[0].startswith('ode_'):np.testing.assert_allclose(post[0],q.sensor_vb_mm.iloc[0],atol=1e-7)

  if key[1]=='final':

   diff=np.diff(post);mon.append(dict(fold=key[0],seed=key[3],case=key[4],start_kind=key[5],method=key[6],steps=len(diff),drops=int((diff < -1e-6).sum()),drop_fraction=np.mean(diff < -1e-6),mean_drop_mm=np.maximum(-diff,0).mean(),max_drop_mm=np.maximum(-diff,0).max()))

 # All fixed methods have identical current label pools within each start/role/seed.

 for _,q in current.groupby(['fold','role','holdout','seed','start_kind'],dropna=False):

  sets=[set(zip(g.case,g.run)) for _,g in q[q.target_vb_mm.notna()&~q.is_initial].groupby('method')];assert all(x==sets[0] for x in sets)

 for _,q in future.groupby(['fold','role','holdout','seed','start_kind'],dropna=False):

  sets=[set(zip(g.case,g.origin_run,g.target_run,g.horizon)) for _,g in q[~q.origin_is_initial].groupby('method')];assert all(x==sets[0] for x in sets)

 # Independently recompute future propagation in mm from saved posterior, across every split.

 reused=pd.read_csv(R/'reused_ode_manifest.csv')

 for r in m[m.type=='monitor'].itertuples():

  match=m[(m.type=='ode')&(m.fold==r.fold)&(m.holdout==r.holdout)] if r.role=='inner' else reused[(reused.fold==r.fold)&(reused.seed==r.seed)];od=Path(match.iloc[0].run_dir);sc=json.loads((od/'scalers.json').read_text());oc=json.loads((od/'provenance.json').read_text())['config'];model=build_dynamics(oc['dynamics']);model.load_state_dict(torch.load(od/'last.pt',map_location='cpu',weights_only=True)['model']);model.eval()

  q=future[(future.fold==r.fold)&(future.role==r.role)&(future.seed==r.seed)]

  cur=current[(current.fold==r.fold)&(current.role==r.role)&(current.seed==r.seed)]

  if r.role=='inner':q=q[q.holdout==r.holdout];cur=cur[cur.holdout==r.holdout]

  q=q.sample(min(80,len(q)),random_state=12);cur=cur.rename(columns={'run':'origin_run','predicted_vb_mm':'initial_vb_mm'});q=q.merge(cur[['case','origin_run','start_kind','method','initial_vb_mm']],on=['case','origin_run','start_kind','method'],validate='many_to_one');assert len(q)>0

  t,_=load_table(project_path(oc['data']['path']),project_path(oc['data']['cache']));lookup={(int(c),int(run)):i for i,(c,run) in enumerate(zip(t['case'],t['run']))};ids=[lookup[(int(c),int(run))] for c,run in zip(q.case,q.origin_run)];cond=np.column_stack([(np.column_stack([t['DOC'][ids],t['feed'][ids]])-sc['condition'][0])/sc['condition'][1],t['material'][ids]==1,t['material'][ids]==2]);v=(q.initial_vb_mm.to_numpy()-sc['wear'][0])/sc['wear'][1]

  with torch.no_grad():pred=model.propagate(torch.tensor(v[:,None],dtype=torch.float32),torch.tensor(cond,dtype=torch.float32),torch.tensor(q.delta_time.to_numpy()[:,None]/sc['time_scale'],dtype=torch.float32)).numpy().ravel()*sc['wear'][1]+sc['wear'][0]

  np.testing.assert_allclose(pred,q.predicted_vb_mm,atol=2e-5,rtol=2e-5)

 mon=pd.DataFrame(mon);mon.to_csv(R/'trajectory_monotonicity.csv',index=False);sel=pd.read_csv(R/'selected_alpha.csv');cur=current[current.target_vb_mm.notna()&~current.is_initial]

 for r in sel.itertuples():

  g=cur[(cur.role=='inner')&(cur.fold==r.fold)&(cur.start_kind==r.start_kind)&(cur.source==r.source)&(cur.monotone==r.monotone)&~cur.method.str.startswith('ode_')];scores=g.assign(ae=abs(g.predicted_vb_mm-g.target_vb_mm)).groupby(['alpha','case']).ae.mean().groupby('alpha').mean();assert float(scores.sort_index().idxmin())==r.alpha

 # Plot selected sensor/ct fusion against their common-initialization ODE and observation-only endpoints.

 final=current[(current.role=='final')&(current.start_kind=='first_record')];parts=[]

 for r in sel[sel.start_kind=='first_record'].itertuples():parts.append(final[(final.fold==r.fold)&(final.method==r.method)].assign(method=f'selected_{r.source}_m{int(r.monotone)}'))

 plot=pd.concat([final[final.method.isin(['sensor_a0_m0','sensor_a1_m0','ct_a1_m0'])]]+parts)

 fig,axs=plt.subplots(4,4,figsize=(18,13),layout='constrained')

 for ax,case in zip(axs.flat,sorted(final.case.unique())):

  q=plot[plot.case==case];real=q.drop_duplicates('run').sort_values('time');ax.plot(real.time,real.target_vb_mm,'ko',label='VB',markersize=3)

  for name in ['sensor_a1_m0','ct_a1_m0','selected_sensor_m1','selected_ct_m1']:

   g=q[q.method==name].groupby('time').predicted_vb_mm.mean();ax.plot(g.index,g.values,label=name)

  ax.set(title=f'Case {case}',xlabel='Time',ylabel='VB (mm)');ax.grid(alpha=.2)

 for ax in axs.flat[14:]:ax.set_visible(False)

 axs.flat[0].legend(fontsize=6);fig.savefig(R/'selected_trajectories.png',dpi=120);plt.close(fig)

 primary=pd.read_csv(R/'current_summary.csv');methods=['sensor_a0_m0','sensor_a1_m0','ct_a1_m0','selected_sensor_m0','selected_sensor_m1','selected_ct_m0','selected_ct_m1'];summary=primary[(primary.start_kind=='first_record')&primary.method.isin(methods)].pivot(index='method',columns='subset',values='mae').reset_index();forecast=pd.read_csv(R/'future_summary.csv');ft=forecast[(forecast.start_kind=='first_record')&(forecast.subset=='overall')&forecast.method.isin(methods)].pivot(index='method',columns='horizon',values='mae').reset_index();ft.columns=ft.columns.map(str)

 chosen=[]

 for r in sel[sel.start_kind=='first_record'].itertuples():chosen.append(mon[(mon.fold==r.fold)&(mon.start_kind==r.start_kind)&(mon.method==r.method)].assign(method=f'selected_{r.source}_m{int(r.monotone)}'))

 mono=pd.concat([mon[(mon.start_kind=='first_record')&mon.method.isin(methods)]]+chosen).groupby('method')[['drop_fraction','mean_drop_mm','max_drop_mm']].mean().reset_index();mono.to_csv(R/'monotonicity_summary.csv',index=False)

 counts=final[final.target_vb_mm.notna()&~final.is_initial].groupby(['seed','method']).size();print('final label counts',counts.unique());print(mono.to_string(index=False))

 for task in ['current','future']:

  c=pd.read_csv(R/f'{task}_case_metrics.csv');c=c[(c.start_kind=='first_record')&c.method.isin(methods)];c.groupby(['method','fold','case']+(['horizon'] if task=='future' else []))[['mae','mse','bias']].mean().reset_index().to_csv(R/f'{task}_selected_case_means.csv',index=False)

 text='''\n\n## 完成审计与主要结论



38次500epoch新训练（24 MLP＋14内层ODE）耗时28.54分钟；基础评估1.73分钟。复用10外层独立ODE、24ct，没有重训。逐epoch任务目标键、训练内top12排名/scaler/clip/后备、验证case隔离、固定末轮重载均核查。独立重算每split抽样未来传播检查物理mm与ODE标准化转换；alpha端点、递推更新、无效sensor跳过、单调性和内层alpha选择均通过。只抽样重算未来数值，其余配对与更新公式全量检查。



Phase1全部138标注点：12维sensor MAE0.09323，ct0.08349。下面融合主表剔除共同初始化点，因此分母不同；不能直接把0.09323与融合0.09300作收益比较。first_record实际每seed为128个标签，first_labeled排除14初始标签后为124点。



### 当前VB：相同初始化、相同评价点



'''+md(summary)+'''\n

sensor融合比同池sensor-only有所改善：不截断0.09768→0.09300，截断0.09208。但仍不及ct-only的0.08739。带截断的ct观测融合为0.08179；注意它仍用sensor12初始化，且fold B验证选择alpha0，即之后完全不使用ct更新。因此不能把它称为纯无sensor系统，也不能说两个fold都证明了融合互补。



主协议sensor融合fold A选择alpha1、fold B选择alpha0.25；fold A并未选择ODE与sensor的中间融合。ct融合不截断为A0.5/B0；截断为A0.75/B0。first_labeled选择结果不同，说明起点设定有实际影响，应独立报告。



纯ODE从sensor估计起点递推，fold A误差0.24538、fold B0.07835，存在明显差异。fold A范围外MAE1.30307，长期外推不稳定；观察到真实VB初值h3也有较大范围外误差，不能将全部漂移归因于sensor初始化。



### 未来VB：后验出发后不再看未来sensor



'''+md(ft)+'''\n

sensor融合不截断h1/h2改善，但h3比sensor-only略差；单调截断也没有保证未来每个horizon改善。ct观测融合加截断在这组主对照中较好。上述h1/2/3是未来有VB标签记录步数，并非固定时长。



Phase3每标注起点单独预测（包含初始origin，池与上表不同）：oracle h1/2/3=0.04544/0.08373/0.12051，sensor初值=0.11686/0.14863/0.18786，ct初值=0.10451/0.14214/0.18379。实测初值是诊断参考，不是可部署结果，也不是数学上的严格性能上界。



### 观测误差与单调性



OOF R为fold A0.02287、B0.02521 mm²，bias为−0.00713、+0.01115mm；case内lag1残差相关均值约0.439/0.595。残差存在连续偏差，未来用简单独立白噪声测量假设需明确其近似性。本批R未参与任何融合调参，不存在用内层被评估标签校准再测试的操作。



下表drop_fraction为先逐case/seed计算相邻下降比例再平均，mean_drop为每步平均下降量，max_drop_mm列为逐case最大下降的平均（非全数据最大）：



'''+md(mono)+'''\n

截断分支无下降是算法保证，不能当成独立精度证据。逐case/seed明细见trajectory_monotonicity.csv。轨迹见selected_trajectories.png。current_selected_case_means.csv与future_selected_case_means.csv保留逐case对照；first_labeled及固定网格见原汇总文件。



没有启动Phase5或联合训练。所有结果属探索性；case相关性、稀疏VB和小样本限制不变，五seed不能视为独立case。

'''

 path=R/'RESULTS.md';body=path.read_text(encoding='utf-8').split('\n\n## 完成审计与主要结论')[0];path.write_text(body+text,encoding='utf-8');write_json(R/'completion_audit.json',dict(passed=True,runs=38,epochs=19000,epoch_prediction_rows=rows,phase1_targets=138,fusion_first_record_targets_per_seed=int(counts.iloc[0])));print('audit complete',rows)

if __name__=='__main__':main()

