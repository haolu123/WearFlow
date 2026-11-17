"""Aggregate results on audited targets; finite coverage and common keys include seed."""

from pathlib import Path

import json

import numpy as np,pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'summary'

NAMES={'CT100':'工况＋时间 MLP（100轮）','sensor12_100':'电流top12 MLP（100轮）','gpr':'普通GPR','svr':'SVR','Zhang_PINN_top12':'Zhang PINN（NASA适配）','sun_pigpr_causal_nasa':'Sun PI-GPR（因果适配）','chen2025_pidggcn_causal_paper_printed':'Chen PIDGGCN（印刷导数）','chen2025_pidggcn_causal_calculus_corrected':'Chen PIDGGCN（修正导数）','Kim_TF_raw_ensemble':'Kim 原始波形Transformer集成','Kim_TF_cubic_mean_NASA':'Kim 三次曲线均值（未来VB扩展）','independent_monitor100_ode500':'A：独立monitor100＋ODE500','current_only_extra100':'B：额外100轮仅当前监督','frozen_ode_future_extra100':'C：额外100轮冻结ODE适配monitor','joint_future_extra100':'D：额外100轮联合适配','linear2_sensor100':'最近两点线性外推','linear2_nonnegative_sensor100':'最近两点非负斜率外推','persistence_sensor100':'当前估计保持不变','sun2020_lstm_module':'Sun LSTM磨损序列模块'}

CK=['fold','case','run'];FK=['fold','seed','case','origin_run','target_run','horizon']

def read(path,**values):

 f=pd.read_csv(path)

 for k,v in values.items():f[k]=v

 if 'seed' not in f:f['seed']=0

 return f



def summarize(df):

 rows=[];df=df.copy();assert np.isfinite(df.pred_mm).all();df['ae']=abs(df.pred_mm-df.target_mm);df['se']=(df.pred_mm-df.target_mm)**2

 for method,g in df.groupby('method'):

  subsets={'overall':np.ones(len(g),bool),'fold_a':g.fold=='fold_a','fold_b':g.fold=='fold_b','fold_a_in':(g.fold=='fold_a')&(g.target_mm<=.74),'fold_a_out':(g.fold=='fold_a')&(g.target_mm>.74)}

  for name,mask in subsets.items():

   q=g[mask]

   if not len(q):continue

   c=q.groupby(['fold','seed','case']).agg(mae=('ae','mean'),mse=('se','mean')).reset_index();c['rmse']=np.sqrt(c.mse);c=c.groupby(['fold','case'])[['mae','rmse']].mean()

   key=['fold','case','origin_run','target_run','horizon'] if 'origin_run' in q else CK

   rows.append(dict(method=method,subset=name,mae_mm=c.mae.mean(),case_macro_rmse_mm=c.rmse.mean(),cases=len(c),unique_targets=len(q.drop_duplicates(key)),seed_target_rows=len(q)))

 return pd.DataFrame(rows)

def markdown(df):

 df=df.copy()

 if 'method' in df:df['method']=df['method'].map(lambda x:NAMES.get(x,x))

 return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def common_future(g):

 valid=g[np.isfinite(g.pred_mm)].copy();sets=[set(map(tuple,valid[valid.method==m][FK].to_numpy())) for m in sorted(g.method.unique())];common=set.intersection(*sets)

 return valid[[tuple(x) in common for x in valid[FK].to_numpy()]],common



def main():

 OUT.mkdir(exist_ok=True);refs=read(ROOT/'references/current_predictions.csv')

 paths=['standard/results/outer_predictions_valid_only.csv','pigpr/formal/final_predictions.csv','zhang/formal/current_predictions.csv','pidggcn/formal/current_predictions.csv','kde/reports/current_predictions.csv.gz']

 assert all((ROOT/p).exists() for p in paths),'Require completed current result files'

 cur=pd.concat([refs]+[read(ROOT/p) for p in paths],ignore_index=True)

 truth=refs[CK+['target_mm']].drop_duplicates();expected=len(truth);checks=[]

 for (method,seed),g in cur.groupby(['method','seed']):

  assert len(g)==expected and not g.duplicated(CK).any() and np.isfinite(g.pred_mm).all(),(method,seed)

  j=g.merge(truth,on=CK,suffixes=('','_truth'),validate='one_to_one');assert len(j)==expected;np.testing.assert_allclose(j.target_mm,j.target_mm_truth,atol=1e-12,rtol=0)

  checks.append(dict(method=method,seed=int(seed),targets=len(j)))

 cs=summarize(cur);cs.to_csv(OUT/'current_summary.csv',index=False);cur.to_csv(OUT/'current_predictions.csv',index=False)

 cur.assign(ae=abs(cur.pred_mm-cur.target_mm)).groupby(['method','fold','seed','case']).ae.mean().reset_index().to_csv(OUT/'current_case_seed_metrics.csv',index=False)

 old=read(ROOT/'references/ours_current_predictions.csv');s=refs[refs.method=='sensor12_100'].merge(old[old.method=='independent_monitor100_ode500'],on=CK+['seed'],suffixes=('','_old'),validate='one_to_one');np.testing.assert_allclose(s.pred_mm,s.pred_mm_old,atol=2e-7,rtol=0)

 summarize(old).to_csv(OUT/'existing_internal_current_summary.csv',index=False)

 future=[read(ROOT/'references/future_predictions.csv'),read(ROOT/'references/ours_future_predictions.csv'),read(ROOT/'linear/future_predictions.csv'),read(ROOT/'kde/reports/future_predictions.csv.gz')]

 l=read(ROOT/'sun_lstm/results/final_predictions.csv')

 for col,value in [('partition','test'),('initialization','estimated'),('protocol','estimated')]:

  if col in l:l=l[l[col]==value]

 if 'task' in l:l=l[l.task!='oracle']

 future.append(l);f=pd.concat(future,ignore_index=True)

 canonical=f[f.method=='persistence_sensor100'][FK+['dt','target_mm']].rename(columns={'dt':'canonical_dt','target_mm':'canonical_target'})

 assert not canonical.duplicated(FK).any()

 merged=f.merge(canonical,on=FK,how='left',validate='many_to_one',indicator=True);assert (merged['_merge']=='both').all(),'Unknown future key'

 np.testing.assert_allclose(merged.target_mm,merged.canonical_target,atol=1e-12,rtol=0)

 observed=merged.dt.notna();np.testing.assert_allclose(merged.loc[observed,'dt'],merged.loc[observed,'canonical_dt'],atol=1e-12,rtol=0)

 merged['dt_from_reference']=~observed;merged['dt']=merged.dt.fillna(merged.canonical_dt);f=merged.drop(columns=['canonical_dt','canonical_target','_merge']);assert np.isfinite(f.dt).all() and (f.dt>0).all()

 assert not np.isinf(f.pred_mm).any()

 for name,g in f.groupby('method'):

  assert not g.duplicated(FK).any(),name

  assert set(g.seed)==set(range(5)) and set(g.fold)=={'fold_a','fold_b'},name

  for h,hg in g.groupby('horizon'):

   patterns=[]

   for seed,sg in hg.groupby('seed'):

    keys_without_seed=[k for k in FK if k!='seed'];patterns.append((set(map(tuple,sg[np.isfinite(sg.pred_mm)][keys_without_seed].to_numpy())),set(map(tuple,sg[~np.isfinite(sg.pred_mm)][keys_without_seed].to_numpy()))))

   assert len(patterns)==5 and all(p==patterns[0] for p in patterns[1:]),(name,h,'seed coverage differs')

 f.to_csv(OUT/'future_predictions.csv',index=False)

 summaries=[];coverage=[];common_details=[]

 for h in (1,2,3):

  g=f[f.horizon==h];valid=g[np.isfinite(g.pred_mm)];common_g,keys=common_future(g);assert len(common_g)>0

  for method,q in g.groupby('method'):

   for seed,p in q.groupby('seed'):

    good=p[np.isfinite(p.pred_mm)];c=common_g[(common_g.method==method)&(common_g.seed==seed)]

    coverage.append(dict(method=method,horizon=h,seed=int(seed),provided_targets=len(p),finite_targets=len(good),common_targets=len(c),common_cases=c.case.nunique(),canonical_targets=len(canonical[(canonical.horizon==h)&(canonical.seed==seed)])))

  for seed in sorted(g.seed.unique()):

   k=common_g[common_g.seed==seed][FK].drop_duplicates();common_details.append(dict(horizon=h,seed=int(seed),targets=len(k),cases=k[['fold','case']].drop_duplicates().shape[0]))

  for scope,q in [('all_available',valid),('all_methods_common',common_g)]:

   t=summarize(q);t['horizon']=h;t['scope']=scope;summaries.append(t)

 fs=pd.concat(summaries);fs.to_csv(OUT/'future_summary.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'future_coverage.csv',index=False);counts=pd.DataFrame(common_details);counts.to_csv(OUT/'common_target_counts.csv',index=False)

 # Common-set case/seed metrics make every comparison independently reconstructable.

 cm=[]

 for h in (1,2,3):

  g,_=common_future(f[f.horizon==h]);c=g.assign(ae=abs(g.pred_mm-g.target_mm)).groupby(['method','fold','seed','case']).ae.mean().reset_index();c['horizon']=h;cm.append(c)

 pd.concat(cm).to_csv(OUT/'future_common_case_seed_metrics.csv',index=False)

 allcase=[]

 for h in (1,2,3):

  for scope,q in [('all_available',f[(f.horizon==h)&np.isfinite(f.pred_mm)]),('all_methods_common',common_future(f[f.horizon==h])[0])]:

   for subset,mask in [('overall',np.ones(len(q),bool)),('fold_a_in',(q.fold=='fold_a')&(q.target_mm<=.74)),('fold_a_out',(q.fold=='fold_a')&(q.target_mm>.74))]:

    v=q[mask].assign(ae=lambda x:abs(x.pred_mm-x.target_mm)).groupby(['method','fold','seed','case']).ae.mean().reset_index();v['horizon']=h;v['scope']=scope;v['subset']=subset;allcase.append(v)

 pd.concat(allcase).to_csv(OUT/'future_case_seed_metrics.csv',index=False)

 cw=cs.pivot(index='method',columns='subset',values='mae_mm')[['overall','fold_a','fold_b','fold_a_in','fold_a_out']]

 fw=fs[(fs.subset=='overall')&(fs.scope=='all_methods_common')].pivot(index='method',columns='horizon',values='mae_mm');fw.columns=['h'+str(i) for i in fw.columns]

 fig,ax=plt.subplots(figsize=(11,7),layout='constrained');cw[['overall','fold_a','fold_b']].plot.barh(ax=ax);ax.set(xlabel='Case-macro MAE (mm)',title=f'Current VB: identical {expected} measured labels');ax.grid(axis='x',alpha=.2);fig.savefig(OUT/'current_mae.png',dpi=160);plt.close(fig)

 fig,ax=plt.subplots(figsize=(12,6),layout='constrained')

 for method,row in fw.iterrows():ax.plot([1,2,3],row.values,marker='o',label=method)

 ax.set(xticks=[1,2,3],xlabel='Horizon (subsequent labeled record)',ylabel='Case-macro MAE (mm)',title='Future VB: finite predictions on all-method common keys');ax.legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left');ax.grid(alpha=.2);fig.savefig(OUT/'future_mae.png',dpi=160);plt.close(fig)

 internal=pd.read_csv(ROOT/'internal/results/summary.csv');it=[]

 for arm in ['current_only','local_only','integral_only','both']:

  row={'arm':arm}

  for task in ['current','h1','h2','h3']:

   src='persistence_'+task if arm=='current_only' and task!='current' else task;x=internal[(internal.arm==arm)&(internal.task==src)&(internal.region=='overall')];assert len(x)==1;row[task]=float(x['mean'].iloc[0])

  it.append(row)

 pd.DataFrame(it).to_csv(OUT/'cold_start_internal_summary.csv',index=False)

 notes=['# 外部基线与内部机制对照汇总','',f'当前VB全部方法使用相同{expected}个原始标签。神经网络先在case内平均5个seed的误差，再对case等权；确定性方法不伪造seed重复。结果均为探索性，未根据测试重新挑参数或epoch。','',markdown(cw.reset_index()),'','![当前VB](current_mae.png)','','## 未来VB：所有方法共同有效目标','',markdown(fw.reset_index()),'',markdown(counts),'','共同目标按fold/seed/case/origin_run/target_run/horizon精确交集，仅纳入有限预测。目标VB和真实dt逐键一致性已核查；NaN表示未覆盖而非零误差。future_coverage.csv报告各方法每seed可用数；future_summary.csv同时保留all_available和all_methods_common，不能拿不同覆盖率的all_available直接排名。','KDE方法Kim_TF_cubic_mean_NASA的未来VB来自10成员单调三次拟合曲线的均值，是原RUL方法的VB任务扩展；KDE众数本身用于阈值到达时间，原RUL诊断单独保存于kde/reports，不与mm误差混合。曲线用全case原始run/cut坐标，真实dt仅对齐元数据；采样点单调不保证外推单调。','KDE使用六通道原始波形、每cut75窗口、10成员及历史预测平滑；sensor12使用12维电流统计特征。原始输入、历史信息、模型容量、ensemble及训练计算量不同，不能把差异归因于单一架构或声称等资源比较。','', '![未来VB](future_mae.png)','','## 内部冷启动机制消融（独立协议）','',markdown(pd.DataFrame(it)),'','同随机monitor/rate初始化、同100轮；current-only未来为persistence。此表保留其全部有标签起点，不沿用上方Kim/LSTM覆盖交集，不能跨表混比。integral-only改善未来VB但当前VB没有提升；both也未超过integral-only。固定lambda=1不是等梯度强度，固定sensor偏导不是轨迹全导，不能推广为全部局部约束无效。旧ABCD有预训练及额外适配预算，只能回答适配收益。','','## 复现与评价边界','','PI-GPR采用单侧因果平滑和训练内参数筛选；Zhang为NASA输入适配；PIDGGCN是因果feature图适配，印刷导数与数学修正版单列。LSTM仅复现磨损序列模块，使用共同sensor monitor历史估计，不是原完整系统。linear2是真实时间的最近两点外推。以上方法边界见各自文档。','paired_case_bootstrap.csv包含预设sensor vs Kim当前、ODE vs Kim未来等case配对区间；五seed先case内平均，再在fold内重采样case。小case数量和反复观察测试限制推断，区间仅作探索性描述。']

 pd.DataFrame([{'method':k,'中文名称':v} for k,v in NAMES.items()]).to_csv(OUT/'method_labels.csv',index=False)

 budget=pd.DataFrame([

 {'方法':'CT100','输入':'工况＋时间','参数':'2497','实际预算':'100轮，5seed/折；历史checkpoint重放'},

 {'方法':'sensor12_100','输入':'工况＋时间＋12维电流','参数':'3265','实际预算':'100轮，5seed/折；历史checkpoint重放'},

 {'方法':'A 独立monitor＋ODE','输入':'monitor同上；rate用VB＋工况','参数':'3265＋1281＝4546','实际预算':'monitor100轮＋独立ODE500轮'},

 {'方法':'B/C/D 适配','输入':'与A相同；C/D含未来VB损失','参数':'4546；B/C冻结rate','实际预算':'共同A起点再100轮；B只当前，C冻结ODE，D两支更新'},

 {'方法':'Kim TF ensemble','输入':'6通道raw，75窗口/cut，历史平滑','参数':'428901/成员 × 10＝4289010/ensemble','实际预算':'2fold×5ensemble×10成员×100轮；batch128，4200次更新/成员(A)、5100次(B)'},

 {'方法':'Chen PIDGGCN','输入':'top12＋工况时间＋因果历史图','参数':'46785（completed.json核验）','实际预算':'2版本×24运行×100轮，RMSprop；原5000轮schedule压缩披露'},

 {'方法':'Zhang PINN','输入':'统一top12适配','参数':'见zhang子报告','实际预算':'新增24×100轮'},

 {'方法':'GPR / SVR','输入':'统一17维特征','参数':'非固定神经网络参数数','实际预算':'训练内拟合/内层选参，确定性'},

 {'方法':'Sun PI-GPR','输入':'统一sensor及工况时间＋HI','参数':'物理均值＋精确GP','实际预算':'3375候选×14内层，2外层拟合，确定性'},

 {'方法':'Sun LSTM模块','输入':'共同monitor历史估计VB','参数':'见sun_lstm子报告','实际预算':'10×100轮，非原完整ResNet系统'}])

 budget.to_csv(OUT/'input_parameter_budget.csv',index=False)

 notes[2:2]=['本轮当前VB最低为普通GPR 0.09373 mm，电流MLP为0.09580，不能宣称我们的monitor全面最佳。Kim raw集成当前为0.13959；sensor相对Kim低0.04380 mm，14case中11个更低，探索性配对95%区间[-0.07139,-0.01900]。','共同未来目标上，独立ODE h1/h2/h3为0.09557/0.10184/0.10640，Kim三次扩展为0.18412/0.35929/0.92118。联合适配D的h3最低0.08129，但h1=0.09730没有超过同预算B的0.09459；因此收益不是所有horizon都成立。','']

 abcd=summarize(old).pivot(index='method',columns='subset',values='mae_mm')[['overall','fold_a','fold_b','fold_a_in','fold_a_out']]

 notes+=['','## 既有ABCD：同138标签的当前VB','',markdown(abcd.reset_index()),'','ABCD当前均直接monitor输出、同138标签，未来仍使用上方共同覆盖。D联合适配当前MAE0.09532，高于同预算B仅继续当前监督的0.09267；C冻结ODE适配为0.09613。不能只展示D的h3改善而隐藏当前退化。B数值略低于GPR0.09373，但B从预训练父模型额外100轮，预算不同，不能当等预算优胜证据。']

 notes+=['','## 输入、参数与真实训练预算','',markdown(budget),'','ABCD预算来源已核对extract_ours.py及references的budget字段；它们取自current12_abcd_from100_20261019，均为单次预测，不是128点融合递推。神经网络参数数通过相同配置实例计数：CT2497、sensor3265、rate1281；PID由completed.json记录。Kim每epoch含多个minibatch，100epoch不等于100次更新，不能与小MLP full-batch100轮当等计算量。','Zhang新增24×100轮约57.15秒；LSTM10×100轮约19.51秒；PI-GPR网格与最终拟合约1.49秒；标准GPR/SVR约1.05秒；PID两版本48×100轮约2小时21分钟；内部96×100轮约9.4分钟。Kim累计训练/评估/保存150539.68秒（41.82小时），文件墙钟跨度42.21小时；全部逐epoch报告重建821.56秒（13.69分钟），未重新训练。上述计时范围不包括文献与实现，不能视为同硬件统一benchmark。','','## 典型case与不确定性','Kim三次曲线远期误差受外推不稳定影响：在共同h3集合，case14的MAE约4.889 mm、case5约2.169 mm、case10约0.919 mm；不要仅以总体均值叙述，应结合逐case轨迹和覆盖。它们来自预测历史拟合而非真实VB重置，也没有全外推区间单调保证。参见../kde/reports/case14_cubic_diagnostic.png及case14_member_diagnostics.json的成员级诊断。','ODE对Kim共同h1/h2/h3的case配对差值分别为-0.08855、-0.25745、-0.81478 mm；区间分别[-0.14038,-0.04165]、[-0.47764,-0.11103]、[-1.56310,-0.27216]。h3仅12case，不能把跨h总体变化纯归因于预测步数。','D对同预算B在h3差-0.02900，区间[-0.05211,-0.00727]；h1/h2区间跨0。更多区间及ODE/Kim单独pairwise覆盖对照见paired_case_bootstrap.csv；共同全部方法与两方法交集不可混用。','current_case_trajectories.png已更新14case真实VB及CT/sensor/GPR/Kim的seed均值轨迹，case13后期Kim下降而真实VB加速增长、case5明显高估，需与case14曲线外推诊断一起分析；图中均值曲线不等于先算每seed误差的主表。current_mae.png和future_mae.png包含全部方法且已视觉检查。未来case_seed_metrics含all_available/common以及Ain/out。','最终epoch100汇总及Kim全部1000组逐epoch报告已完成；1000个完成标记、3000份CSV及与原始成员预测的对应关系见kde/reports/all_epochs_audit.json。']

 notes.insert(0, '> **2026-11-10 口径更正：本页是原始monitor与单次传播的组件比较，不是完整融合方法的主结果。** 原ABCD的sensor观测融合＋单调截断结果仍正确，当前为128点、未来为114/100/86对每seed。完整方法原表和逐键核查见[结果口径更正](../reconciliation_20261110/RESULTS.md)。不能将本页138点或外部共同目标结果替代原融合表。\n\n')

 (OUT/'RESULTS.md').write_text('\n'.join(notes),encoding='utf8');(OUT/'audit.json').write_text(json.dumps({'current_checks':checks,'current_parent_replay':True,'future_keys_include_seed':True,'future_target_dt_identical':True,'future_nan_excluded_from_common':True,'finite_missing_coverage_identical_across_seeds':True,'common_counts':common_details,'missing_methods':[]},indent=2))

 print(cw.to_string());print(fw.to_string());print(counts.to_string(index=False))

if __name__=='__main__':main()









