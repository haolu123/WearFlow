"""Figures, scope-specific summaries and Chinese interpretation; no training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
def read(n):return pd.read_csv(P/n)
def save(fig,name):
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(P/(name+'.'+ext),dpi=220,bbox_inches='tight')
    plt.close(fig)
def md(d):
    fmt=lambda v:f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(map(str,d.columns))+' |','| '+' | '.join(['---']*len(d.columns))+' |']+['| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])
o=read('oracle_pair_predictions.csv.gz');s=read('E1_oracle_summary.csv');b=read('E1_case_bootstrap.csv')
e=read('E2_summary.csv');r=read('rate_curves.csv.gz');obs=read('observed_interval_rates.csv');t=read('E5A_summary.csv')
colors={'warm_A':'#777777','warm_D':'#007f86','cold_local_only':'#c27b28','cold_integral_only':'#3659b8','cold_both':'#a54c83'}
labels={'warm_A':'A: pretrained rate','warm_D':'D: jointly adapted rate','cold_local_only':'Cold: local','cold_integral_only':'Cold: integral','cold_both':'Cold: both'}
main=s[(s.scope=='original_fusion')&(s.region=='overall')]
# E1: seed-average scatter and case-macro errors; paired cases remain the inference unit.
fig,axs=plt.subplots(1,3,figsize=(15,4))
for model in ['warm_A','warm_D']:
    q=o[(o.model==model)&o.original_fusion].groupby(['fold','case','origin_run','target_run','horizon'])[['true_increment','pred_increment']].mean()
    axs[0].scatter(q.true_increment,q.pred_increment,s=14,alpha=.4,label=labels[model],color=colors[model])
lim=max(axs[0].get_xlim()[1],axs[0].get_ylim()[1]);axs[0].plot([0,lim],[0,lim],'k--',lw=1)
axs[0].set(xlabel='Measured interval increment (mm)',ylabel='Oracle predicted increment (mm)',title='(a) Measured initial VB');axs[0].legend(fontsize=8)
for model in colors:
    q=main[main.model==model];axs[1].plot(q.horizon,q.abs_error,'o-',color=colors[model],label=labels[model])
axs[1].set(xlabel='Subsequent labeled record',ylabel='Case-macro increment MAE (mm)',xticks=[1,2,3],title='(b) Same diagnostic targets');axs[1].legend(fontsize=7)
tim=[]
for (model,fold,c,seed,bin),q in o[o.original_fusion].groupby(['model','fold','case','seed','dt_bin']):
    tim.append(dict(model=model,fold=fold,case=c,seed=seed,dt_bin=bin,mae=q.abs_error.mean(),bias=q.error.mean(),pairs=len(q),dt=q.dt.mean()))
tc=pd.DataFrame(tim);tc.to_csv(P/'E1_time_case_seed.csv',index=False)
ts=tc.groupby(['model','fold','case','dt_bin'])[['mae','bias','dt']].mean().groupby(['model','dt_bin']).mean().reset_index();ts.to_csv(P/'E1_time_summary.csv',index=False)
for model in ['warm_A','warm_D','cold_local_only','cold_integral_only']:
    q=ts[ts.model==model].set_index('dt_bin').reindex(['short','medium','long']);axs[2].plot(range(3),q.mae,'o-',color=colors[model],label=labels[model])
axs[2].set(xticks=range(3),xticklabels=['dt <= 6','6 < dt <= 12','dt > 12'],ylabel='Case-macro increment MAE (mm)',title='(c) Training-defined time bins');axs[2].tick_params(axis='x',labelsize=8)
save(fig,'E1_integrated_increment')
# E2 MSE components, additive with identical weights (cross term may be negative).
fig,axs=plt.subplots(1,3,figsize=(14,4))
de=e[(e.scope=='original_fusion')&(e.region=='overall')]
for h,ax in enumerate(axs,1):
    q=de[de.horizon==h].set_index('method').loc[['fusion_A','fusion_B','fusion_C','fusion_D']]
    x=np.arange(4)
    for offset,col,label,color in [(-.27,'initial_mse','Initial','#728ab3'),(-.09,'increment_mse','Increment','#e49a49'),(.09,'cross_term','Cross term','#a65a8a'),(.27,'future_mse','Future total','#187e78')]:
        ax.bar(x+offset,q[col],width=.18,label=label,color=color)
    ax.axhline(0,color='k',lw=.6);ax.set(xticks=x,xticklabels=['A','B','C','D'],ylabel='Case-macro mean square (mm²)',title=f'h{h}: exact additive decomposition')
axs[0].legend(fontsize=8);save(fig,'E2_error_decomposition')
fig,axs=plt.subplots(1,3,figsize=(13,4))
ep=read('E2_error_decomposition_pairs.csv.gz')
for h,ax in enumerate(axs,1):
    for method,color in [('fusion_A','#777777'),('fusion_D','#007f86')]:
        q=ep[(ep.method==method)&(ep.horizon==h)].groupby(['fold','case','origin_run','target_run'])[['initial_error','increment_error']].mean()
        ax.scatter(q.initial_error,q.increment_error,s=15,alpha=.5,label=method[-1],color=color)
    ax.axhline(0,color='k',lw=.5);ax.axvline(0,color='k',lw=.5)
    ax.set(title=f'h{h}: seed-mean pair errors',xlabel='Initial error (mm)',ylabel='Increment error (mm)');ax.legend()
save(fig,'E2_error_association')
# E8 plot each actual condition separately within per-fold training VB bounds.
def rateplot(models,name):
    conditions=list(r[['fold','condition']].drop_duplicates().itertuples(index=False,name=None))
    fig,axs=plt.subplots(int(np.ceil(len(conditions)/4)),4,figsize=(16,3.2*np.ceil(len(conditions)/4)),squeeze=False)
    for ax,(fold,cond) in zip(axs.flat,conditions):
        for model in models:
            q=r[(r.fold==fold)&(r.condition==cond)&(r.model==model)]
            z=q.groupby('vb').rate.agg(['mean','std'])
            ax.plot(z.index,z['mean'],color=colors[model],label=labels[model]);ax.fill_between(z.index,z['mean']-z['std'],z['mean']+z['std'],alpha=.12,color=colors[model])
        q=obs[(obs.fold==fold)&(obs.condition==cond)&(obs.horizon==1)].copy()
        q['midpoint']=(q.target_mm+q.origin_target_mm)/2
        ax.scatter(q.midpoint,q.observed_average_rate,c='black',marker='x',s=22,label='Test interval average (not instantaneous)')
        ax.set(title=f'{fold}\n(depth, feed, material) = {cond}',xlabel='VB (mm)',ylabel='Rate (mm / dataset time unit)');ax.tick_params(labelsize=8)
    for ax in list(axs.flat)[len(conditions):]:ax.set_visible(False)
    axs.flat[0].legend(fontsize=6)
    save(fig,name)
rateplot(['warm_A','warm_D'],'E8_warm_rate_conditions')
rateplot(['cold_local_only','cold_integral_only','cold_both'],'E8_cold_rate_conditions')
rate_change=r[r.model=='warm_D'].merge(r[r.model=='warm_A'],on=['fold','seed','condition','vb'],suffixes=('_D','_A'))
rate_change['delta_rate']=rate_change.rate_D-rate_change.rate_A
rate_change.groupby(['fold','condition'])[['rate_A','rate_D','delta_rate']].mean().reset_index().to_csv(P/'E8_rate_change_descriptive.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for scope,ax in zip(['original_fusion','external_common'],axs):
    for method,color in zip(['fusion_A','fusion_B','fusion_C','fusion_D'],['#777777','#b88a24','#8d5a95','#007f86']):
        q=t[(t.scope==scope)&(t.region=='overall')&(t.grouping=='dt_bin')&(t.method==method)].set_index('interval_group').reindex(['short','medium','long'])
        ax.plot(range(3),q.future_mae,'o-',label=method[-1],color=color)
    ax.set(xticks=range(3),xticklabels=['dt <= 6','6 < dt <= 12','dt > 12'],ylabel='Future case-macro MAE (mm)',title=scope);ax.legend()
save(fig,'E5A_actual_interval_errors')
warm=main[main.model.isin(['warm_A','warm_D'])].pivot(index='model',columns='horizon',values='abs_error').reset_index()
bs=b[(b.scope=='original_fusion')&(b.method=='warm_D')][['horizon','delta','ci_low','ci_high','cases']]
cold=main[main.model.str.startswith('cold')].pivot(index='model',columns='horizon',values='abs_error').reset_index()
d3=de[de.horizon==3].set_index('method')
delta=d3.loc['fusion_D',['initial_mse','increment_mse','cross_term','future_mse']]-d3.loc['fusion_A',['initial_mse','increment_mse','cross_term','future_mse']]
long=t[(t.scope=='original_fusion')&(t.region=='overall')&(t.grouping=='dt_bin')][['method','interval_group','future_mae','future_error','cases']]
coverage=o[o.model=='warm_A'].groupby(['original_fusion','common','horizon','seed']).size().reset_index(name='pairs')
coverage.to_csv(P/'pair_pool_coverage.csv',index=False)
audit=json.loads((P/'audit.json').read_text())
text='''# E1／E2／E8／E5A 第一批补充结果

本批仅重载既有checkpoint进行评估，没有训练、选超参数、改变主模型或覆盖原稿。所有速率单位为mm/数据集加工时间单位；250Hz不用于推断加工时间单位。时间边界在查看分组性能之前由各折训练配对确定。

## 1. 结论

**积分监督对区间磨损的帮助得到进一步支持，但联合微调相对已预训练ODE的改善仍存在case层面的不确定性。** 冷启动integral相对local的oracle增量误差明显下降；warm D相对A在三个horizon均数值改善，但95% case配对区间包含零。不能把它写成确定恢复真实磨损率。

## 2. E1：实测初值传播与磨损增量

下表主池沿用原完整融合的114/100/86对每seed，h1/h2有14case、h3有13case。它和外部共同目标100/86/73池不同；两个池以及全有标签起点池均单列在E1_oracle_summary.csv。所有模型诊断时都以实测VB作为初值，绝非部署成绩。

''' + md(warm) + '\n\nD−A差值及95% case配对bootstrap区间（负数为D更好）：\n\n'+md(bs)+'''

A、B、C的rate参数相同，oracle增量预测完全等价，故表中只展示A。C的monitor变化可以改变部署初值，但不改变这一实测初值诊断。

冷启动模型在同一诊断目标池上的结果如下。它们只在冷启动组内解释，不与有500轮ODE预训练的warm模型作等预算排名。

'''+md(cold)+'''

真实初值下，增量误差与oracle末状态误差数学上完全相同；这是一次隔离monitor误差的诊断，不是两份独立证据。增量不除以很小的真实增量计算百分比，避免零增量导致失真。

![实测初值增量](E1_integrated_increment.png)

## 3. E2：误差抵消能否解释改善

逐条核验 e_future=e_initial+e_increment；平方误差为initial MSE＋increment MSE＋2E[e_initial e_increment]，使用相同case等权权重。MAE不作加法分解。

原完整融合h3中D−A的MSE变化为：

'''+md(delta.rename('D_minus_A_mm2').reset_index().rename(columns={'index':'component'}))+'''

这表明增量平方误差下降是D相对A远期改善的主要来源，交叉项变化也贡献一部分；不是仅凭更强误差抵消得到低未来误差。注意“增量误差”仍是在实际预测初值上计算，状态依赖的rate会受到初值影响，因此不能将其下降完全因果归因于rate参数。

E2_summary.csv保留三个horizon、总体/两fold/A范围内外、两种覆盖的有符号bias、MAE、MSE、交叉项与异号比例。E2_within_case_correlations.csv记录逐case/seed/horizon相关系数；不足3对或零方差不硬算相关。配对/seed均不当独立case。

![误差分解](E2_error_decomposition.png)

![初值与增量误差关联](E2_error_association.png)

## 4. E8：工况相关rate

每个面板仅包含一种实际工况和一个fold。曲线只画该fold训练集中这一工况所覆盖的真实VB最小—最大范围，不把不同工况连成时变过程。条件元组依原档案为depth、feed、material代码。曲线阴影为五seed标准差，不是置信区间。

黑叉是对应测试case相邻有标签记录的区间平均磨损率，以区间VB中点定位，**不是瞬时rate真值**。它与曲线仅作行为对照，严格的区间拟合准确性应使用E1的实际积分。训练支持区间以外的黑叉只展示观测，不延伸模型曲线假装有训练覆盖。不同训练支持范围上的均值不能用来断言工况因果作用。全数据中少量实测VB相邻值下降，产生负区间平均rate，这些原始观测保留、不裁剪；非负rate模型不能拟合其负增量。此前Figure3只展示的五个case子集没有负实测增量，两者不矛盾。

![Warm工况曲线](E8_warm_rate_conditions.png)

![Cold工况曲线](E8_cold_rate_conditions.png)

观察到的变化：在每个fold、每种工况各自训练支持VB网格上，material代码2的所有工况中，D的平均rate均低于A；material代码1的变化方向并不一致。这与warm D减少总体正向增量bias的结果相容，但不是材料作用的因果证明。部分相同工况在两个fold中呈现不同曲线形状（例如一折先降后升、另一折持续上升），说明rate函数仍有跨case识别不稳定性。不能只展示某条看起来符合预期的曲线来宣称恢复了真实磨损机理。逐工况变化数值见E8_rate_change_descriptive.csv。

## 5. E5A：实际时间间隔

两折的训练h1–h3配对时间范围均为[2,24]，三分位边界均为6和12。短段dt<=6、中段6<dt<=12、长段dt>12。区间分组混合horizon，所以同时保留每条记录的horizon；不能把长时间误差的差异解释为纯时间因果效应。各组case和配对覆盖随分组变化。

'''+md(long)+'''

![实际时间分层](E5A_actual_interval_errors.png)

误差并非随时间组严格单调增长：中间时间组可以比最长时间组更难。这与不同时间组包含不同case、VB范围和horizon有关，不应把它解读为长时间预测更容易或模型拥有时间外推优势。

本测试配对全部在训练[2,24]范围内（范围外0条），所以这只是不同跨度的诊断，**没有证明未见长间隔外推**，也不意味着每个具体dt值均曾训练。E5A_horizon_bin_summary.csv进一步同时按horizon与时间组细分，保留每组覆盖差异。未执行E5B或任何新训练。

## 6. 核查、文件与使用建议

原Figure3的实测增量经再次核对确为target VB−origin真实VB，导出数据无负实测增量，不存在意见中猜测的“减去预测初值”错误。它与本次oracle图回答不同问题，不应互相替代或混淆。

checkpoint_replay_audit.csv核查warm A/D共20个模型从原融合起点重新传播，与已保存未来预测差异<4e-6 mm。A/C rate参数逐元素一致；scaler与时间尺度匹配。audit.json记录源文件和checkpoint SHA256。原稿和已有结果保持不变。

建议将E1图、E2图用于正文核心证据，工况曲线按版面选择正文/补充材料。后续E3权重对照仍有意义，因为现有local/integral结果仍对应固定权重1；本批不代替该公平性实验。

重建：先运行analyze.py，再运行report.py。所有逐配对结果、case/seed指标、bootstrap和图的PNG/PDF/SVG均保存在本目录。
'''
text+=f'\n本次checkpoint推理、数据导出与核心数值汇总用时{audit["seconds"]:.2f}秒（不含Python导入、绘图和人工审阅）；没有训练耗时。\n'
(P/'RESULTS.md').write_text(text,encoding='utf-8')
print(main[['model','horizon','abs_error']].to_string(index=False))
print('Outside-time rows:',len(read('E5A_deployment_pairs.csv.gz').query("dt_support == 'outside_train'")))
