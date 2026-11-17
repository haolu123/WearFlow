"""Post-hoc descriptive analysis of frozen predictions. No fitting or selection."""
from pathlib import Path
import json
import hashlib
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'tool_analysis'
COLORS={'local':'#C66B29','integral':'#157A9B'}

def export(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=220,bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.svg',bbox_inches='tight')
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    raw=pd.read_csv(ROOT/'reports/test_predictions_epoch100.csv.gz')
    targets=pd.read_csv(ROOT/'causal_targets.csv').reset_index(names='record_index')
    metrics=pd.read_csv(ROOT/'reports/tool_metrics_seed_mean.csv')
    # Preserve mean-of-seed errors, rather than scoring an ensemble mean prediction.
    selected=metrics[metrics.method.isin(['local','integral'])]
    pair=selected.pivot(index=['task','horizon','tool'],columns='method',values='mae').reset_index()
    pair['difference_integral_minus_local']=pair.integral-pair.local
    pair['improvement_mm']=pair.local-pair.integral
    pair['improvement_percent']=100*pair.improvement_mm/pair.local
    pair.to_csv(OUT/'paired_tool_mae.csv',index=False)
    overview=[];rng=np.random.default_rng(20261111)
    for (task,h),g in pair.groupby(['task','horizon']):
        delta=g.difference_integral_minus_local.to_numpy()
        boot=rng.choice(delta,(20000,len(delta)),replace=True).mean(axis=1)
        positive=g.improvement_mm.clip(lower=0)
        leave=[np.delete(delta,i).mean() for i in range(len(delta))]
        overview.append({'task':task,'horizon':h,'local_mae':g.local.mean(),'integral_mae':g.integral.mean(),'difference':delta.mean(),'improved_tools':int((delta<0).sum()),'worse_tools':int((delta>0).sum()),'lower95':np.quantile(boot,.025),'upper95':np.quantile(boot,.975),'leave_one_tool_out_min_difference':min(leave),'leave_one_tool_out_max_difference':max(leave),'largest_positive_gain_tool':int(g.loc[g.improvement_mm.idxmax(),'tool']),'largest_share_of_positive_gains':positive.max()/positive.sum() if positive.sum()>0 else np.nan})
    overview=pd.DataFrame(overview);overview.to_csv(OUT/'improvement_breadth.csv',index=False)
    # Join each future prediction to its own monitor initial state and oracle forecast.
    key=['method','seed','tool','origin','target','horizon']
    future=raw[(raw.method.isin(['local','integral']))&(raw.task=='future_mm')].copy()
    oracle=raw[(raw.method.isin(['local','integral']))&(raw.task=='oracle_mm')][key+['prediction']].rename(columns={'prediction':'oracle_prediction'})
    future=future.merge(oracle,on=key,validate='one_to_one')
    current=raw[(raw.method.isin(['local','integral']))&(raw.task=='current')][['method','seed','tool','filename','prediction']]
    current=current.merge(targets[['record_index','filename']],on='filename',validate='many_to_one').rename(columns={'record_index':'origin','prediction':'initial_prediction'})
    future=future.merge(current.drop(columns='filename'),on=['method','seed','tool','origin'],validate='many_to_one')
    future['initial_error']=future.initial_prediction-future.origin_VB_mm
    future['future_error']=future.prediction-future.VB_mm
    future['true_increment']=future.VB_mm-future.origin_VB_mm
    future['predicted_increment']=future.prediction-future.initial_prediction
    future['increment_error']=future.predicted_increment-future.true_increment
    future['oracle_error']=future.oracle_prediction-future.VB_mm
    future['propagated_initial_effect']=future.prediction-future.oracle_prediction
    future['oracle_increment']=future.oracle_prediction-future.origin_VB_mm
    for a,b in [('initial_error','increment_error'),('oracle_error','propagated_initial_effect')]:
        assert np.max(np.abs(future[a]+future[b]-future.future_error))<1e-12
    future['cancellation_mm']=abs(future.oracle_error)+abs(future.propagated_initial_effect)-abs(future.future_error)
    future['opposite_sign']=(future.oracle_error*future.propagated_initial_effect<0)
    future['estimated_better_than_oracle']=(abs(future.future_error)<abs(future.oracle_error))
    future['origin_seconds']=targets.cumulated_tool_contact_time.to_numpy()[future.origin.to_numpy(int)]
    future['target_seconds']=targets.cumulated_tool_contact_time.to_numpy()[future.target.to_numpy(int)]
    future[['method','seed','tool','origin','target','horizon','actual_seconds','origin_seconds','target_seconds','VB_mm','origin_VB_mm','initial_prediction','prediction','oracle_prediction','initial_error','increment_error','future_error','oracle_error','propagated_initial_effect','cancellation_mm','opposite_sign','estimated_better_than_oracle','true_increment','predicted_increment','oracle_increment']].to_csv(OUT/'pair_error_decomposition.csv.gz',index=False)
    summaries=[]
    for (method,tool,seed,h),g in future.groupby(['method','tool','seed','horizon']):
        item={'method':method,'tool':tool,'seed':seed,'horizon':h,'pairs':len(g)}
        for c in ['initial_error','increment_error','future_error','oracle_error','propagated_initial_effect']:
            item[c+'_mae']=abs(g[c]).mean();item[c+'_bias']=g[c].mean()
        item.update(cancellation_mm=g.cancellation_mm.mean(),opposite_sign_fraction=g.opposite_sign.mean(),estimated_better_than_oracle_fraction=g.estimated_better_than_oracle.mean())
        item['cancellation_fraction']=item['cancellation_mm']/(item['oracle_error_mae']+item['propagated_initial_effect_mae'])
        summaries.append(item)
    seed=pd.DataFrame(summaries);seed.to_csv(OUT/'decomposition_per_tool_seed.csv',index=False)
    tool=seed.groupby(['method','tool','horizon'],as_index=False).mean(numeric_only=True).drop(columns='seed')
    tool.to_csv(OUT/'decomposition_per_tool.csv',index=False)
    macro=tool.groupby(['method','horizon'],as_index=False).mean(numeric_only=True).drop(columns='tool')
    macro.to_csv(OUT/'decomposition_macro.csv',index=False)
    # Recompute final MAEs from the joined raw data, independent of report aggregation.
    for r in tool.itertuples():
        old=selected[(selected.method==r.method)&(selected.tool==r.tool)&(selected.horizon==r.horizon)]
        assert abs(old[old.task=='future_mm'].mae.iloc[0]-r.future_error_mae)<1e-12
        assert abs(old[old.task=='oracle_mm'].mae.iloc[0]-r.oracle_error_mae)<1e-12
    main_table=metrics[metrics.task.isin(['current','future_mm','linear_mm'])].groupby(['method','task','horizon'],as_index=False).agg(mae_mm=('mae','mean'),tools=('tool','size'),targets_per_seed=('n','sum'))
    main_table.to_csv(OUT/'table_main.csv',index=False)
    coverage=pd.read_csv(ROOT/'forecast_coverage.csv');coverage.to_csv(OUT/'table_coverage.csv',index=False)
    budgets=[]
    for method in ['time','current','local','integral']:
        files=list((ROOT/'formal').glob(f'*/{method}_seed*/complete.json'))
        first=json.loads(files[0].with_name('identity.json').read_text())
        budgets.append({'method':method,'fits':len(files),'final_fits':45,'inner_fits':9,'epochs_per_fit':100,'effective_parameters':first['effective_parameters'],'training_inputs':'time' if method=='time' else '12 train-ranked force features + time','seconds':sum(json.loads(f.read_text())['seconds'] for f in files),'training_records':'all training tools, tool-equal losses'})
    files=list((ROOT/'formal').glob('*/gpr/complete.json'))
    budgets.append({'method':'gpr','fits':len(files),'final_fits':9,'inner_fits':9,'epochs_per_fit':np.nan,'effective_parameters':np.nan,'training_inputs':'same 12 features + time','seconds':sum(json.loads(f.read_text())['seconds'] for f in files),'training_records':'at most 128 per tool; exact GPR kernel marginal likelihood optimization'})
    pd.DataFrame(budgets).to_csv(OUT/'table_budget.csv',index=False)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    panels=[('current',0)]+[('future_mm',h) for h in [240,480,720]]+[('oracle_mm',h) for h in [240,480,720]]
    fig,axes=plt.subplots(2,4,figsize=(13.5,6.4))
    for ax,(task,h) in zip(axes.flat,panels):
        g=pair[(pair.task==task)&(pair.horizon==h)]
        for method in ['local','integral']:ax.plot(g.tool,g[method]*1000,'o-',color=COLORS[method],label=method.capitalize(),markersize=4)
        title='Current VB' if task=='current' else ('Forecast' if task=='future_mm' else 'Label-initial diagnostic')+f': {h} s'
        ax.set(title=title,xlabel='Held-out tool',ylabel='MAE (µm)',xticks=range(1,10));ax.grid(alpha=.2)
    axes.flat[0].legend(frameon=False)
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.9,'Each point: mean of 5 seed MAEs\nEach tool has equal statistical weight\n\n100 epochs, cold start\nPublished labels include interpolation\nNo test-based model selection',va='top')
    fig.tight_layout();export(fig,'figure1_tool_comparison')
    # Figures use seed-mean predictions only for visualization; not the scored estimator.
    curve_rows=[];fig,axes=plt.subplots(3,3,figsize=(12,9))
    for col,t in enumerate([1,5,8]):
        target=targets[targets.tool==t]
        axes[0,col].plot(target.cumulated_tool_contact_time,target.VB_mm*1000,'k-',label='Published VB',lw=1.8)
        axes[1,col].plot(target.cumulated_tool_contact_time,target.VB_mm*1000,'k-',label='Published VB',lw=1.8)
        for method in ['local','integral']:
            g=raw[(raw.tool==t)&(raw.method==method)&(raw.task=='current')].groupby('cumulated_tool_contact_time',as_index=False).prediction.mean()
            axes[0,col].plot(g.cumulated_tool_contact_time,g.prediction*1000,color=COLORS[method],label=method.capitalize(),lw=.9)
            for r in g.itertuples():curve_rows.append({'tool':t,'method':method,'panel':'current','seconds':r.cumulated_tool_contact_time,'prediction_mm':r.prediction})
            g=future[(future.tool==t)&(future.method==method)&(future.horizon==720)].groupby(['origin','target'],as_index=False).mean(numeric_only=True)
            axes[1,col].plot(g.target_seconds,g.prediction*1000,color=COLORS[method],lw=.9)
            axes[2,col].plot(g.origin_seconds,g.oracle_increment*1000,color=COLORS[method],lw=1,label=method.capitalize())
            for r in g.itertuples():curve_rows.append({'tool':t,'method':method,'panel':'future720_and_oracle_increment','seconds':r.target_seconds,'origin_seconds':r.origin_seconds,'prediction_mm':r.prediction,'oracle_increment_mm':r.oracle_increment,'published_increment_mm':r.true_increment})
            if method=='local':axes[2,col].plot(g.origin_seconds,g.true_increment*1000,'k-',lw=1.8,label='Published increment')
        axes[0,col].set_title(f'Predeclared tool T{t}')
        axes[0,col].set(ylabel='Current VB (µm)',xlabel='State time (s)')
        axes[1,col].set(ylabel='Future VB, 720 s (µm)',xlabel='Target time (s)')
        axes[2,col].set(ylabel='Label-initial increment (µm)',xlabel='Origin time (s)')
        for ax in axes[:,col]:ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);axes[2,0].legend(fontsize=8)
    fig.tight_layout();export(fig,'figure2_predeclared_trajectories')
    pd.DataFrame(curve_rows).to_csv(OUT/'figure2_curve_data.csv',index=False)
    focal=tool[(tool.tool.isin([1,5,8]))&(tool.horizon==720)]
    focal.to_csv(OUT/'table_focal_error_sources.csv',index=False)
    report=['# Stiehl：逐刀具、误差来源与论文图表','', '本分析只复用固定 100 epoch 的预测，不重新训练、不据测试扩充配置。MAE 先在每个 seed/刀具中计算，再平均五 seed 和九刀；不能把平均预测轨迹的 MAE 当作本报告 MAE。','', '## 1. 优势覆盖范围','', '| 任务 | 秒 | Local MAE | Integral MAE | 改善刀具 | 差异95%区间 | 去掉任一刀后平均差异范围 |','|---|---:|---:|---:|---:|---|---|']
    for r in overview.itertuples():report.append(f'| {r.task} | {r.horizon} | {r.local_mae:.6f} | {r.integral_mae:.6f} | {r.improved_tools}/9 | [{r.lower95:+.6f}, {r.upper95:+.6f}] | [{r.leave_one_tool_out_min_difference:+.6f}, {r.leave_one_tool_out_max_difference:+.6f}] |')
    report+=['','差异定义为 Integral − Local，负值有利于 Integral。留一刀敏感性仅重新汇总已有九个测试结果，不是重新训练，也不是额外泛化检验。bootstrap 以刀具为单位，未将 seed 或密集插值 run 当成独立样本；机床/批次相关性与九刀小样本限制仍在。区间为逐项描述性区间，未校正多重比较。','', '## 2. 可核验的误差分解','', '令 y_i 为发布起点标签，m_i 为 monitor 输出，Φ 为同一训练模型的 ODE：', '', '- e_initial = m_i − y_i；e_increment = [Φ(m_i) − m_i] − [y_j − y_i]；e_future = e_initial + e_increment。', '- e_oracle = Φ(y_i) − y_j；a_initial = Φ(m_i) − Φ(y_i)；**e_future = e_oracle + a_initial**。', '- a_initial 是初值误差经过非线性传播后的影响，不应直接替换成 e_initial。e_oracle 也不是无噪声真实速率误差，而是发布标签初值下的终点误差。', '- cancellation = |e_oracle| + |a_initial| − |e_future|，单位 mm；两项异号时发生抵消。它是误差代数分解，不是两个机制独立可加的因果贡献。', '', '对全部配对核验两条恒等式误差 < 1e-12，并独立重算 MAE 与原报告一致。初值误差统计使用各 horizon 的相同 origin 池，不能直接拿全体当前 VB MAE 替代。','', '### T1、T5、T8：720 秒（mm）','', '| 刀具 | 方法 | 起点MAE | 标签初值传播MAE | 初值传播影响MAE | 实际未来MAE | 抵消量 | 两项异号比例 |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in focal.itertuples():report.append(f'| T{r.tool} | {r.method} | {r.initial_error_mae:.6f} | {r.oracle_error_mae:.6f} | {r.propagated_initial_effect_mae:.6f} | {r.future_error_mae:.6f} | {r.cancellation_mm:.6f} | {100*r.opposite_sign_fraction:.1f}% |')
    report+=['','### 总体误差分解（mm）','','| 方法 | 秒 | 起点MAE | 标签初值MAE | 初值传播影响MAE | 实际未来MAE | 抵消量 | 异号比例 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in macro.itertuples():report.append(f'| {r.method} | {r.horizon} | {r.initial_error_mae:.6f} | {r.oracle_error_mae:.6f} | {r.propagated_initial_effect_mae:.6f} | {r.future_error_mae:.6f} | {r.cancellation_mm:.6f} | {100*r.opposite_sign_fraction:.1f}% |')
    report+=['','## 3. 正式图表与源数据','','- 主表：table_main.csv。current/future_mm 为保持当前值；current/linear_mm 为过去两次预测的线性外推；time/future_mm 为目标时间查询。GPR 只比较当前监测。','- Figure 1：figure1_tool_comparison（PDF/SVG/PNG），七个任务的逐刀配对 MAE。源数据 paired_tool_mae.csv。','- Figure 2：figure2_predeclared_trajectories（PDF/SVG/PNG），预先指定 T1/T5/T8 的当前轨迹、720 秒未来轨迹及标签初值积分增量。曲线为五 seed 预测均值，仅作可视化，不作为新的 ensemble 性能。当前/未来横轴分别为状态/目标时间，增量横轴为起点时间，发布标签连线不代表各点独立实测。','- 误差来源表：table_focal_error_sources.csv；完整逐刀/seed/跨度结果与原始分解均保留。','- 覆盖表：table_coverage.csv；预算表：table_budget.csv。每个 seed 当前 6409 点；三个未来跨度分别 6007/5626/5256 对，方法配对池一致。GPR 每刀最多128训练样本，其预算不同于全量 MLP；100 epoch 也不意味着相同参数量或算力。','', '两张图加主表和误差来源表即可支撑正文；完整九刀、覆盖、预算与种子细节适合补充材料。所有 VB 单位表中为 mm、图中为 µm。']
    (OUT/'ANALYSIS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    audit={'raw_source_sha256':hashlib.sha256((ROOT/'reports/test_predictions_epoch100.csv.gz').read_bytes()).hexdigest(),'pairs_with_method_and_seed':len(future),'decomposition_exact':True,'matches_previous_mae':True,'no_training':True,'predeclared_examples':[1,5,8]}
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2))
    print(overview.to_string(index=False));print('\nFOCAL\n',focal.to_string(index=False));print('\nMACRO\n',macro.to_string(index=False))

if __name__=='__main__':main()
