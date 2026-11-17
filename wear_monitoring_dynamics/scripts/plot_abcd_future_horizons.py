"""Separate h1/h2/h3 learning curves and final ABCD comparisons; no training."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .report_current12_fusion import md
ROOT=Path('wear_monitoring_dynamics/outputs/current12_abcd_from100_20261019')

def main():
    manifest=pd.read_csv(ROOT/'study_manifest.csv');parts=[]
    for r in manifest.itertuples():
        q=pd.read_csv(Path(r.run_dir)/'fusion_epoch_predictions.csv.gz');q=q[q.task=='future'].copy();q['ae']=abs(q.predicted_vb_mm-q.target_vb_mm)
        per=q.groupby(['epoch','split','horizon','case']).ae.mean().reset_index().assign(group=r.group,fold=r.fold,seed=r.seed)
        parts.append(per)
        if r.group=='B':
            base=per[per.epoch==0].copy().assign(group='A')
            parts.extend(base.assign(epoch=e) for e in range(101))
    cases=pd.concat(parts,ignore_index=True);cases.to_csv(ROOT/'future_horizon_epoch_cases.csv.gz',index=False)
    summaries=[]
    for subset in ['overall','fold_a','fold_b']:
        q=cases if subset=='overall' else cases[cases.fold==subset]
        perseed=q.groupby(['group','epoch','split','horizon','seed']).ae.mean().reset_index()
        summaries.append(perseed.groupby(['group','epoch','split','horizon']).ae.agg(['mean','std']).reset_index().assign(subset=subset))
    summary=pd.concat(summaries);summary.to_csv(ROOT/'future_horizon_epoch_summary.csv',index=False)
    final=pd.read_csv(ROOT/'summary.csv');final=final[(final.protocol=='fusion')&(final.task=='future')]
    check=summary[(summary.epoch==100)&(summary.split=='test_diagnostic')].merge(final,on=['group','subset','horizon'],validate='one_to_one')
    assert len(check)==36;np.testing.assert_allclose(check['mean'],check.mae,atol=1e-12)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    colors={'A':'#333333','B':'#2474b5','C':'#bd7a15','D':'#a73158'}
    labels={'A':'A 冻结基线','B':'B 当前监督续训','C':'C 冻结ODE适配','D':'D 联合微调'}
    for include_train in [False,True]:
        fig,axs=plt.subplots(3,3,figsize=(15,11),sharex=True)
        for row,h in enumerate([1,2,3]):
            for col,subset in enumerate(['overall','fold_a','fold_b']):
                ax=axs[row,col]
                for group in ['A','B','C','D']:
                    for split in (['test_diagnostic','train'] if include_train else ['test_diagnostic']):
                        q=summary[(summary.group==group)&(summary.subset==subset)&(summary.horizon==h)&(summary.split==split)]
                        style=':' if group=='A' else ('--' if split=='train' else '-')
                        ax.plot(q.epoch,q['mean'],color=colors[group],ls=style,label=labels[group]+(' /训练' if split=='train' else ' /测试'))
                        if not include_train and group!='A':ax.fill_between(q.epoch,q['mean']-q['std'],q['mean']+q['std'],color=colors[group],alpha=.12)
                ax.set(title=f'{subset} / h{h}',xlabel='额外训练epoch',ylabel='Case MAE (mm)');ax.grid(alpha=.2);ax.legend(fontsize=7)
        fig.suptitle('未来VB分horizon比较：同一monitor100＋ODE500起点；融合权重/截断固定\n'+('实线测试、虚线训练；A固定基线' if include_train else '阴影为五seed标准差，不是case置信区间；固定第100轮，不按曲线挑epoch'))
        fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(ROOT/('future_horizon_train_test.png' if include_train else 'future_horizon_test.png'),dpi=150);plt.close(fig)
    table=final[final.subset.isin(['overall','fold_a','fold_b'])].pivot(index=['subset','group'],columns='horizon',values='mae').reset_index();table.columns=['subset','group','h1','h2','h3'];table.to_csv(ROOT/'future_horizon_comparison.csv',index=False)
    overall=table[table.subset=='overall'].set_index('group');changes=[]
    for ref in ['A','B','C']:
        for h in ['h1','h2','h3']:
            changes.append(dict(reference=ref,horizon=h,relative_reduction_pct=100*(overall.loc[ref,h]-overall.loc['D',h])/overall.loc[ref,h]))
    changes=pd.DataFrame(changes);changes.to_csv(ROOT/'future_D_relative_changes.csv',index=False)
    bs=pd.read_csv(ROOT/'case_bootstrap.csv');bs=bs[(bs.protocol=='fusion')&(bs.task=='future')]
    report='''# ABCD未来VB：h1、h2、h3单列

复用已有逐epoch原始预测，无新训练。评价的是当前sensor融合状态作为初值后，ODE积分到未来目标的VB。未来不读取目标时刻的sensor，不做未来观测校正。h1/h2/h3指后续第1/2/3个有标签记录，不是等长时间；使用实际dt。每seed分别114/100/86个配对，h1/h2覆盖14个case、h3覆盖13个case，各horizon对所有方法使用同一目标池。

A冻结，B继续当前监督，C冻结ODE适配monitor，D联合微调；全部同一monitor100+ODE500起点，B/C/D额外100轮。alpha为原先内层选值A折1、B折.25，均单调截断。

## 固定第100轮结果（mm）

'''+md(table)+'''

总体h1最好B，h2/h3最好D。D对A的h2/h3相对降低7.09%/15.49%；D对B的h2/h3降低7.37%/18.38%。D对B的h1反而略差，不能声称所有预测距离都改善。

Fold A：B的h1最低，D的h2/h3最低。D对B的h3从0.15017降至0.10310，改善约31.35%。Fold B：D的h1/h2/h3都最低，但h3对B只从0.10758降至0.10451，改善约2.86%。总体远期收益主要集中Fold A。

C相对B的h3也改善；因此未来监督对monitor初值的适配已经有作用。D对C的进一步改善均未通过本批case配对区间排除0，不能把最低均值解释为已证明需要更新ODE。

## 不确定性

差值=group−reference，负数为改善。每case先平均五seed，再fold内成对bootstrap5000次，不把seed当独立case，未作多重比较校正。本批为已查看测试的探索性结果。D对B的h3区间[-0.04731,-0.00144]不包含0；D对A及D对C的h1/h2/h3区间均包含0。

'''+md(bs[['group','reference','horizon','delta_mae','ci_low','ci_high','cases']])+'''

## 曲线

![测试曲线](future_horizon_test.png)

![训练与测试](future_horizon_train_test.png)

曲线只是诊断，所有主结果固定额外第100轮。不同horizon的目标记录和case数不同，不能把MAE随horizon变化全部解释为时间传播误差累积。图表末轮已与原summary.csv逐项复核（36项均一致）。逐case/seed数据在future_horizon_epoch_cases.csv.gz，绘图数据在future_horizon_epoch_summary.csv。
'''
    (ROOT/'FUTURE_COMPARISON.md').write_text(report,encoding='utf-8')
    for filename in ['README.md','RESULTS.md']:
        p=ROOT/filename;s=p.read_text(encoding='utf-8');note='\n\n未来h1/h2/h3单独比较与曲线见 [FUTURE_COMPARISON.md](FUTURE_COMPARISON.md)，对应图为future_horizon_test.png和future_horizon_train_test.png。复现：python -m wear_monitoring_dynamics.scripts.plot_abcd_future_horizons。\n'
        if 'FUTURE_COMPARISON.md' not in s:p.write_text(s+note,encoding='utf-8')
    print(table.to_string(index=False));print(changes.to_string(index=False))
if __name__=='__main__':main()

