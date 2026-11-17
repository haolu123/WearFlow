"""Descriptive per-case errors, seed variability and leave-one-case-out sensitivity."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run_psd_sweep import ROOT

def md(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    dest=ROOT/'case_diagnostics';dest.mkdir(exist_ok=True)
    selection=pd.read_csv(ROOT/'selected_dimensions.csv')
    p=pd.read_csv(ROOT/'test_predictions.csv').merge(selection[['fold','name']],on=['fold','name'],validate='many_to_one')
    p['error']=p.predicted_vb_mm-p.vb_mm;p['abs_error']=abs(p.error)
    per=p.groupby(['fold','case','group','seed']).agg(mae=('abs_error','mean'),bias=('error','mean')).reset_index()
    per.to_csv(dest/'per_case_seed.csv',index=False)
    mean=per.groupby(['fold','case','group']).agg(mae=('mae','mean'),seed_std=('mae','std'),bias=('bias','mean')).reset_index()
    mean.to_csv(dest/'per_case_summary.csv',index=False)
    wide=mean.pivot(index=['fold','case'],columns='group',values='mae')
    meta=p[p.group=='A'].groupby(['fold','case']).agg(n=('run','nunique'),vb_max=('vb_mm','max'))
    table=meta.join(wide);table.to_csv(dest/'case_mae_table.csv')
    delta=wide[['B','C','D']].sub(wide.A,axis=0);delta.to_csv(dest/'delta_vs_baseline.csv')
    loo=[]
    for (fold,case),row in wide.iterrows():
        for group in ['B','C','D']:
            for scope in ['overall',fold]:
                pool=wide if scope=='overall' else wide.loc[[fold]]
                before=pool[group].mean()-pool.A.mean();remaining=pool.drop(index=(fold,case))
                loo.append(dict(scope=scope,omitted_case=int(case),group=group,original_delta=before,remaining_delta=remaining[group].mean()-remaining.A.mean(),remaining_group_mae=remaining[group].mean(),remaining_A_mae=remaining.A.mean()))
    loo=pd.DataFrame(loo);loo.to_csv(dest/'leave_one_case_out_diagnostic.csv',index=False)
    worst=p.sort_values('abs_error',ascending=False).groupby(['fold','group','seed']).head(5);worst.to_csv(dest/'largest_point_errors.csv',index=False)
    colors={'A':'#333333','B':'#287bb5','C':'#ef8a22','D':'#36a657'}
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for ax,fold in zip(axes,['fold_a','fold_b']):
        q=wide.loc[fold];x=np.arange(len(q))
        for j,group in enumerate('ABCD'):
            std=mean[(mean.fold==fold)&(mean.group==group)].set_index('case').loc[q.index,'seed_std']
            ax.bar(x+(j-1.5)*.19,q[group],.19,yerr=std,capsize=2,label=group,color=colors[group])
        ax.set_xticks(x,[str(int(c)) for c in q.index]);ax.set(title=fold,xlabel='Test case',ylabel='MAE (mm), mean ± seed SD');ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('Validation-selected dimensions: A=0, B=18, C=12, D=18; error bars are not case confidence intervals')
    fig.savefig(dest/'case_mae.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(19,15),layout='constrained')
    for ax,((fold,case),q) in zip(axes.ravel(),p.groupby(['fold','case'])):
        actual=q[['run','time','vb_mm']].drop_duplicates().sort_values('time');ax.plot(actual.time,actual.vb_mm,'ko-',lw=2,ms=3,label='Measured VB')
        for group in 'ABCD':
            s=q[q.group==group].groupby('time').predicted_vb_mm.agg(['mean','std'])
            ax.plot(s.index,s['mean'],color=colors[group],label=group);ax.fill_between(s.index,s['mean']-s['std'],s['mean']+s['std'],color=colors[group],alpha=.10)
        ax.set(title=f'{fold}: case {int(case)} (n={len(actual)})',xlabel='Dataset time',ylabel='VB (mm)');ax.grid(alpha=.2)
    for ax in axes.ravel()[14:]:ax.set_visible(False)
    axes[0,0].legend(fontsize=8,ncol=3)
    fig.suptitle('Predictions averaged over 5 seeds; shading = seed SD (not confidence interval)',fontsize=15)
    fig.savefig(dest/'case_trajectories.png',dpi=130);plt.close(fig)
    lines=['# 测试case误差诊断','',
        'A：工况+时间；B：时域18；C：频域12；D：联合18。维数全部由内层验证选出；每个case先计算每个seed的MAE，再平均五seed，非先平均预测再算MAE。','',md(table.reset_index()),
        '','## 是否是单一异常case拖累？','',
        '确有误差较大的case，但不能称为已确认的坏数据。case5、8、13、14值得优先检查；其中5/8/14标签较少（6/5/7个），主指标对每个case等权。case13/14有超出fold A训练最大VB=0.74的观测。',
        'case5对时域B的退步贡献特别明显；频域C反而改善该case。C相对A的退步分布在case10、11、12、13及3、4、8等，不是单一case造成。联合D也包含多case退步。',
        '下面仅为事后敏感性分析：每次从计分中临时去掉一个case，不重训、不重新选维数、不作为正式成绩。remaining_delta为剩余case上sensor组减A的MAE，正值代表仍差于A。','',
        md(loo[(loo.scope=='overall')&(loo.omitted_case==5)]),
        '','## 图与明细','',
        '![各case MAE](case_mae.png)','', '![预测轨迹](case_trajectories.png)','',
        'per_case_summary.csv含每case的MAE、seed标准差和有符号偏差；per_case_seed.csv保留五seed；largest_point_errors.csv定位具体run；leave_one_case_out_diagnostic.csv包含全部单case剔除诊断。seed标准差不是独立case置信区间。',
        '此前case12/run1的无效sensor已跳过；本轮case12频域误差较大不能直接归因于旧的坏首记录。若要判断新的输入异常，需要回到原始波形及质量证据，不能按测试误差定义清洗规则。']
    (dest/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(table.round(5).to_string());print(loo[(loo.scope=='overall')&(loo.omitted_case==5)].to_string(index=False))
    print('Minimum overall delta after omitting any one case:',loo[loo.scope=='overall'].groupby('group').remaining_delta.min().to_dict())

if __name__=='__main__':main()
