"""Visualize raw input-integrity failures without modifying data or training."""
import json
from collections import Counter
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path, write_json
from ..data.features import SENSORS, sensor_features

ROOT=project_path('wear_monitoring_dynamics/outputs/input_quality_20261008')


def main():
    ROOT.mkdir(exist_ok=True)
    records=np.load(project_path('mill_numpy/mill.npy'),allow_pickle=True).tolist()
    original=loadmat(project_path('mill/mill.mat'),simplify_cells=True)['mill']
    by_key={(int(r['case']),int(r['run'])):r for r in records}
    for a in original:
        b=by_key[(int(a['case']),int(a['run']))]
        for c in SENSORS:
            np.testing.assert_array_equal(np.asarray(a[c]).ravel(),np.asarray(b[c]).ravel())
    stats=[]
    for r in records:
        for c in SENSORS:
            x=np.asarray(r[c]).ravel()
            stats.append(dict(case=r['case'],run=r['run'],time=r['time'],VB=r['VB'],channel=c,
                n=len(x),peak_abs=float(abs(x).max()),std=float(x.std()),rms=float(np.sqrt(np.mean(x*x))),
                energy=float(np.mean(x*x)),median=float(np.median(x)),q01=float(np.quantile(x,.01)),q99=float(np.quantile(x,.99)),
                fraction_abs_lt_1e_6=float(np.mean(abs(x)<1e-6)),count_abs_gt100=int(np.sum(abs(x)>100))))
    stats=pd.DataFrame(stats);stats.to_csv(ROOT/'raw_channel_statistics.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    fig,axes=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    for ax,c in zip(axes.ravel(),SENSORS):
        for run,color in [(1,'#c0392b'),(2,'#2471a3')]:
            x=np.asarray(by_key[(12,run)][c]).ravel()
            ax.plot(np.arange(len(x)),x,color=color,lw=.6,alpha=.85,label=f'Run {run}: n={len(x):,}')
            peaks=np.flatnonzero(abs(x)>100)
            if len(peaks):ax.scatter(peaks,x[peaks],s=23,color=color,zorder=4)
        ax.set_yscale('symlog',linthresh=1e-6);ax.set_title(c);ax.set_xlabel('Sample index within record')
        ax.set_ylabel('Stored signal amplitude (symlog)');ax.grid(alpha=.15);ax.legend(fontsize=8,loc='upper right')
    fig.suptitle('Case 12: suspicious run 1 versus neighboring run 2\nRun 1: mostly near-zero positive values, isolated identical 2771.19 spikes, unusual length',fontsize=14)
    fig.savefig(ROOT/'case12_raw_signals.png',dpi=160);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,4.6),layout='constrained')
    for ax,c in zip(axes,['smcAC','vib_spindle']):
        for run,color in [(1,'#c0392b'),(2,'#2471a3')]:
            x=np.asarray(by_key[(2,run)][c]).ravel()
            ax.plot(np.arange(len(x)),x,color=color,lw=.7,label=f'Run {run}')
        ax.set_yscale('symlog',linthresh=1);ax.set_title(c);ax.set_xlabel('Sample index');ax.set_ylabel('Stored amplitude (symlog)')
        ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Case 2 / run 1: corruption-like tail with amplitudes up to 1.04e34\nAlready quarantined by the existing raw-amplitude rule',fontsize=13)
    fig.savefig(ROOT/'case2_raw_signals.png',dpi=160);plt.close(fig)

    previous=project_path('wear_monitoring_dynamics/outputs/history_phase13_20261008')
    pred=pd.read_csv(previous/'phase1_test_predictions.csv')
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    subset=stats[(stats.case==12)&(stats.channel=='smcAC')].sort_values('time')
    ax=axes[0]
    for column,color in [('std','#e67e22'),('energy','#8e44ad')]:
        ax.plot(subset.time,subset[column],'-o',ms=4,color=color,label=column)
    ax.set_yscale('log');ax.set_xlabel('Dataset time field');ax.set_ylabel('Feature value (log scale)')
    ax.set_title('One bad record distorts summary features');ax.legend();ax.grid(alpha=.2)
    ax=axes[1]
    for name,color in [('mlp_L1','#27ae60'),('mlp_L3','#c0392b'),('cnn_L3','#8e44ad')]:
        q=pred[(pred.case==12)&(pred.name==name)].groupby('time').predicted_vb_mm.mean()
        ax.plot(q.index,q.values,'-o',ms=4,color=color,label=name+' (5-seed mean)')
    real=subset[subset.VB.notna()]
    ax.plot(real.time,real.VB,'k-o',label='Measured VB',ms=4)
    ax.axvspan(1,6,color='#c0392b',alpha=.08,label='Run 1 enters L=3 windows')
    ax.set_yscale('symlog',linthresh=.2);ax.set_xlabel('Dataset time field');ax.set_ylabel('VB (mm; symlog)')
    ax.set_title('Case 12: history spreads the anomaly to labeled runs');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(ROOT/'case12_features_predictions.png',dpi=160);plt.close(fig)
    summary=stats.groupby(['case','run']).agg(n=('n','max'),peak_abs=('peak_abs','max')).reset_index()
    summary.to_csv(ROOT/'record_overview.csv',index=False)
    write_json(ROOT/'audit.json',dict(records=len(records),channels_per_record=6,mat_npy_arrays_equal=True,
        lengths={str(k):v for k,v in Counter(len(np.asarray(r['smcAC']).ravel()) for r in records).items()},
        no_training_or_data_changes=True))
    selected=stats[(stats.case==12)&stats.run.isin([1,2])]
    selected.to_csv(ROOT/'case12_comparison.csv',index=False)
    lines=['# 输入质量诊断：2026-10-08','',
        '未更改数据、质量规则或训练结果。使用本地原始MAT与NPY逐记录逐通道交叉核验，167×6个数组完全一致。只说明异常已在本地MAT中，不能证明官方原始分发也有同样异常。','',
        '## 明显异常','',
        '- case2/run1：各通道出现极端数值，最高约1.04e34；当前raw_sensor_abs_limit=1e6已隔离整条记录。不是删除整个case2。',
        '- case12/run1：唯一15360点记录，其余166条9000点。VB缺失；所有通道约80%的样本绝对值小于1e-6，且四通道出现相同2771.189697尖峰。远不只是“磨损大导致振幅大”。',
        '- 长度不同或VB缺失本身不是坏数据判据；联合数值分布、多个通道、波形形态与相邻记录差异，才构成强烈的完整性疑点。',
        '- 本次peak>100与长度不同的描述性筛查定位上述两条；不代表其他记录全部正常，也不是预先验证过的部署阈值。','',
        '## 图','',
        '![case12](case12_raw_signals.png)','', '![case2](case2_raw_signals.png)','',
        '![feature and prediction effects](case12_features_predictions.png)','',
        '波形横轴是采样点索引，不假设采样率；纵轴为原文件数值，未假定物理单位。symlog用于同时展示近零、正常幅值和极端值，不是训练时变换。特征/预测横轴使用原始time字段。',
        'VB真值从case12/run2的0.05mm开始，没有run1的VB标签。预测曲线是固定500epoch的5seed均值。','',
        '## 影响链条','',
        '原始异常→std/rms/energy/max等特征失真→训练标准化不能消除异常→历史窗口重复使用该记录→后续有标签目标出现极端VB估计。首次sensor初始化也会把错误VB输入ODE。',
        'fold A将case12作为测试，因此异常影响推断；fold B将case12作为训练，即使run1没有VB标签，也会影响sensor标准化和后续窗口的监督梯度。',
        '既有敏感性实验仅把该记录替换为训练均值，固定模型不变，受影响L3窗口MAE显著恢复；这支持定位，但不是用测试数据决定清洗后可直接宣称的正式性能提升。','',
        '## 建议','',
        '1. 保留原文件和校验记录；如能获得可信的原始数据副本，再交叉核验或恢复。无法恢复时，先隔离具体记录，不删除整case。',
        '2. 将信号有效性与VB标签有效性分开；允许保留有效VB用于ODE，但坏sensor不能进入特征拟合、标准化、历史编码或初始化。case2已有VB标签，可在独立协议下研究保留其标签的收益，不能静默改变现有比较。',
        '3. 统一使用预设完整性规则及仅在训练case校准的通道质量阈值；长度仅作标志，不能单独据此删除。避免根据测试误差或VB残差定阈值。',
        '4. 两种可部署输入策略：跳过坏sensor记录并按真实时间间隔构建最近L个有效观测；或保留时间位置，用训练均值填充并显式提供质量mask。坏的首记录应延迟sensor初始化或使用预先训练的ct后备估计，不能等待第一个测试VB标签来决定。',
        '5. 校验通过后再比较稳健缩放/训练分位数裁剪；单纯换RobustScaler不保证抑制极端幅值。不能只削掉case12的几个尖峰，因为六通道其余大部分数值也异常。',
        '6. 重新做所有内层特征筛选、标准化、网络训练和验证选择，再报告新协议结果。对所有结构采用同一质量规则，并保留原协议作对照。',
        '7. 非负VB输出/初始化保护作为独立模型消融；它防止负VB，但不能修复损坏输入，也不能替代质量控制。','',
        '复现：`python -m wear_monitoring_dynamics.scripts.inspect_input_quality`。']
    (ROOT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(selected[['run','channel','n','peak_abs','std','energy','fraction_abs_lt_1e_6']].to_string(index=False))


if __name__=='__main__':main()
