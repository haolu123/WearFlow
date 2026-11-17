"""Causal initial-reference feature diagnostics. No network training or test tuning."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path, write_json, file_hash
from ..data.records import load_table
from ..data.feature_archive import attach_features
from ..data.sensor_quality import apply_quality
from ..data.splits import POOLS
from .diagnose_sensor_learning import md

ROOT=project_path('wear_monitoring_dynamics/outputs/relative_initial_features_20261012')
PARENT=ROOT.with_name('current_gate_20261012')
ACTIVE=sorted(POOLS['fold_a']+POOLS['fold_b'])
NONNEG_SUFFIX=('_std','_rms','_peak_to_peak','_energy','_power','_fraction')


def anchors(table,count=1):
    result={}
    for case in ACTIVE:
        ids=np.flatnonzero((table['case']==case)&table['sensor_valid'])
        ids=ids[np.argsort(table['run'][ids],kind='stable')]
        assert len(ids)>=count
        result[case]=ids[:count]
    return result


def centered(values,case_ids,refs):
    out=values.copy()
    for case,ids in refs.items():out[case_ids==case]-=np.median(values[ids],axis=0)
    return out


def scale_within(values,table,train_cases):
    """Equal-case RMS within-case SD; removes between-case offset from denominator."""
    vv=[]
    for case in train_cases:
        mask=(table['case']==case)&table['sensor_valid']&np.isfinite(table['VB'])
        vv.append(np.var(values[mask],axis=0))
    return np.sqrt(np.mean(vv,axis=0))


def gap_between(axis_a,values_a,axis_b,values_b):
    a=pd.DataFrame(dict(x=axis_a,y=values_a)).groupby('x').y.mean().sort_index()
    b=pd.DataFrame(dict(x=axis_b,y=values_b)).groupby('x').y.mean().sort_index()
    if len(a)<2 or len(b)<2:return None
    lo=max(a.index.min(),b.index.min());hi=min(a.index.max(),b.index.max())
    na=int(((a.index>=lo)&(a.index<=hi)).sum());nb=int(((b.index>=lo)&(b.index<=hi)).sum())
    if hi<=lo or min(na,nb)<2:return None
    grid=np.linspace(lo,hi,101)
    ya=np.interp(grid,a.index,a.values);yb=np.interp(grid,b.index,b.values)
    return dict(gap=float(np.trapezoid(abs(ya-yb),grid)/(hi-lo)),signed_gap=float(np.trapezoid(ya-yb,grid)/(hi-lo)),
                overlap_low=float(lo),overlap_high=float(hi),n_a=na,n_b=nb)


def feature_family(name):
    if '_psd_' not in name:return 'time_shape' if name.endswith(('skewness','kurtosis')) else 'time_level_amplitude'
    if name.endswith('_fraction'):return 'spectral_fraction'
    if name.endswith('_power'):return 'spectral_power'
    return 'spectral_shape'


def prepare(cfg,scalers):
    t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))
    t=apply_quality(attach_features(t,cfg['data']['feature_archive']),cfg['data']['sensor_quality'])
    indices=[j for j,f in enumerate(cfg['data']['feature_archive']['names']) if f.startswith(('smcAC_','smcDC_'))]
    names=[cfg['data']['feature_archive']['names'][j] for j in indices]
    raw=t['sensor'][:,indices];refs1=anchors(t,1);refs3=anchors(t,3)
    train_cases=scalers['train_cases'];mask=np.isin(t['case'],train_cases)&t['sensor_valid']
    log=np.full_like(raw,np.nan);floors={}
    for k,name in enumerate(names):
        if not name.endswith(NONNEG_SUFFIX):continue
        assert (raw[:,k]>=0).all()
        pos=raw[mask,k];pos=pos[pos>0]
        if not len(pos):continue
        floor=max(float(np.median(pos))*1e-6,1e-30)
        log[:,k]=np.log(np.maximum(raw[:,k],floor));floors[name]=floor
    values=dict(absolute=raw,delta_first=centered(raw,t['case'],refs1),delta_first3=centered(raw,t['case'],refs3),
                log_absolute=log,log_ratio_first=centered(log,t['case'],refs1),log_ratio_first3=centered(log,t['case'],refs3))
    sd=scale_within(raw,t,train_cases);lsd=scale_within(log,t,train_cases)
    return t,names,indices,values,refs1,refs3,sd,lsd,floors


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    mf=pd.read_csv(PARENT/'reused_sensor_manifest.csv');mf=mf[(mf.role=='final')&(mf.seed==0)]
    rows=[];anchor_rows=[];skips=[];protocols=[];cache={};pairs=None;floor_rows=[]
    for r in mf.itertuples():
        d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config'];s=json.loads((d/'scalers.json').read_text())
        t,names,indices,v,ref1,ref3,sd,lsd,floors=prepare(cfg,s)
        selected=[names.index(name) for name in cfg['monitor']['selected_features']]
        cache[r.fold]=(t,names,indices,v,ref1,ref3,sd,lsd,selected)
        assert len(indices)==50 and len(selected)==18
        # Pair only observed identical cutting conditions; do not assume case numbering.
        meta=pd.DataFrame({k:t[k] for k in ['case','DOC','feed','material']}).drop_duplicates()
        meta=meta[meta.case.isin(ACTIVE)]
        assert not meta.case.duplicated().any()
        pairs=[]
        for cond,g in meta.groupby(['DOC','feed','material']):
            cases=sorted(g.case.astype(int));assert len(cases)==2
            assert sum(c in POOLS['fold_a'] for c in cases)==1
            pairs.append(tuple(cases))
        for case in ACTIVE:
            i=int(ref1[case][0]);ids=ref3[case]
            if r.fold=='fold_a':
                lab=np.flatnonzero((t['case']==case)&t['sensor_valid']&np.isfinite(t['VB']))
                anchor_rows.append(dict(case=case,first_sensor_run=int(t['run'][i]),first_sensor_time=float(t['time'][i]),
                    anchor_vb=float(t['VB'][i]),first_labeled_run=int(t['run'][lab[0]]),first_labeled_vb=float(t['VB'][lab[0]]),
                    first3_ready_run=int(t['run'][ids[-1]]),first3_ready_time=float(t['time'][ids[-1]]),
                    first3_vb_values=';'.join(map(str,t['VB'][ids])),n_labeled=len(lab)))
            for k,name in enumerate(names):
                if name in floors:
                    floor_rows.append(dict(fold=r.fold,case=case,feature=name,anchor_value=float(v['absolute'][i,k]),floor=floors[name],
                        anchor_floored=bool(v['absolute'][i,k]<floors[name])))
        for pool,ks in [('all50',range(50)),('selected18',selected)]:
            for a,b in pairs:
                pair=f'{a}-{b}'
                for anchor_kind,refs,delta_name,log_name in [('first',ref1,'delta_first','log_ratio_first'),('first3',ref3,'delta_first3','log_ratio_first3')]:
                    # All calibration records excluded for BOTH reference and transformed curves.
                    masks={c:((t['case']==c)&t['sensor_valid']&np.isfinite(t['VB'])&(t['run']>t['run'][refs[c][-1]])) for c in [a,b]}
                    for axis_kind in (['VB','delta_VB'] if anchor_kind=='first' else ['VB']):
                        if axis_kind=='delta_VB' and not all(np.isfinite(t['VB'][ref1[c][0]]) for c in [a,b]):
                            skips.append(dict(fold=r.fold,pool=pool,pair=pair,anchor=anchor_kind,axis=axis_kind,reason='unknown anchor VB'));continue
                        axes={c:t['VB'][masks[c]]-(t['VB'][ref1[c][0]] if axis_kind=='delta_VB' else 0) for c in [a,b]}
                        for k in ks:
                            for base,method,scale in [('absolute',delta_name,sd[k]),('log_absolute',log_name,lsd[k])]:
                                if not np.isfinite(scale) or scale<1e-12:continue
                                ga=gap_between(axes[a],v[base][masks[a],k],axes[b],v[base][masks[b],k])
                                gd=gap_between(axes[a],v[method][masks[a],k],axes[b],v[method][masks[b],k])
                                if ga is None or gd is None:
                                    skips.append(dict(fold=r.fold,pool=pool,pair=pair,anchor=anchor_kind,axis=axis_kind,feature=names[k],reason='insufficient measured overlap'));continue
                                assert ga['n_a']==gd['n_a'] and ga['overlap_low']==gd['overlap_low']
                                rho=[]
                                for c in [a,b]:
                                    x=pd.Series(v[base][masks[c],k]);y=pd.Series(axes[c])
                                    rho.append(float(x.corr(y,method='spearman')) if x.nunique()>1 and y.nunique()>1 else np.nan)
                                rows.append(dict(fold=r.fold,pool=pool,pair=pair,a=a,b=b,anchor=anchor_kind,axis=axis_kind,feature=names[k],
                                    baseline=base,method=method,baseline_gap=ga['gap']/scale,relative_gap=gd['gap']/scale,
                                    delta_gap=(gd['gap']-ga['gap'])/scale,improved=gd['gap']<ga['gap'],
                                    ratio=gd['gap']/ga['gap'] if ga['gap']>1e-12 else np.nan,
                                    overlap_low=ga['overlap_low'],overlap_high=ga['overlap_high'],n_a=ga['n_a'],n_b=ga['n_b'],
                                    rho_a=rho[0],rho_b=rho[1],within_case_scale=scale))
        protocols.append(dict(fold=r.fold,source_run=str(d),train_cases=s['train_cases'],selected_features=cfg['monitor']['selected_features'],
            log_floors=floors,raw_within_sd=sd.tolist(),log_within_sd=[float(x) if np.isfinite(x) else None for x in lsd]))
    data=pd.DataFrame(rows);data.to_csv(ROOT/'pair_feature_gaps.csv',index=False)
    pd.DataFrame(anchor_rows).to_csv(ROOT/'anchors.csv',index=False)
    pd.DataFrame(skips).to_csv(ROOT/'skipped_comparisons.csv',index=False)
    pd.DataFrame(floor_rows).to_csv(ROOT/'log_floor_audit.csv',index=False)
    group=['fold','pool','pair','anchor','axis','method']
    pair=data.groupby(group).agg(baseline_gap=('baseline_gap','mean'),relative_gap=('relative_gap','mean'),
        median_delta=('delta_gap','median'),features_improved=('improved','sum'),features=('improved','size'),
        n_a=('n_a','first'),n_b=('n_b','first')).reset_index()
    pair['change_percent']=100*(pair.relative_gap/pair.baseline_gap-1)
    pair.to_csv(ROOT/'pair_summary.csv',index=False)
    # Each fold/pair receives one vote; features/folds do not supply independent replications.
    summary=pair.groupby(['pool','anchor','axis','method']).agg(baseline_gap=('baseline_gap','mean'),relative_gap=('relative_gap','mean'),
        improved_pair_directions=('change_percent',lambda x:int((x<0).sum())),pair_directions=('change_percent','size')).reset_index()
    summary['change_percent']=100*(summary.relative_gap/summary.baseline_gap-1)
    summary.to_csv(ROOT/'summary.csv',index=False)
    data['family']=data.feature.map(feature_family)
    families=data.groupby(['fold','pool','pair','anchor','axis','method','family']).agg(
        baseline_gap=('baseline_gap','mean'),relative_gap=('relative_gap','mean'),median_delta=('delta_gap','median'),
        fraction_improved=('improved','mean')).reset_index()
    families.to_csv(ROOT/'feature_family_pair_summary.csv',index=False)
    families.groupby(['pool','anchor','axis','method','family'])[['baseline_gap','relative_gap','median_delta','fraction_improved']].mean().reset_index().to_csv(ROOT/'feature_family_summary.csv',index=False)
    # Predefined diagnostic features from the preceding study, not ranked by improvement here.
    examples=[(3,11,'smcAC_psd_band_0_5_fraction'),(4,10,'smcAC_psd_band_0_5_fraction'),
              (3,11,'smcAC_rms'),(4,10,'smcDC_rms')]
    t,names,indices,v,ref1,ref3,sd,lsd,selected=cache['fold_a']
    fig,axs=plt.subplots(4,3,figsize=(14,13),layout='constrained')
    for row,(a,b,feature) in enumerate(examples):
        k=names.index(feature)
        for col,(method,axis_name) in enumerate([('absolute','VB'),('delta_first','VB'),('delta_first','delta_VB')]):
            ax=axs[row,col]
            for c,color in [(a,'tab:blue'),(b,'tab:orange')]:
                mask=(t['case']==c)&t['sensor_valid']&np.isfinite(t['VB']);x=t['VB'][mask].copy()
                if axis_name=='delta_VB':x-=t['VB'][ref1[c][0]]
                y=v[method][mask,k]/sd[k]
                ax.plot(x,y,'o-',color=color,label=f'Case {c} (VB0={t["VB"][ref1[c][0]]:.2f})',markersize=3)
            ax.set(title=f'{a} vs {b}: {feature}\n{method}',xlabel='VB (mm)' if axis_name=='VB' else 'VB - initial VB (mm)',ylabel='Feature / training within-case SD')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Initial-reference diagnosis; anchor points shown here but excluded from gap metrics',fontsize=14)
    fig.savefig(ROOT/'initial_reference_examples.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for row,(a,b) in enumerate([(3,11),(4,10)]):
        k=names.index('smcAC_psd_band_0_5_fraction')
        for col,method in enumerate(['absolute','delta_first']):
            ax=axs[row,col]
            for c,color in [(a,'tab:blue'),(b,'tab:orange')]:
                mask=(t['case']==c)&t['sensor_valid']&np.isfinite(t['VB'])
                ax.plot(t['VB'][mask],v[method][mask,k]/sd[k],'o-',color=color,label=f'Case {c}',markersize=4)
            ax.set(title=f'Cases {a}/{b}: '+('original feature' if col==0 else 'feature minus its initial value'),
                   xlabel='Measured VB (mm)',ylabel='Feature / training within-case SD')
            ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Same AC 0-5 Hz power-fraction feature: centering can worsen or improve alignment')
    fig.savefig(ROOT/'centering_counterexample.png',dpi=160);plt.close(fig)
    q=pair[(pair.pool=='all50')&(pair.axis=='VB')&(pair.anchor=='first')]
    q=q.groupby(['pair','method'])[['baseline_gap','relative_gap']].mean().reset_index()
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,method in zip(axs,['delta_first','log_ratio_first']):
        z=q[q.method==method].set_index('pair');z[['baseline_gap','relative_gap']].plot.bar(ax=ax)
        ax.set(title=method+'; matched absolute VB',ylabel='Mean gap / training within-case SD',xlabel='Same-condition pair')
        ax.grid(axis='y',alpha=.2);ax.tick_params(axis='x',rotation=0)
    fig.savefig(ROOT/'pair_gaps.png',dpi=160);plt.close(fig)
    # Complete per-direction 18-feature plots, including unsuccessful transformations.
    for fold,(t,names,indices,v,ref1,ref3,sd,lsd,selected) in cache.items():
        for a,b in [(3,11),(4,10)]:
            fig,axs=plt.subplots(6,3,figsize=(15,19),layout='constrained')
            for ax,k in zip(axs.flat,selected):
                for c,color in [(a,'tab:blue'),(b,'tab:orange')]:
                    mask=(t['case']==c)&t['sensor_valid']&np.isfinite(t['VB'])
                    ax.plot(t['VB'][mask]-t['VB'][ref1[c][0]],v['delta_first'][mask,k]/sd[k],'o-',label=str(c),color=color,markersize=3)
                ax.set(title=names[k],xlabel='Delta VB (mm)',ylabel='Delta feature / training SD');ax.grid(alpha=.2)
            axs.flat[0].legend(title='Case');fig.suptitle(f'{fold}: selected 18 features; cases {a}/{b}; initial-reference change')
            fig.savefig(ROOT/f'all18_{fold}_{a}_{b}.png',dpi=110);plt.close(fig)
    write_json(ROOT/'protocol.json',dict(scope='diagnostic only; no model training or feature selection',active_cases=ACTIVE,pairs=pairs,
        anchor='first valid sensor record, without consulting VB; optional first3 median only after third record',
        gap='trapezoid average absolute distance on 101-point common wear grid; no extrapolation; at least two observed distinct x per case in overlap',
        anchor_excluded_from_metrics=True,normalization='training-case RMS within-case SD, shared by absolute and centered variants',
        axis='absolute VB for all pairs; delta VB only when both first sensor anchors have measured VB; no VB imputation',
        logs='only prespecified nonnegative amplitude/power/fraction features; log(max(x,training-positive-median*1e-6)); compare log-absolute with log-relative',
        first3='median of first 3 valid sensor records; calibration rows excluded in both arms; absolute VB axis only',
        note='Curve alignment is not prediction accuracy. 101 interpolation locations and correlated features are not independent samples.',
        source_sha256=file_hash(__file__),folds=protocols))
    write_json(ROOT/'audit.json',dict(completed=True,diagnostic_only=True,training_runs=0,cases=len(ACTIVE),condition_pairs=len(pairs),
        pair_feature_comparisons=len(data),log_anchor_floor_count=int(pd.DataFrame(floor_rows).anchor_floored.sum()),
        unknown_initial_vb_cases=[x['case'] for x in anchor_rows if not np.isfinite(x['anchor_vb'])]))
    pairview=pair[(pair.pool=='selected18')&(pair.anchor=='first')&(pair.method=='delta_first')].groupby(['axis','pair'])[['baseline_gap','relative_gap']].mean().reset_index()
    pairview['change_percent']=100*(pairview.relative_gap/pairview.baseline_gap-1)
    pairview.to_csv(ROOT/'selected18_pair_readout.csv',index=False)
    selected=data[(data.pool=='selected18')&(data.anchor=='first')&(data.method=='delta_first')]
    counts=selected.groupby('axis').agg(improved=('improved','sum'),comparisons=('improved','size'),median_delta=('delta_gap','median')).reset_index()
    counts.to_csv(ROOT/'selected18_improvement_counts.csv',index=False)
    body='''# 相对初始状态特征：数据诊断

已完成，无新网络训练。结论是：**直接把18维绝对特征统一替换为“减去初值”，没有总体改善跨case曲线一致性；存在有用的局部改善，也有更大的反例。** 目前不宜直接把它当作已验证的解决方案。

## 范围与公平比较

使用现有14个研究case、7对同工况case。主分析保留两fold各自训练内选出的18维名单；另检查全部50维AC/DC电流候选，避免只展示有利特征。没有按本次测试诊断结果重新选特征。

对每条特征分别比较原始绝对值、减去首条有效sensor值；对预先定义的非负幅值/功率/占比特征，另比较log绝对值与log相对初值。对数使用训练正值中位数×1e-6作下限，首条锚点没有触发下限。绝对值与其差分版本使用完全相同的尺度；log与log差分也使用完全相同尺度。

本诊断从原始特征出发，没有先做旧网络输入的1%/99%裁剪。尺度是对应fold训练case的case内标准差平方平均再开方，减少case间常数偏移对归一化分母的影响。因此这里的距离单位不等于旧报告的clip-zscore单位。

用两case共同VB范围内的分段线性插值曲线间绝对距离衡量一致性，不外推；每case在共同区间内至少有两个不同的实测VB。积分网格101点只用于数值积分，不能当101个独立样本。首条锚点从距离计算中排除，避免“两个起点强制为零”制造改善。所有处理只依赖当前及此前可用sensor，训练尺度只用训练case。

## 首条sensor不等于零磨损

'''+md(pd.DataFrame(anchor_rows)[['case','first_sensor_run','anchor_vb','first_labeled_run','first_labeled_vb']])+'''

case2/run1已被既有原始数值规则隔离；case12/run1被既有质量规则判为无效，二者从run2取sensor锚点。case13/14/16首条有效sensor没有VB标签，不用未来标签插值补齐初始VB。它们可以计算相对sensor输入，但不能在本诊断中计算真实Delta VB。

## 主结果：沿用两fold的18维名单

正change_percent表示跨case距离变大，负值表示更接近。数字不是预测MAE，也不证明某种表示训练后的预测性能。

'''+md(summary[(summary.pool=='selected18')&(summary.anchor=='first')])+'''

在相同绝对VB下，减初值的平均跨case距离增加约29.8%，七对case中只有4/10总体改善。log相对初值相对log绝对值的平均距离增加约46.8%，也不是普遍有效。

对已知初始VB的四对case，改在相同Delta VB下比较，减初值的总体距离增加约4.9%。但不少单特征确实改善：

'''+md(counts)+'''

这些“特征×方向”组合相关性很强，同一对case会按两个fold的名单重复评估，不能当成独立统计样本。它说明收益并非不存在，而是局部大退化抵消了许多小改善。

### 逐对case

'''+md(pairview)+'''

例如，相同绝对VB下4/10平均改善约24.5%，3/11反而恶化约35.9%。相同Delta VB下，2/12改善约36.2%、4/10改善约15.3%；1/9和3/11仍变差。1/9与3/11起点VB均为0，说明问题不能全部归因于起始磨损不同。

## 最直观的反例：变化幅度并不相同

![初始校正反例](centering_counterexample.png)

上排3/11的AC 0–5 Hz功率占比：最初相差很大，后期却逐渐接近。减初值相当于把这份早期差异沿整个轨迹固定扣除，于是原来逐渐靠近的后期曲线被拉开。按fold A训练尺度，共同VB区间距离从1.178增到3.428，约变为2.91倍。两个case的初始VB都是0；明显不符合“共同曲线加一个固定偏移”的简单假设。

同一特征在4/10上距离从2.331降到1.159，约减少50.3%。这说明处理是否有效与特征和case组合有关。完整图中也保留RMS等较稳定特征，避免只看失败例子。

## 多条早期记录作基准的敏感性

额外用前三条有效sensor的逐特征中位数作基准，仅在第三条之后使用；前三条记录在对照双方的距离计算中全部排除。由于剩余重叠点不足，只有五对case可比较，不能直接把其总体数值与首条锚点的七对case结果横比。

在这个匹配子集里，18维减去三点中位数相对同期绝对特征的距离仍增加约27.9%。三条早期记录也不能都当成“相同初始状态”：例如case5的前三次VB为0、0.16、0.29，case8为0、0.18、0.30。中位数可能混入实质磨损变化，而不只是降低测量噪声。

## 下一步思路，而非已验证结论

1. 暂不统一用差分替换绝对特征。更合理的候选是保留当前绝对值，同时提供初始值/变化量，让模型保留判断基准水平的机会；三者有代数冗余，需要控制容量和正则。
2. 初始状态不同且初始VB已知时，可研究Delta feature到Delta VB的关系；初始VB未知时，不能默认从零磨损起算。
3. 对3/11这类情况，后续应关注变化幅度和曲线形状，而不是仅估计常数偏移。可能涉及响应增益、阶段差异或信号组成；本诊断没有确定物理来源。
4. 若比较按特征类别选择绝对/相对表示，所有选择必须回到训练内验证；本次外层图只能提出假设，不能据此手工删除不利特征。

## 限制与复现

均为已查看测试数据后的探索性数据诊断。使用实测VB来配对曲线，只用于解释，不作为部署输入。相减或正比例缩放不会改变单case的Spearman排序，因此本次检查跨case距离，而不把相关系数不变解释为方法无效。曲线更接近仍不等于预测更准；绝对初始水平可能本身携带有用信息。

三项测试覆盖固定偏移消除、未来记录不影响过去变换、仅重叠范围比较和稀疏点拒绝。所有对比数据、跳过原因、log下限、锚点和训练尺度均已保存。原数据、模型和旧结果未改动。

```text
python -m wear_monitoring_dynamics.scripts.diagnose_relative_features
python -m unittest wear_monitoring_dynamics.tests.test_relative_features -v
```

- [逐特征距离](pair_feature_gaps.csv)
- [逐case对汇总](pair_summary.csv)
- [特征类别汇总](feature_family_summary.csv)
- [完整示例图](initial_reference_examples.png)
- [协议](protocol.json)
- all18_fold_a_3_11.png、all18_fold_b_3_11.png、all18_fold_a_4_10.png、all18_fold_b_4_10.png保留各方向完整18维变化曲线。
'''
    (ROOT/'RESULTS.md').write_text(body,encoding='utf-8')
    print(pd.DataFrame(anchor_rows).to_string(index=False));print(summary.to_string(index=False))
    print(pair[(pair.pool=='selected18')&(pair.anchor=='first')&(pair.method=='delta_first')].to_string(index=False))


if __name__=='__main__':main()
