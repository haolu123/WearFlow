"""Current-only distribution and matched-condition descriptive diagnostics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path
from ..data.records import load_table
from ..data.feature_archive import attach_features
from ..data.sensor_quality import apply_quality
from .run_psd_sweep import ROOT

def md(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.4f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    dest=ROOT/'current_condition_diagnostics';dest.mkdir(exist_ok=True)
    manifest=pd.read_csv(ROOT/'study_manifest.csv');details=[];case_rows=[];point_rows=[]
    for fold in ['fold_a','fold_b']:
        r=manifest[(manifest.fold==fold)&(manifest.role=='final')&(manifest.group=='A')&(manifest.seed==0)].iloc[0]
        d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config'];scaler=json.loads((d/'scalers.json').read_text());split=json.loads((d/'split.json').read_text())
        t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))
        t=apply_quality(attach_features(t,cfg['data']['feature_archive']),cfg['data']['sensor_quality'])
        names=cfg['data']['feature_archive']['names'];idx=[j for j,f in enumerate(names) if f.startswith(('smcAC_','smcDC_'))];assert len(idx)==50
        x=t['sensor'];valid=t['sensor_valid'];train=valid&np.isin(t['case'],split['train']);mu,sd=map(np.asarray,scaler['sensor']);z=(x-mu)/sd
        if fold=='fold_a':
            for case in np.unique(t['case']):
                mask=valid&(t['case']==case);i=np.flatnonzero(mask)[0]
                for j in idx:
                    low=mask&np.isfinite(t['VB'])&(t['VB']<=.2)
                    case_rows.append(dict(case=int(case),DOC=t['DOC'][i],feed=t['feed'][i],material=int(t['material'][i]),feature=names[j],median=np.median(x[mask,j]),minimum=x[mask,j].min(),maximum=x[mask,j].max(),n=mask.sum(),low_vb_n=low.sum(),low_vb_median=np.median(x[low,j]) if low.any() else np.nan))
        for case in split['test']:
            mask=valid&(t['case']==case);i=np.flatnonzero(mask)[0]
            matched=train.copy()
            for key in ['DOC','feed','material']:matched &= t[key]==t[key][i]
            assert matched.any();partners=sorted(set(t['case'][matched].astype(int)))
            for j in idx:
                values=x[mask,j];lo=x[train,j].min();hi=x[train,j].max();ml=x[matched,j].min();mh=x[matched,j].max()
                details.append(dict(fold=fold,case=case,matched_case=partners[0],feature=names[j],train_min=lo,train_max=hi,matched_min=ml,matched_max=mh,test_min=values.min(),test_median=np.median(values),test_max=values.max(),max_abs_z=abs(z[mask,j]).max(),median_z=np.median(z[mask,j]),pct_outside_global=100*np.mean((values<lo)|(values>hi)),pct_outside_matched=100*np.mean((values<ml)|(values>mh))))
            for i in np.flatnonzero(mask):
                for j in idx:point_rows.append(dict(fold=fold,case=case,run=int(t['run'][i]),time=t['time'][i],VB=t['VB'][i],feature=names[j],raw=x[i,j],z=z[i,j]))
    det=pd.DataFrame(details);cases=pd.DataFrame(case_rows);points=pd.DataFrame(point_rows)
    det.to_csv(dest/'current_feature_shift.csv',index=False);cases.to_csv(dest/'current_case_statistics.csv',index=False);points.to_csv(dest/'current_record_z.csv',index=False)
    # Balanced 2x2x2 design, two cases per condition. Cases 6/15 included only in descriptive effects.
    effects=[]
    for feature,q in cases.groupby('feature'):
        for factor in ['material','DOC','feed']:
            low,high=sorted(q[factor].unique());a=q[q[factor]==low]['median'];b=q[q[factor]==high]['median'];s=q['median'].std(ddof=0)
            delta=b.mean()-a.mean();ss=((q['median']-q['median'].mean())**2).sum()
            eta=(len(a)*(a.mean()-q['median'].mean())**2+len(b)*(b.mean()-q['median'].mean())**2)/ss if ss else 0
            effects.append(dict(feature=feature,factor=factor,low_level=low,high_level=high,mean_case_median_low=a.mean(),mean_case_median_high=b.mean(),standardized_difference=delta/s if s else 0,eta_squared=eta))
    effects=pd.DataFrame(effects);effects.to_csv(dest/'condition_associations.csv',index=False)
    selected=['smcAC_rms','smcAC_psd_ac_power','smcAC_psd_centroid_hz','smcAC_psd_band_5_15_fraction','smcDC_mean','smcDC_psd_ac_power','smcDC_psd_centroid_hz','smcDC_psd_band_15_30_power']
    fig,axes=plt.subplots(4,2,figsize=(16,15),layout='constrained')
    conditions=cases[['material','DOC','feed']].drop_duplicates().sort_values(['material','DOC','feed'])
    labels=[]
    for ax,feature in zip(axes.ravel(),selected):
        q=cases[cases.feature==feature];labels=[]
        for position,cond in enumerate(conditions.itertuples(index=False)):
            a=q[(q.material==cond.material)&(q.DOC==cond.DOC)&(q.feed==cond.feed)].sort_values('case')
            labels.append(f'M{cond.material}\nd{cond.DOC:g}\nf{cond.feed:g}')
            for offset,row in zip([-.12,.12],a.itertuples()):
                color='#287bb5' if row.case<=8 else '#e68632'
                ax.vlines(position+offset,row.minimum,row.maximum,color=color,alpha=.5,lw=2)
                ax.scatter(position+offset,row.median,color=color,s=30)
                ax.annotate(str(row.case),(position+offset,row.median),xytext=(3,4),textcoords='offset points',fontsize=8)
        ax.set_xticks(range(8),labels,fontsize=8);ax.set_title(feature);ax.set_ylabel('Median and min–max across valid runs');ax.grid(alpha=.2)
        if 'power' in feature:ax.set_yscale('log')
    fig.suptitle('Same machining condition, two repeated cases: blue=1–8; orange=9–16\nLabels are case IDs. M=material, d=depth, f=feed. Includes cases 6/15 for description only.',fontsize=13)
    fig.savefig(dest/'current_by_condition.png',dpi=130);plt.close(fig)
    summary=det.assign(channel=det.feature.str.split('_').str[0]).groupby(['fold','case','matched_case','channel']).agg(max_abs_z=('max_abs_z','max'),mean_pct_outside_global=('pct_outside_global','mean'),mean_pct_outside_matched=('pct_outside_matched','mean')).reset_index()
    summary.to_csv(dest/'case_current_summary.csv',index=False)
    lines=['# 电流特征：分布偏移与工况','',
        '仅检查smcAC/smcDC共50维（18时域＋32PSD），不是只看当前模型选中的特征。无新训练或清洗。以实际外层训练scaler为基准；同工况对照仅使用对应fold训练case。所有有效sensor记录均参与，包括无VB标签。','',
        '## 测试case与同工况训练case','',md(summary),
        '','## 偏移最大的特征','',md(det.sort_values('max_abs_z',ascending=False).head(16)),
        '','## 工况关联（case等权）','',md(effects[effects.feature.isin(selected)]),
        '','工况关联使用全部16case的每case特征中位数（含6/15仅作描述）。2材料×2切深×2进给各有两次实验。eta²为单因素对16个case中位数变异的描述比例，不是因果贡献或预测性能；特征中位数混有磨损覆盖、时间范围和重复实验差异，不能把run当独立重复。完整低VB≤0.2的统计也保存在current_case_statistics.csv，但部分case没有该阶段标签，未强行作平衡比较。',
        '同工况训练case范围更窄，超出比例高不直接等于更严重异常；但同工况仍明显偏离，说明仅靠名义工况不能解释全部差异。原始range包含真实磨损进展，不能按测试范围再拟合归一化。',
        '','![按工况对照](current_by_condition.png)','',
        'current_feature_shift.csv为每case每特征的训练/同工况/测试范围；current_record_z.csv定位具体run；condition_associations.csv为全部特征的工况关联。']
    (dest/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(summary.round(2).to_string(index=False));print(det.sort_values('max_abs_z',ascending=False).head(8).to_string(index=False));print(effects[effects.feature.isin(selected)][['feature','factor','standardized_difference','eta_squared']].round(3).to_string(index=False))

if __name__=='__main__':main()
