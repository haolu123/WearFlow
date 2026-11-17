"""Descriptive feature/VB trajectories; never selects model inputs using test labels."""
import html
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path, write_json
from ..data.records import load_table
from ..data.features import SENSORS, STATISTICS
from ..data.splits import POOLS

ROOT=project_path('wear_monitoring_dynamics/outputs/sensor_feature_inspection_20260927')
PRIMARY=sorted(POOLS['fold_a']+POOLS['fold_b'])


def overview(frame, per):
    """Compact examples; complete 16-case pages remain available for every feature."""
    features=['smcAC__std','vib_table__min','AE_spindle__min']
    cases=[1,9,7,13]
    fig,axes=plt.subplots(3,4,figsize=(16,10))
    for i,name in enumerate(features):
        for j,case in enumerate(cases):
            ax=axes[i,j];g=frame[frame.case==case].sort_values('time');v=g[g.VB.notna()]
            ax.plot(g.time,g[name],'.-',color='#2463a6',ms=4)
            other=ax.twinx();other.plot(v.time,v.VB,'o--',color='#d95b16',ms=3)
            other.set_ylim(0,1.65)
            rho=per[(per.case==case)&(per.feature==name)].spearman.iloc[0]
            ax.set_title(f'{name}\nCase {case} | rho={rho:.2f}',fontsize=11)
            ax.set_xlabel('Dataset time');ax.set_ylabel('Feature (raw)',color='#2463a6',fontsize=9)
            other.set_ylabel('VB (mm)',color='#d95b16',fontsize=9)
            ax.tick_params(axis='y',labelcolor='#2463a6',labelsize=8);other.tick_params(axis='y',labelcolor='#d95b16',labelsize=8)
            ax.grid(alpha=.15)
    fig.suptitle('Feature trajectories vs measured VB | blue: feature; orange: VB\nExamples only: case 1/9 share conditions; case 7/13 share conditions. Feature axes are independent.',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(ROOT/'figures'/'overview_examples.png',dpi=130);plt.close(fig)


def corr(x,y,minimum=4):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    good=np.isfinite(x)&np.isfinite(y);x,y=x[good],y[good]
    if len(x)<minimum or np.ptp(x)==0 or np.ptp(y)==0:
        return np.nan
    return float(np.corrcoef(x,y)[0,1])


def partial_spearman(x,y,t):
    """Pearson correlation of rank residuals after a linear rank-time fit."""
    if len(x)<5 or np.ptp(x)==0 or np.ptp(y)==0:
        return np.nan
    rx,ry,rt=rankdata(x),rankdata(y),rankdata(t)
    design=np.column_stack([np.ones(len(t)),rt])
    ex=rx-design@np.linalg.lstsq(design,rx,rcond=None)[0]
    ey=ry-design@np.linalg.lstsq(design,ry,rcond=None)[0]
    if np.linalg.norm(ex)<1e-10*np.linalg.norm(rx) or np.linalg.norm(ey)<1e-10*np.linalg.norm(ry):
        return np.nan
    return corr(ex,ey,5)


def main():
    ROOT.mkdir(parents=True,exist_ok=True);figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
    table,signature=load_table(project_path('mill_numpy/mill.npy'),project_path('wear_monitoring_dynamics/outputs/cache'))
    names=[s+'__'+f for s in SENSORS for f in STATISTICS]
    frame=pd.DataFrame({k:table[k] for k in ('case','run','time','VB','DOC','feed','material')})
    frame[['case','run']]=frame[['case','run']].astype(int)
    frame[names]=table['sensor'];frame.to_csv(ROOT/'raw_features_and_vb.csv',index=False)
    case_meta=[];rows=[]
    for case,g in frame.groupby('case'):
        labeled=g[g.VB.notna()]
        case_meta.append(dict(case=case,n_records=len(g),n_labeled=len(labeled),primary=case in PRIMARY,
            DOC=g.DOC.iloc[0],feed=g.feed.iloc[0],material=g.material.iloc[0],time_vb_spearman=corr(rankdata(labeled.time),rankdata(labeled.VB))))
        for name in names:
            x,y,t=labeled[name].to_numpy(),labeled.VB.to_numpy(),labeled.time.to_numpy()
            rows.append(dict(case=case,feature=name,n=len(x),primary=case in PRIMARY,
                pearson=corr(x,y),spearman=corr(rankdata(x),rankdata(y)),
                feature_time_spearman=corr(rankdata(x),rankdata(t)),
                time_adjusted_spearman=partial_spearman(x,y,t)))
    per=pd.DataFrame(rows);per.to_csv(ROOT/'per_case_correlations.csv',index=False)
    overview(frame,per)
    pd.DataFrame(case_meta).to_csv(ROOT/'case_metadata.csv',index=False)
    summaries=[];labeled=frame[frame.case.isin(PRIMARY)&frame.VB.notna()]
    for name in names:
        q=per[(per.feature==name)&per.primary];r=q.spearman.dropna()
        summaries.append(dict(feature=name,valid_cases=len(r),median_abs_spearman=r.abs().median(),
            median_spearman=r.median(),min_spearman=r.min(),max_spearman=r.max(),
            cases_abs_lt_03=int((r.abs()<.3).sum()),cases_ge_05=int((r>=.5).sum()),cases_le_minus05=int((r<=-.5).sum()),
            positive_cases=int((r>0).sum()),negative_cases=int((r<0).sum()),
            pooled_spearman=corr(rankdata(labeled[name]),rankdata(labeled.VB)),
            time_adjusted_valid_cases=int(q.time_adjusted_spearman.notna().sum()),
            median_abs_time_adjusted=q.time_adjusted_spearman.abs().median()))
    summary=pd.DataFrame(summaries).sort_values('median_abs_spearman',ascending=False)
    summary.to_csv(ROOT/'feature_summary.csv',index=False)
    # Raw feature scale and real VB scale, never normalize across cases to force agreement.
    for name in names:
        fig,axes=plt.subplots(4,4,figsize=(18,13))
        for (case,g),ax in zip(frame.groupby('case'),axes.flat):
            g=g.sort_values('time',kind='stable');v=g[g.VB.notna()]
            ax.plot(g.time,g[name],'.-',color='#2463a6',lw=1,ms=3)
            ax2=ax.twinx();ax2.plot(v.time,v.VB,'o--',color='#d95b16',lw=1.3,ms=3)
            ax2.set_ylim(0,1.05*frame.VB.max())
            r=per[(per.case==case)&(per.feature==name)].spearman.iloc[0]
            rho=f'{r:.2f}' if np.isfinite(r) else 'NA'
            ax.set_title(f'Case {case} | nVB={len(v)} | rho={rho}'+(' [extra]' if case not in PRIMARY else ''),fontsize=10)
            ax.tick_params(axis='y',labelcolor='#2463a6',labelsize=7);ax2.tick_params(axis='y',labelcolor='#d95b16',labelsize=7)
            ax.tick_params(axis='x',labelsize=7);ax.set_xlabel('Dataset time',fontsize=8)
            ax.grid(alpha=.15);ax.set_ylabel('Feature (raw)',fontsize=8,color='#2463a6');ax2.set_ylabel('VB (mm)',fontsize=8,color='#d95b16')
        fig.suptitle(name+' vs measured VB | blue: feature, orange: VB\nIndependent feature axes; common VB scale. Lines connect observations only; no VB imputation.',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(figdir/f'{name}.png',dpi=125);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(15,11),sharey=True)
    for (case,g),ax in zip(frame.groupby('case'),axes.flat):
        v=g[g.VB.notna()].sort_values('time');ax.plot(v.time,v.VB,'o-',color='#d95b16',ms=3)
        ax.set_title(f'Case {case} | DOC={g.DOC.iloc[0]:g}, feed={g.feed.iloc[0]:g}, mat={g.material.iloc[0]:g}'+(' [extra]' if case not in PRIMARY else ''),fontsize=9)
        ax.set_xlabel('Dataset time');ax.set_ylabel('VB (mm)');ax.grid(alpha=.2)
    fig.suptitle('Measured VB trajectories (no interpolation of missing labels)');fig.tight_layout(rect=(0,0,1,.96));fig.savefig(figdir/'vb_trajectories.png',dpi=130);plt.close(fig)
    matrix=per[per.primary].pivot(index='feature',columns='case',values='spearman').reindex(names)
    fig,ax=plt.subplots(figsize=(11,16));cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#dddddd')
    im=ax.imshow(matrix.to_numpy(),vmin=-1,vmax=1,cmap=cmap,aspect='auto')
    ax.set_yticks(range(54),names,fontsize=8);ax.set_xticks(range(len(matrix.columns)),[f'Case {c}' for c in matrix.columns],rotation=45,ha='right')
    for i in range(54):
        for j in range(len(matrix.columns)):
            v=matrix.iloc[i,j]
            if np.isfinite(v):ax.text(j,i,f'{v:.1f}',ha='center',va='center',fontsize=6,color='white' if abs(v)>.7 else 'black')
    for y in np.arange(8.5,54,9):ax.axhline(y,color='black',lw=.7)
    ax.set_title('Within-case feature / VB Spearman correlation\nGray = insufficient labels (<4) or constant values; descriptive, not significance')
    fig.colorbar(im,ax=ax,label='Spearman rho',fraction=.035);fig.tight_layout();fig.savefig(figdir/'correlation_heatmap.png',dpi=150);plt.close(fig)
    options=''.join(f'<option value="{n}">{n}</option>' for n in names)
    links=''.join(f'<li><a href="figures/{n}.png">{n}</a></li>' for n in names)
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Sensor 特征与真实 VB</title>
    <style>body{{font-family:system-ui;margin:24px;max-width:1500px}}img{{max-width:100%}}select{{font-size:18px;padding:8px}}p{{line-height:1.7}}</style>
    <h1>Sensor 特征随时间变化 vs 真实 VB</h1><p>蓝色为当前 run 的原始统计特征，橙色为真实 VB。每页包含16个case；标记extra的6/15不属于此前14-case实验。左侧特征纵轴各case独立，右侧VB纵轴统一，不应从线条视觉重合判断相关性。没有VB标签的位置不补标签。</p>
    <p>相关性只使用同一run的有效标签。至少4个标签才计算；case16只有3个、case6只有1个，因此标为NA。完整工况和样本数见<a href="case_metadata.csv">case_metadata.csv</a>。</p>
    <h2>逐特征查看</h2><select id="feature">{options}</select><p><a id="full" href="figures/{names[0]}.png">打开原尺寸图</a></p><img id="plot" src="figures/{names[0]}.png">
    <h2>真实 VB</h2><a href="figures/vb_trajectories.png"><img src="figures/vb_trajectories.png"></a>
    <h2>同一case内的特征—VB相关性</h2><a href="figures/correlation_heatmap.png"><img src="figures/correlation_heatmap.png"></a>
    <p>红色正相关，蓝色负相关；负相关也可能提供有效信息。灰色不是零相关。共同随时间变化不等于传感器提供了时间之外的信息。</p>
    <p><a href="README.md">方法和结果解读</a> · <a href="feature_summary.csv">54特征统计</a> · <a href="per_case_correlations.csv">逐case相关性</a> · <a href="raw_features_and_vb.csv">原始特征表</a></p>
    <details><summary>全部54张原尺寸图</summary><ul>{links}</ul></details>
    <script>document.getElementById('feature').onchange=function(){{const p='figures/'+this.value+'.png';document.getElementById('plot').src=p;document.getElementById('full').href=p;}};</script></html>'''
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    write_json(ROOT/'audit.json',dict(data_signature=signature,records=len(frame),labeled=int(frame.VB.notna().sum()),
        primary_cases=PRIMARY,primary_labeled=len(labeled),features=len(names),png_files=len(list(figdir.glob('*.png'))),
        excluded_raw_record='case2/run1 abs>1e6, same QC as training',correlation_min_n=4,partial_min_n=5))
    assert len(names)==54 and len(list(figdir.glob('*.png')))==57
    print(summary[['feature','median_abs_spearman','median_spearman','cases_ge_05','cases_le_minus05','median_abs_time_adjusted']].to_string(index=False))


if __name__=='__main__':
    main()
