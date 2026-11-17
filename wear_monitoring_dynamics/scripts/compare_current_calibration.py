"""Compare absolute feature values at matched VB; exploratory, no case centering."""

from pathlib import Path

from itertools import combinations

import json

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt



BASE=Path(__file__).resolve().parents[1]

SOURCE=BASE/'outputs/current_features_20261013'

OUT=BASE/'outputs/current_calibration_consistency_20261013'

PAIRS=[(1,9),(2,12),(3,11),(4,10),(5,16),(7,13),(8,14)]

ACTIVE=sorted(set(sum(([a,b] for a,b in PAIRS),[])))



def compare(a,b):

    """Two-column finite VB/value arrays. Require two measured points in overlap."""

    def curve(v):

        v=np.asarray(v,float);v=v[np.isfinite(v).all(axis=1)]

        return pd.DataFrame(v,columns=['x','y']).groupby('x').y.mean().sort_index()

    a,b=curve(a),curve(b)

    if min(len(a),len(b))<2:return None

    lo=max(a.index.min(),b.index.min());hi=min(a.index.max(),b.index.max())

    na=int(((a.index>=lo)&(a.index<=hi)).sum());nb=int(((b.index>=lo)&(b.index<=hi)).sum())

    if hi<=lo or min(na,nb)<2:return None

    # Include knots and zero crossings for exact integral of absolute piecewise-linear difference.

    x=np.unique(np.r_[lo,hi,a.index[(a.index>lo)&(a.index<hi)],b.index[(b.index>lo)&(b.index<hi)]])

    d=np.interp(x,a.index,a.values)-np.interp(x,b.index,b.values)

    crossed=np.flatnonzero(d[:-1]*d[1:]<0)

    roots=[x[i]-d[i]*(x[i+1]-x[i])/(d[i+1]-d[i]) for i in crossed]

    x=np.unique(np.r_[x,roots]);ya=np.interp(x,a.index,a.values);yb=np.interp(x,b.index,b.values);d=ya-yb

    gap=np.trapezoid(abs(d),x)/(hi-lo);bias=np.trapezoid(d,x)/(hi-lo)

    swing=(np.ptp(ya)+np.ptp(yb))/2

    return dict(raw_gap=gap,signed_bias=bias,bias_fraction=abs(bias)/gap if gap>1e-12 else 0,

                overlap_swing=swing,overlap_low=lo,overlap_high=hi,n_a=na,n_b=nb,

                sparse=min(na,nb)<4)



def main():

    OUT.mkdir(exist_ok=True)

    source=pd.read_csv(SOURCE/'candidate_features.csv')

    data=source[source.case.isin(ACTIVE)&source.VB.notna()].copy()

    features=list(source.columns[7:]);old=pd.read_csv(SOURCE/'consistency_summary.csv').set_index('feature')

    rows=[];summaries=[]

    for feature in features:

        # Equal-case RMS of within-case P90-P10 ranges. A shared feature scale, never case normalization.

        ranges=data.groupby('case')[feature].apply(lambda s:s.quantile(.9)-s.quantile(.1))

        scale=float(np.sqrt(np.nanmean(ranges**2)))

        fr=[]

        for a,b in combinations(ACTIVE,2):

            result=compare(data.loc[data.case==a,['VB',feature]].values,data.loc[data.case==b,['VB',feature]].values)

            if result is None:continue

            result.update(feature=feature,pair=f'{a}/{b}',same_condition=(a,b) in PAIRS,scale=scale)

            result['scaled_gap']=result['raw_gap']/scale if scale>1e-12 else np.nan

            # Small overlap variation can make agreement uninformative. Explicit heuristic flag, not a test.

            result['low_variation']=result['overlap_swing']<=.05*scale+1e-12

            result['gap_over_overlap_swing']=result['raw_gap']/result['overlap_swing'] if result['overlap_swing']>1e-12 else np.nan

            fr.append(result);rows.append(result)

        p=pd.DataFrame(fr);s=p[p.same_condition];inform=s[~s.low_variation]

        summaries.append(dict(feature=feature,coverage=data[feature].notna().mean(),scale=scale,median_case_range=ranges.median(),

            gap_over_median_case_range=s.raw_gap.median()/ranges.median() if ranges.median()>1e-12 else np.nan,

            pairs=len(s),informative_pairs=len(inform),sparse_pairs=int(s['sparse'].sum()),low_variation_pairs=int(s.low_variation.sum()),

            median_gap=s.scaled_gap.median(),mean_gap=s.scaled_gap.mean(),worst_gap=s.scaled_gap.max(),

            informative_median_gap=inform.scaled_gap.median(),median_raw_gap=s.raw_gap.median(),

            median_gap_over_overlap_swing=inform.gap_over_overlap_swing.median(),

            median_bias_fraction=s.bias_fraction.median(),all_conditions_pairs=len(p),all_conditions_median_gap=p.scaled_gap.median(),

            direction_cases=old.loc[feature,'direction_cases'],defined_cases=old.loc[feature,'defined_cases'],median_abs_rho=old.loc[feature,'median_abs_rho']))

    details=pd.DataFrame(rows);summary=pd.DataFrame(summaries).sort_values('median_gap').reset_index(drop=True)

    summary.insert(0,'gap_rank',np.arange(1,len(summary)+1))

    summary.to_csv(OUT/'all_feature_comparison.csv',index=False);details.to_csv(OUT/'pair_details.csv',index=False)

    for _,r in summary.iterrows():

        feature=r.feature;fig,axes=plt.subplots(2,4,figsize=(16,7),sharey=True)

        for ax,(a,b) in zip(axes.flat,PAIRS):

            for case in (a,b):

                d=data[data.case==case].sort_values('VB');ax.plot(d.VB,d[feature],'.-',label=f'case {case}',ms=5,lw=1)

            p=details[(details.feature==feature)&(details.pair==f'{a}/{b}')]

            if len(p):

                q=p.iloc[0];ax.axvspan(q.overlap_low,q.overlap_high,color='green',alpha=.07)

                title=f'{a}/{b}: gap/scale={q.scaled_gap:.2f}'

                if q.low_variation:title+=' [flat]'

                if q['sparse']:title+=' [sparse]'

            else:title=f'{a}/{b}: insufficient overlap'

            ax.set_title(title,fontsize=10);ax.set_xlabel('VB (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8)

        axes.flat[-1].axis('off');axes.flat[-1].text(0,1,f'Median gap / scale: {r.median_gap:.3f}\nWorst pair: {r.worst_gap:.3f}\nCoverage: {r.coverage:.0%}\nEvaluable pairs: {r.pairs}/7\nLow-variation pairs: {r.low_variation_pairs}\n\nShared raw feature axis.\nGreen: compared VB overlap.\nLines: descriptive interpolation.\nSparse: <4 overlap points / case.',va='top',transform=axes.flat[-1].transAxes)

        fig.suptitle(feature+' | same VB, absolute feature values',fontsize=14);fig.tight_layout();fig.savefig(OUT/(feature+'.png'),dpi=115);plt.close(fig)

    matrix=details[details.same_condition].pivot(index='feature',columns='pair',values='scaled_gap').reindex(summary.feature)

    fig,ax=plt.subplots(figsize=(10,19));im=ax.imshow(matrix.values,aspect='auto',vmin=0,vmax=2,cmap='magma_r')

    ax.set_yticks(range(len(matrix)),matrix.index,fontsize=7);ax.set_xticks(range(len(matrix.columns)),matrix.columns)

    ax.set_title('Same-VB absolute disagreement / within-case P90-P10 scale\nLower is closer; saturated/constant and sparse features need flags')

    fig.colorbar(im,ax=ax,label='Gap / scale (colors capped at 2)');fig.tight_layout();fig.savefig(OUT/'all_features_heatmap.png',dpi=130);plt.close(fig)

    tr=[]

    for _,r in summary.iterrows():

        tr.append(f'<tr><td>{r.gap_rank}</td><td><a href="{r.feature}.png">{r.feature}</a></td><td>{r.median_gap:.3f}</td><td>{r.worst_gap:.3f}</td><td>{r.informative_median_gap:.3f}</td><td>{r.median_gap_over_overlap_swing:.3f}</td><td>{r.coverage:.0%}</td><td>{r.pairs}/7</td><td>{r.low_variation_pairs}</td><td>{r.sparse_pairs}</td><td>{r.all_conditions_median_gap:.3f}</td><td>{r.direction_cases}/{r.defined_cases}</td></tr>')

    page='''<!doctype html><meta charset="utf-8"><title>同VB特征一致性：全部61特征</title>

<style>body{font:15px system-ui;margin:30px;line-height:1.6}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border-bottom:1px solid #ddd;text-align:right}th{position:sticky;top:0;background:#e8eff5;cursor:pointer}td:nth-child(2){text-align:left}input{padding:8px;width:400px}p{max-width:1100px}</style>

<h1>相同磨损 VB，特征数值是否相同？</h1>

<p>之前的“13个case正相关”只描述趋势方向，不等于数值标定一致。本页比较全部61个电流候选特征，主比较为7对同工况case。点击特征名可看7对原始数值曲线；点击表头排序。</p>

<p><b>主指标：同VB绝对差 / 特征变化尺度，越小越一致。</b>在两case都有观测覆盖的VB区间内，线性插值后积分平均绝对差；不外推、不减去case初值。尺度为14个case各自P90−P10的均方根，每个特征共用一个尺度。0.20表示典型差异约为此尺度的20%，不是20%的磨损预测误差。不同特征量纲不同，不能直接比较原始差值。</p>

<p><b>差异/重叠段变化</b>：每对同VB平均绝对差除以重叠VB段内两曲线的平均最大最小跨度，再取各对中位数；排除低变化对。它只利用实际比较区间，更能识别“大部分case变化很小、少数case变化很大”造成的虚低排名。仍受观测噪声和VB跨度影响。CSV另附以各case变化范围中位数为尺度的敏感性结果。</p><p>排序只表示数值接近程度，<b>不是特征优劣总排名</b>。低变化对：重叠区间内两曲线平均跨度不超过该尺度5%，属于提示阈值；双方均零或均饱和也可能得到很小的差异。稀疏对：任一case在重叠区间少于4个不同VB点。有效比较至少需要每边2点。覆盖率分母为研究池137条有VB记录。case6、15不属于此前研究池，未混入统计。跨工况列纳入14个case所有可比较组合，数值同时包含工况差异。</p>

<p><a href="all_feature_comparison.csv">全部结果CSV</a> · <a href="pair_details.csv">每对明细（含原始差值、偏移、VB范围）</a> · <a href="all_features_heatmap.png">全部特征热图</a> · <a href="RESULTS.html">方法与结论</a></p>

<input id="q" placeholder="输入 amplitude / p90 / width / saturation 等特征名筛选" oninput="document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.innerText.toLowerCase().includes(this.value.toLowerCase()))">

<table><thead><tr>'''+''.join('<th>'+s+'</th>' for s in ['差异序号','特征（点击看图）','同工况中位差异↓','最差一对↓','排除低变化对后↓','差异/重叠段变化↓','覆盖率','可比较对','低变化对','稀疏对','全部工况中位差异↓','多数趋势方向case数'])+'''</tr></thead><tbody>'''+''.join(tr)+'''</tbody></table><script>document.querySelectorAll('th').forEach((h,i)=>h.onclick=()=>{let b=document.querySelector('tbody'),rs=[...b.rows];h.asc=!h.asc;rs.sort((a,c)=>{let x=a.cells[i].innerText,y=c.cells[i].innerText;let z=i===1?x.localeCompare(y):parseFloat(x)-parseFloat(y);return h.asc?z:-z});rs.forEach(r=>b.append(r))})</script>'''

    (OUT/'index.html').write_text(page,encoding='utf-8')

    (OUT/'protocol.json').write_text(json.dumps(dict(features=len(features),labeled_records=len(data),cases=ACTIVE,pairs=PAIRS,scale='RMS across cases of within-case P90-P10; common per feature',min_overlap_points=2,low_variation_threshold=.05,no_extrapolation=True,no_case_centering=True,exploratory=True),indent=2),encoding='utf-8')

    print(summary[['feature','median_gap','worst_gap','informative_median_gap','coverage','low_variation_pairs']].to_string(index=False))



if __name__=='__main__':main()

