"""Descriptive temporal monotonicity, with one shared direction per feature."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path, write_json
from ..data.splits import POOLS

ROOT=project_path('wear_monitoring_dynamics/outputs/sensor_feature_inspection_20260927')


def temporal_stats(g, feature, direction):
    g=g.sort_values(['time','run'])
    x=g[feature].to_numpy();t=g.time.to_numpy()
    d=np.diff(x)[np.diff(t)>0]
    eps=1e-12*max(1.,float(np.max(np.abs(x))))
    d[np.abs(d)<=eps]=0.
    signed=direction*d
    variation=float(np.abs(d).sum())
    return dict(pairs=len(d),violations=int((signed<0).sum()),
                violation_rate=float((signed<0).mean()) if len(d) else np.nan,
                reverse_variation_fraction=float(np.maximum(-signed,0).sum()/variation) if variation else np.nan,
                flat_pairs=int((d==0).sum()),
                rho=float(spearmanr(t,x).statistic) if len(x)>=3 and np.ptp(x)>eps and np.ptp(t)>0 else np.nan)


def main():
    frame=pd.read_csv(ROOT/'raw_features_and_vb.csv')
    frame=frame[frame.case.isin(POOLS['fold_a']+POOLS['fold_b'])]
    features=[c for c in frame if '__' in c]
    rankings=[];details=[]
    for view in ('all_records','labeled_only','without_case12_run1_sensitivity'):
        f=frame.copy()
        if view=='labeled_only':f=f[f.VB.notna()]
        if view=='without_case12_run1_sensitivity':f=f[~((f.case==12)&(f.run==1))]
        for feature in features:
            raw=[temporal_stats(g,feature,1)['rho'] for _,g in f.groupby('case')]
            direction=1 if np.nanmedian(raw)>=0 else -1
            rows=[dict(view=view,feature=feature,case=int(c),direction=direction,**temporal_stats(g,feature,direction)) for c,g in f.groupby('case')]
            details.extend(rows);q=pd.DataFrame(rows)
            rankings.append(dict(view=view,feature=feature,direction='increasing' if direction==1 else 'decreasing',
                case_macro_violation_rate=q.violation_rate.mean(),pooled_violation_rate=q.violations.sum()/q.pairs.sum(),
                violations=q.violations.sum(),pairs=q.pairs.sum(),median_abs_time_rho=q.rho.abs().median(),
                median_signed_time_rho=direction*q.rho.median(),direction_agree_cases=int((direction*q.rho>0).sum()),
                direction_valid_cases=int(q.rho.notna().sum()),strictly_no_reversal_cases=int(((q.violations==0)&(q.pairs>0)).sum()),
                constant_cases=int(q.reverse_variation_fraction.isna().sum()),
                mean_reverse_variation_fraction=q.reverse_variation_fraction.mean()))
    rank=pd.DataFrame(rankings).sort_values(['view','case_macro_violation_rate','mean_reverse_variation_fraction','feature'])
    rank['rank']=rank.groupby('view').cumcount()+1
    rank.to_csv(ROOT/'temporal_monotonicity_ranking.csv',index=False)
    pd.DataFrame(details).to_csv(ROOT/'temporal_monotonicity_per_case.csv',index=False)
    top=rank[rank.view=='all_records'].head(15).iloc[::-1]
    fig,ax=plt.subplots(figsize=(11,8))
    ax.barh(top.feature,top.case_macro_violation_rate*100,color=['#2874a6' if d=='increasing' else '#b66a30' for d in top.direction])
    for i,r in enumerate(top.itertuples()):
        ax.text(r.case_macro_violation_rate*100+.12,i,f'{r.case_macro_violation_rate:.1%} | {r.direction_agree_cases}/{r.direction_valid_cases} cases same direction',va='center',fontsize=9)
    ax.set_xlim(0,float(top.case_macro_violation_rate.max()*100+14))
    ax.set_xlabel('Mean within-case adjacent reversal rate (%) — lower is better')
    ax.set_title('Top 15 sensor features by temporal monotonicity\n14 study cases, all retained runs; blue=increasing, brown=decreasing')
    fig.tight_layout();fig.savefig(ROOT/'figures'/'temporal_monotonicity_top15.png',dpi=150);plt.close(fig)
    print(rank[rank.view=='all_records'].head(18).to_string(index=False))
    print('\nSENSITIVITY TOP 10\n',rank[rank.view=='without_case12_run1_sensitivity'].head(10).to_string(index=False))
    assert len(rank)==162 and len(details)==2268
    toy=pd.DataFrame(dict(time=[0,1,2,3],run=[1,2,3,4],f=[1,2,2,4]))
    assert temporal_stats(toy,'f',1)['violations']==0
    assert temporal_stats(toy,'f',-1)['violations']==2
    write_json(ROOT/'monotonicity_audit.json',dict(feature_count=54,views=3,case_count=14,per_case_rows=len(details),
        rule='One common direction per feature from median case Spearman(time,feature); macro adjacent reversal rate ascending.',
        tolerance='1e-12 * max(1, max absolute feature within case); plateaus permitted; positive time gaps only',
        training_data_changed=False,sanity_checks_passed=True))


if __name__=='__main__':
    main()
