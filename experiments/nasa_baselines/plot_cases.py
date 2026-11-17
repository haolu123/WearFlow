"""Descriptive per-case current-VB plots; no model or hyperparameter selection."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
df=pd.read_csv(R/'summary/current_predictions.csv')
rows=[]
for fold in ['fold_a','fold_b']:
    a=np.load(R/'data'/fold/'final/arrays.npz')
    rows.extend(dict(case=int(c),run=int(r),time=float(t)) for c,r,t in zip(a['case'],a['run'],a['time']))
times=pd.DataFrame(rows).drop_duplicates(['case','run'])
df=df.merge(times,on=['case','run'],suffixes=('_export',''),validate='many_to_one')
caseids=sorted(df.case.unique())
fig,axes=plt.subplots(4,4,figsize=(17,13),layout='constrained')
methods=['CT100','sensor12_100','gpr','svr','Zhang_PINN_top12','sun_pigpr_causal_nasa']
labels=['CT MLP','Sensor MLP','GPR','SVR','Zhang PINN','PI-GPR causal']
for ax,case in zip(axes.ravel(),caseids):
    g=df[df.case==case]
    truth=g[['run','time','target_mm']].drop_duplicates().sort_values('time')
    ax.plot(truth.time,truth.target_mm,'ko-',label='Measured VB',linewidth=2,markersize=3)
    for method,label in zip(methods,labels):
        p=g[g.method==method].groupby(['run','time']).pred_mm.mean().reset_index().sort_values('time')
        ax.plot(p.time,p.pred_mm,label=label,linewidth=1.2)
    ax.set(title=f'Case {case}',xlabel='Dataset time',ylabel='VB (mm)');ax.grid(alpha=.2)
for ax in axes.ravel()[len(caseids):]:ax.set_visible(False)
handles,labels=axes.ravel()[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside upper center',ncol=7)
fig.savefig(R/'summary/current_case_trajectories.png',dpi=150)
print('Saved 14-case trajectories; neural-network predictions averaged over seeds.')
