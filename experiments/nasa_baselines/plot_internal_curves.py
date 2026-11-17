"""Recompute case-macro outer curves from raw predictions, without epoch selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent/'internal/results'
status=json.loads((R/'status.json').read_text())
assert status['status']=='complete' and status['completed']==96
rows=[]
for folder in sorted(R.glob('fold_*/final/seed_*/*')):
    if not (folder/'completed.json').exists():continue
    a=np.load(folder/'epoch_predictions.npz')
    ck=torch.load(folder/'model.pt',weights_only=True)
    mu,sigma=ck['scalers']['wear'];arm=folder.name;seed=int(folder.parent.name.split('_')[-1]);fold=folder.parents[2].name
    tasks={'current':(np.flatnonzero(a['test_mask']),None)}
    for h in [1,2,3]:
        pairs=a['pairs'];k=np.flatnonzero((pairs[:,2]==h)&a['test_mask'][pairs[:,0]])
        tasks['h'+str(h)]=(pairs[k,1],k)
    for task,(targets,k) in tasks.items():
        if k is None:pred=a['current_z'][:,targets]*sigma+mu
        elif arm=='current_only':pred=a['current_z'][:,a['pairs'][k,0]]*sigma+mu
        else:pred=a['future_z'][:,k]*sigma+mu
        ae=abs(pred-a['target_mm'][targets][None,:])
        for case in np.unique(a['case'][targets]):
            vals=ae[:,a['case'][targets]==case].mean(axis=1)
            rows.extend(dict(arm=arm,task=task,fold=fold,seed=seed,case=int(case),epoch=e,mae_mm=float(v)) for e,v in enumerate(vals))
d=pd.DataFrame(rows)
assert len(d[d.task=='current'].drop_duplicates(['arm','fold','seed']))==40
c=d.groupby(['arm','task','fold','case','epoch']).mae_mm.mean().groupby(['arm','task','epoch']).mean().reset_index()
c.to_csv(R/'outer_case_macro_epoch_curves.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,task in zip(axes.ravel(),['current','h1','h2','h3']):
    for arm,g in c[c.task==task].groupby('arm'):
        label=arm+(' (persistence)' if arm=='current_only' and task!='current' else '')
        ax.plot(g.epoch,g.mae_mm,label=label)
    ax.set(title=task,xlabel='Epoch (fixed final=100)',ylabel='Case-macro test MAE (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(R/'outer_epoch_curves.png',dpi=160)
print('Saved curves from 40 final runs; no epoch selection.')
