"""Report fixed epoch100 without selecting methods on external test."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).parent/'results'
def main():
    rows=[]; audits=[]
    for folder in sorted(ROOT.glob('fold_*/*/seed_*/*')):
        if not (folder/'completed.json').exists():continue
        a=dict(np.load(folder/'epoch_predictions.npz')); meta=json.loads((folder/'identity.json').read_text()); done=json.loads((folder/'completed.json').read_text())
        assert a['current_z'].shape==(101,len(a['case'])) and a['future_z'].shape[0]==101 and a['oracle_z'].shape==a['future_z'].shape
        import torch
        ck=torch.load(folder/'model.pt',weights_only=True);mu,sigma=ck['scalers']['wear']
        assert len(pd.read_csv(folder/'history.csv'))==101
        audits.append({'run':str(folder.relative_to(ROOT)),'initial_hash':done['initial_hash'],'seconds':done['seconds'],'epochs':100,'reload':done['reload_exact']})
        if folder.parents[1].name!='final':continue
        fold=folder.parents[2].name;seed=int(folder.parent.name.split('_')[-1]);arm=folder.name
        for idx in np.flatnonzero(a['test_mask']):
            rows.append(dict(arm=arm,fold=fold,seed=seed,task='current',case=int(a['case'][idx]),run=int(a['run'][idx]),origin_run=int(a['run'][idx]),target_run=int(a['run'][idx]),horizon=0,dt=0.,target_mm=a['target_mm'][idx],pred_mm=a['current_z'][-1,idx]*sigma+mu))
        for k,(origin,target,h) in enumerate(a['pairs']):
            if not a['test_mask'][origin]:continue
            assert a['case'][origin]==a['case'][target] and a['time'][target]>a['time'][origin]
            for name,pred in [('h'+str(h),a['future_z'][-1,k]),('oracle_h'+str(h),a['oracle_z'][-1,k]),('persistence_h'+str(h),a['current_z'][-1,origin])]:
                rows.append(dict(arm=arm,fold=fold,seed=seed,task=name,case=int(a['case'][target]),run=int(a['run'][target]),origin_run=int(a['run'][origin]),target_run=int(a['run'][target]),horizon=int(h),dt=float(a['time'][target]-a['time'][origin]),target_mm=a['target_mm'][target],pred_mm=pred*sigma+mu))
    assert len(audits)==96, 'Full report requires 96 completed runs'
    assert len(list(ROOT.glob('fold_*/*/seed_*')))==24
    pd.DataFrame(audits).to_csv(ROOT/'run_audit.csv',index=False)
    d=pd.DataFrame(rows);d['ae']=(d.pred_mm-d.target_mm).abs();d.to_csv(ROOT/'final_predictions.csv.gz',index=False)
    pieces=[]
    for region,mask in [('overall',np.ones(len(d),bool)),('fold_a',d.fold=='fold_a'),('fold_b',d.fold=='fold_b'),('A_in',(d.fold=='fold_a')&(d.target_mm<=.74)),('A_out',(d.fold=='fold_a')&(d.target_mm>.74))]:
        x=d[mask].groupby(['arm','task','fold','case','seed']).ae.mean().reset_index(name='mae');x['region']=region;pieces.append(x)
    c=pd.concat(pieces);c.to_csv(ROOT/'case_seed_metrics.csv',index=False)
    cm=c.groupby(['arm','task','region','fold','case']).mae.mean().reset_index();cm.to_csv(ROOT/'case_mean_metrics.csv',index=False)
    summary=cm.groupby(['arm','task','region']).mae.agg(['mean','count']).reset_index();summary.to_csv(ROOT/'summary.csv',index=False)
    rng=np.random.default_rng(20261103);boot=[]
    def resample(diff,folds):
        strata=[np.flatnonzero(np.asarray(folds)==f) for f in sorted(set(folds))]
        indices=np.concatenate([rng.choice(ix,size=(10000,len(ix)),replace=True) for ix in strata],axis=1)
        return diff[indices].mean(1)
    for task in ('current','h1','h2','h3'):
        x=cm[(cm.task==task)&(cm.region=='overall')].pivot(index=['fold','case'],columns='arm',values='mae')
        for comp in ('current_only','local_only','both'):
            diff=(x.integral_only-x[comp]).to_numpy(); means=resample(diff,x.index.get_level_values('fold'))
            boot.append({'task':task,'contrast':'integral_only minus '+comp+(' (random untrained rate diagnostic)' if comp=='current_only' and task!='current' else ''),'mean':diff.mean(),'low':np.quantile(means,.025),'high':np.quantile(means,.975),'cases':len(diff)})
    for h in (1,2,3):
        left=cm[(cm.arm=='integral_only')&(cm.task=='h'+str(h))&(cm.region=='overall')].set_index(['fold','case']).mae
        right=cm[(cm.arm=='current_only')&(cm.task=='persistence_h'+str(h))&(cm.region=='overall')].set_index(['fold','case']).mae
        paired=(left-right).dropna();diff=paired.to_numpy()
        means=resample(diff,paired.index.get_level_values('fold'))
        boot.append({'task':'h'+str(h),'contrast':'integral_only ODE minus current_only persistence (deploy comparison)','mean':diff.mean(),'low':np.quantile(means,.025),'high':np.quantile(means,.975),'cases':len(diff)})
    pd.DataFrame(boot).to_csv(ROOT/'case_paired_bootstrap.csv',index=False)
    hashes=pd.DataFrame(audits);hashes['group']=hashes.run.map(lambda s:str(Path(s).parent));assert hashes.groupby('group').initial_hash.nunique().max()==1
    for parent in sorted(ROOT.glob('fold_*/*/seed_*')):
        files=[parent/arm/'epoch_predictions.npz' for arm in ('current_only','local_only','integral_only','both')]
        assert all(p.exists() for p in files)
        packs=[dict(np.load(p)) for p in files]
        for pack in packs[1:]:
            for key in ('current_z','future_z','oracle_z'):assert np.array_equal(packs[0][key][0],pack[key][0])
            for key in ('pairs','train_pairs_used','train_observations','test_mask'):assert np.array_equal(packs[0][key],pack[key])
    for (arm,seed),g in d[d.task=='current'].groupby(['arm','seed']):assert len(g)==138
    save={'runs':len(audits),'epochs_per_run':100,'prediction_epochs':101,'epoch0_initial_hashes_match':True,'sum_training_evaluation_seconds':float(hashes.seconds.sum())}
    (ROOT/'report_audit.json').write_text(json.dumps(save,indent=2))
    print(summary[(summary.region=='overall')&summary.task.isin(['current','h1','h2','h3'])].to_string(index=False))
if __name__=='__main__':main()













