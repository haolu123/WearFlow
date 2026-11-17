"""Independent post-training audit and oracle/curve diagnostics; never trains."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..config import file_hash,write_json,project_path
from ..training.single_shot_adaptation import load_parent
from ..data.datasets import pair_tensors,observations
from ..data.preprocessing import fit_scalers
from .run_psd_sweep import rank
from .report_current12_fusion import metrics,md

ROOT=Path('wear_monitoring_dynamics/outputs/current12_single_joint_20261014')

def main():
    start=time.perf_counter();torch.set_num_threads(1);m=pd.read_csv(ROOT/'study_manifest.csv');assert len(m)==120
    selection=pd.read_csv(ROOT/'inner_selection.csv');pred=pd.read_csv(ROOT/'predictions.csv.gz');audits=[];oracle=[];curves=[];seen=set()
    for r in m.itertuples():
        d=Path(r.run_dir);identity=json.loads((d/'identity.json').read_text());task=identity['task'];met=json.loads((d/'metrics.json').read_text());grad=json.loads((d/'gradient_audit.json').read_text())
        assert identity['execution_hash']==file_hash('wear_monitoring_dynamics/training/single_shot_adaptation.py')
        assert identity['sensor_hash']==file_hash(Path(task['sensor_dir'])/'last.pt') and identity['ode_hash']==file_hash(Path(task['ode_dir'])/'last.pt')
        assert met['epochs']==500 and met['reload_verified'] and grad['no_future_sensor'] and grad['future_to_monitor_norm']>0
        assert (met['ode_max_change']>0 and grad['future_to_ode_norm']>0) if r.mode=='joint' else (met['ode_max_change']==0 and grad['future_to_ode_norm']==0)
        model,batch,t,parent=load_parent(task['sensor_dir'],task['ode_dir'],'cpu');sp=parent['split'];sc=parent['scalers'];cfg=parent['config']
        assert sp==json.loads((d/'split.json').read_text())
        key=(r.fold,tuple(sp['train']))
        if key not in seen:
            assert json.loads(json.dumps(fit_scalers(t,sp['train'],cfg['data']['sensor_preprocessing'])))==json.loads((d/'scalers.json').read_text())
            assert cfg['data']['sensor_quality']['rules']['train_cases']==sp['train']
            score,_=rank(t,sp['train']);names=cfg['data']['feature_archive']['names'];ordered=sorted([j for j,n in enumerate(names) if n.startswith(('smcAC_','smcDC_'))],key=lambda j:(-score[j],names[j]));assert cfg['monitor']['selected_indices']==ordered[:12]
            seen.add(key)
        if r.role=='inner':assert r.holdout not in sp['train']
        cases=sp['validation'] or sp['test'];pairs=pair_tensors(t,cases,[1,2,3],'cpu');idx=observations(t,cases)
        state=torch.load(d/'last.pt',map_location='cpu',weights_only=True);model.load_state_dict(state['model']);model.eval()
        with torch.no_grad():
            now=model.estimate(batch,idx).numpy().ravel()*sc['wear'][1]+sc['wear'][0]
            future=model.forecast(batch,pairs).numpy().ravel()*sc['wear'][1]+sc['wear'][0]
        q=pred[(pred.fold==r.fold)&(pred.role==r.role)&(pred.seed==r.seed)&(pred.method==r.name)]
        if r.role=='inner':q=q[q.holdout==r.holdout]
        for typ,values in [('monitoring',now),('forecasting',future)]:np.testing.assert_allclose(values,q[q.task==typ].predicted_vb_mm,rtol=1e-5,atol=2e-6)
        actual_change=max(float((state['model']['dynamics.'+k]-v).abs().max()) for k,v in torch.load(Path(task['ode_dir'])/'last.pt',map_location='cpu',weights_only=True)['model'].items())
        assert abs(actual_change-met['ode_max_change'])<1e-7
        if r.role=='final':
            h=pd.read_csv(d/'history.csv');curves.append(h.assign(method=r.name,fold=r.fold,seed=r.seed))
            for label in ([r.name,'independent_500'] if r.name=='continue_current' else [r.name]):
                if label=='independent_500':model,_,_,_=load_parent(task['sensor_dir'],task['ode_dir'],'cpu')
                raw=pairs.numpy()
                with torch.no_grad():y=model.dynamics.propagate(batch['target'][pairs[:,0]],batch['condition'][pairs[:,0]],batch['time'][pairs[:,1]]-batch['time'][pairs[:,0]]).numpy().ravel()*sc['wear'][1]+sc['wear'][0]
                oracle.append(pd.DataFrame(dict(case=t['case'][raw[:,0]],origin_run=t['run'][raw[:,0]],target_run=t['run'][raw[:,1]],horizon=raw[:,2],target_vb_mm=t['VB'][raw[:,1]],predicted_vb_mm=y)).assign(method=label,fold=r.fold,seed=r.seed))
        audits.append(dict(id=r.id,reload_error=float(max(abs(now-q[q.task=='monitoring'].predicted_vb_mm))),ode_change=actual_change))
    p=pred[(pred.role=='final')&(pred.task=='monitoring')]
    for _,g in p.groupby(['method','seed']):assert len(g)==138 and len(g[['case','origin_run']].drop_duplicates())==138
    # Independently reconstruct lambda selection solely from inner current MAE.
    for r in selection[selection.selected].itertuples():
        q=pred[(pred.role=='inner')&(pred.fold==r.fold)&(pred.task=='monitoring')&pred.method.str.startswith(r.family)]
        scores=q.assign(ae=abs(q.predicted_vb_mm-q.target_vb_mm)).groupby(['method','case']).ae.mean().groupby('method').mean();assert scores.sort_index().idxmin()==r.method
    oo=pd.concat(oracle);extra=[]
    for r in selection[selection.selected].itertuples():extra.append(oo[(oo.fold==r.fold)&(oo.method==r.method)].assign(method='selected_'+r.family))
    oo=pd.concat([oo]+extra);oo.to_csv(ROOT/'oracle_predictions.csv.gz',index=False);per,s=metrics(oo,['method','horizon']);s.to_csv(ROOT/'oracle_summary.csv',index=False);per.to_csv(ROOT/'oracle_case_metrics.csv',index=False)
    cc=pd.concat(curves);cc.to_csv(ROOT/'epoch_curves.csv.gz',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for col,subset in enumerate(['overall','fold_a','fold_b']):
        q=cc if subset=='overall' else cc[cc.fold==subset]
        for row,task in enumerate(['monitoring','forecasting']):
            ax=axes[row,col]
            for method,g in q.groupby('method'):
                mean=g.groupby('epoch')[f'test_diagnostic_{task}_case_macro_mae'].mean();ax.plot(mean.index,mean,label=method)
            ax.set(title=f'{subset} / {task}',xlabel='Additional fine-tuning epoch',ylabel='Outer-test case MAE (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Diagnostic curves only; fixed final epoch 500');fig.tight_layout();fig.savefig(ROOT/'epoch_curves.png',dpi=150);plt.close(fig)
    selected=['independent_500','continue_current','selected_frozen','selected_joint'];case_tables=[]
    for task in ['monitoring','forecasting']:
        a=pd.read_csv(ROOT/f'{task}_case_metrics.csv');keys=['fold','case']+(['horizon'] if task=='forecasting' else []);case=a[a.method.isin(selected)].groupby(keys+['method']).mae.mean().unstack('method').reset_index();case.to_csv(ROOT/f'{task}_selected_cases.csv',index=False)
        summary=pd.read_csv(ROOT/f'{task}_summary.csv');summary[summary.method.isin(selected)].to_csv(ROOT/f'{task}_selected_summary.csv',index=False)
    pd.DataFrame(audits).to_csv(ROOT/'independent_run_audit.csv',index=False);write_json(ROOT/'independent_audit.json',dict(status='passed',runs=len(audits),training_splits=len(seen),current_targets_per_seed=138,seconds=time.perf_counter()-start,oracle_is_true_initial_diagnostic=True))
    print(s[(s.subset=='overall')&s.method.isin(selected)].to_string(index=False))

if __name__=='__main__':main()

