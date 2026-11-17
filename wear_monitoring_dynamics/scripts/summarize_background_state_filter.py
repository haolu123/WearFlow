"""Freeze inner-only filtering choices, then evaluate causal outer trajectories."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import write_json,project_path
from .run_sensor_mechanisms import context,OLD
from .run_background_state_filter import ROOT,KINDS,GRID,PositiveRate,dynamics_inputs,fit_observation,evaluate_case,filter_sequence
from .diagnose_sensor_learning import md

LABELS={'ct_baseline':'Condition + time MLP','current_only':'Sensor MLP','dynamics_prior':'Dynamics prior',
        'sensor_filter':'Sensor filter','background_filter':'Case-background filter','separated_background_filter':'Orthogonal-background filter'}


def main():
    torch.set_num_threads(1)
    mf=pd.read_csv(ROOT/'study_manifest.csv');assert len(mf)==24
    source=pd.read_csv(OLD/'reused_sensor_manifest.csv');contexts={r.run_dir:context(r) for r in source.itertuples()}
    curves=pd.concat([pd.read_csv(Path(r.run_dir)/'history.csv').assign(fold=r.fold,role=r.role,holdout=r.holdout,seed=r.seed) for r in mf.itertuples()])
    curves.to_csv(ROOT/'epoch_metrics.csv.gz',index=False)
    inner=curves[curves.role=='inner'].groupby(['fold','epoch']).validation_mae.mean().reset_index()
    epochs=inner.sort_values(['validation_mae','epoch']).groupby('fold').first().reset_index()
    epochs.to_csv(ROOT/'selected_epochs.csv',index=False)
    def selected_epoch(fold,policy):return 500 if policy=='fixed500' else int(epochs[epochs.fold==fold].epoch.iloc[0])
    scores=[];audits=[];coverage=[];maxreload=maxprefix=maxquad=maxdynamicprefix=0.
    for row in mf.itertuples():
        c=contexts[row.source_run];d=Path(row.run_dir);a=np.load(d/'epoch_predictions.npz');obs=dict(np.load(d/'observation.npz'))
        def condition(k):
            i=np.flatnonzero(c['table']['case']==k)[0]
            return tuple(float(c['table'][f][i]) for f in ['DOC','feed','material'])
        seen={condition(k) for k in c['split']['train']}
        for k in (c['split']['validation'] if row.role=='inner' else c['split']['outer_test_reserved']):
            coverage.append(dict(fold=row.fold,role=row.role,seed=row.seed,case=k,condition_seen=condition(k) in seen))
        ident=json.loads((d/'identity.json').read_text());complete=json.loads((d/'complete.json').read_text())
        assert complete['reload_verified'] and ident['train_indices']==c['train'].tolist()
        assert not set(obs['train_cases']).intersection(c['split']['validation']+c['split']['outer_test_reserved'])
        assert np.array_equal(obs['train_indices'],c['train']) and c['valid'][c['train']].all()
        assert a['predictions'].shape[0]==500 and np.isfinite(a['predictions']).all()
        changed=copy.copy(c);changed['table']=dict(c['table']);changed['table']['VB']=c['table']['VB'].copy()
        changed['table']['VB'][~np.isin(c['table']['case'],c['split']['train'])]=1234.
        refit=fit_observation(changed)
        for key in obs:np.testing.assert_array_equal(obs[key],refit[key])
        weights=np.load(d/'epoch_weights.npz')['weights']
        for policy in ['fixed500','inner_epoch']:
            epoch=selected_epoch(row.fold,policy);prior=a['predictions'][epoch-1]
            model=PositiveRate();torch.nn.utils.vector_to_parameters(torch.tensor(weights[epoch-1]),model.parameters())
            args=dynamics_inputs(c)
            with torch.no_grad():restored=model(*args).numpy()*c['scalers']['wear'][1]
            maxreload=max(maxreload,float(abs(restored-prior).max()));np.testing.assert_allclose(restored,prior,atol=2e-6)
            # Independent higher-order quadrature audit of saved dynamics.
            tau=c['table']['time']/c['scalers']['time_scale'];dt=args[3].numpy();x,w=np.polynomial.legendre.leggauss(8)
            nodes=tau[:,None]-dt[:,None]+dt[:,None]*(x[None,:]+1)/2
            with torch.no_grad():hi=model(args[0],torch.tensor(nodes,dtype=torch.float32),torch.tensor(w/2,dtype=torch.float32),args[3],args[4]).numpy()*c['scalers']['wear'][1]
            maxquad=max(maxquad,float(abs(hi-prior).max()))
            for k in c['split']['outer_test_reserved'] if row.role=='final' else c['split']['validation']:
                ids=np.flatnonzero(c['table']['case']==k);ids=ids[np.argsort(c['table']['run'][ids])];ids=ids[:max(1,len(ids)//2)]
                short=dict(table={f:c['table'][f][ids] for f in ['case','run','time']},ct=c['ct'][ids],scalers=c['scalers'])
                with torch.no_grad():prefix=model(*dynamics_inputs(short)).numpy()*c['scalers']['wear'][1]
                err=float(abs(prefix-prior[ids]).max());maxdynamicprefix=max(maxdynamicprefix,err)
                np.testing.assert_allclose(prefix,prior[ids],atol=2e-6)
            if row.role=='inner':
                for kind in KINDS:
                    for hyper in GRID:
                        ix,out=evaluate_case(c,prior,obs,kind,hyper,int(row.holdout));target=c['table']['VB'][ix];mask=np.isfinite(target)
                        error=out['wear'][mask]*c['scalers']['wear'][1]-target[mask]
                        scores.append(dict(fold=row.fold,holdout=int(row.holdout),policy=policy,epoch=epoch,kind=kind,**hyper,mae=float(abs(error).mean())))
        audits.append(dict(fold=row.fold,role=row.role,holdout=row.holdout,seed=row.seed,parameters=complete['parameters'],seconds=complete['seconds'],
            wear_sensitivity_rank=int(obs['wear_sensitivity_rank']),background_overlap=float(obs['background_wear_overlap'][0]),
            separated_overlap=float(obs['background_wear_overlap'][1]),observation_train_rmse=float(obs['observation_train_rmse'].mean())))
    score=pd.DataFrame(scores);score.to_csv(ROOT/'inner_filter_scores.csv',index=False)
    avg=score.groupby(['fold','policy','kind','name','q','r_scale']).mae.mean().reset_index()
    # Off wins exact ties; then smaller process noise and larger observation noise.
    avg['on']=avg.r_scale>0
    choices=avg.sort_values(['mae','on','q','r_scale'],ascending=[True,True,True,False]).groupby(['fold','policy','kind']).first().reset_index()
    choices.to_csv(ROOT/'selected_filters.csv',index=False);avg.to_csv(ROOT/'inner_filter_grid.csv',index=False)
    write_json(ROOT/'selection.json',dict(rule='mean seven inner held-out case MAEs; off then smaller q then larger R break exact ties',
        epochs=epochs.to_dict('records'),filters=choices.to_dict('records'),grid=GRID))
    print('Inner choices frozen before outer aggregation',flush=True)
    rows=[];traj=[]
    for row in mf[mf.role=='final'].itertuples():
        c=contexts[row.source_run];d=Path(row.run_dir);a=np.load(d/'epoch_predictions.npz');obs=dict(np.load(d/'observation.npz'))
        for policy in ['fixed500','inner_epoch']:
            epoch=selected_epoch(row.fold,policy);prior=a['predictions'][epoch-1];scale=c['scalers']['wear'][1]
            for case in c['split']['outer_test_reserved']:
                for kind in ['dynamics_prior']+KINDS:
                    hyper=GRID[0] if kind=='dynamics_prior' else choices[(choices.fold==row.fold)&(choices.policy==policy)&(choices.kind==kind)].iloc[0].to_dict()
                    ix,out=evaluate_case(c,prior,obs,kind,hyper,case)
                    # Actual final sequences: prefix truncation, future perturbation and label permutation.
                    cut=max(1,len(ix)//2);tau=c['table']['time'][ix]/c['scalers']['time_scale'];dt=np.diff(np.r_[0.,tau])
                    prefix=filter_sequence(prior[ix[:cut]]/scale,dt[:cut],c['sensor'][ix[:cut],:18].numpy(),c['ct'][ix[0],:4].numpy(),c['valid'][ix[:cut]],obs,kind,hyper['q'],hyper['r_scale'])
                    maxprefix=max(maxprefix,float(abs(prefix['wear']-out['wear'][:cut]).max()));np.testing.assert_array_equal(prefix['wear'],out['wear'][:cut])
                    altered=copy.copy(c);altered['table']=dict(c['table']);altered['table']['VB']=np.full_like(c['table']['VB'],999.)
                    _,without_labels=evaluate_case(altered,prior,obs,kind,hyper,case)
                    np.testing.assert_array_equal(out['wear'],without_labels['wear'])
                    # Future sensor perturbation cannot change prefix even when full sequence is passed.
                    future=c['sensor'][ix,:18].numpy().copy();future[cut:]=777.
                    perturbed=filter_sequence(prior[ix]/scale,dt,future,c['ct'][ix[0],:4].numpy(),c['valid'][ix],obs,kind,hyper['q'],hyper['r_scale'])
                    np.testing.assert_array_equal(perturbed['wear'][:cut],out['wear'][:cut])
                    for j,i in enumerate(ix):
                        target=c['table']['VB'][i];pred=out['wear'][j]*scale
                        record=dict(name=kind,policy=policy,epoch=epoch,fold=row.fold,seed=row.seed,case=case,run=int(c['table']['run'][i]),time=float(c['table']['time'][i]),
                            target=target,prediction=pred,ae=abs(pred-target),se=(pred-target)**2)
                        traj.append(dict(**record,valid=bool(c['valid'][i]),prior=float(prior[i]),sd_proxy=out['sd'][j]*scale,z1=out['background'][j,0],z2=out['background'][j,1],
                            innovation=out['innovation'][j],correction=out['correction'][j]*scale,filter_choice=hyper['name']))
                        if np.isfinite(target):rows.append(record)
    rec=pd.DataFrame(rows)
    old=pd.read_csv(project_path('wear_monitoring_dynamics/outputs/initial_context_20261012/test_predictions.csv'))
    old=old[old.name.isin(['ct_baseline','current_only'])]
    rec=pd.concat([rec,old[old.columns.intersection(rec.columns)]],ignore_index=True)
    ref=None
    for key,q in rec.groupby(['name','policy','seed']):
        assert len(q)==138 and not q.duplicated(['case','run']).any()
        labels=q.set_index(['case','run']).target.sort_index()
        if ref is None:ref=labels
        else:pd.testing.assert_series_equal(ref,labels)
    rec.to_csv(ROOT/'test_predictions.csv',index=False);traj=pd.DataFrame(traj);traj.to_csv(ROOT/'state_trajectories.csv',index=False)
    case=rec.groupby(['name','policy','fold','seed','case']).agg(mae=('ae','mean'),mse=('se','mean'),n=('ae','size')).reset_index()
    case.to_csv(ROOT/'case_metrics.csv',index=False)
    summary=case.groupby(['name','policy']).agg(mae=('mae','mean'),mse=('mse','mean')).reset_index()
    summary=summary.merge(case.groupby(['name','policy','fold']).mae.mean().unstack().reset_index(),on=['name','policy'])
    summary.to_csv(ROOT/'summary.csv',index=False)
    seed=case.groupby(['name','policy','seed']).mae.mean().reset_index();seed.to_csv(ROOT/'seed_metrics.csv',index=False)
    contrasts=[]
    for policy in ['fixed500','inner_epoch']:
        for ctrl,treatment in [('ct_baseline','dynamics_prior'),('dynamics_prior','sensor_filter'),('dynamics_prior','background_filter'),('sensor_filter','background_filter'),('sensor_filter','separated_background_filter'),('ct_baseline','background_filter')]:
            aa=case[(case.name==ctrl)&(case.policy==policy)].set_index(['seed','case']).mae
            bb=case[(case.name==treatment)&(case.policy==policy)].set_index(['seed','case']).mae
            diff=(bb-aa).rename('delta').reset_index()
            contrasts.append(dict(policy=policy,control=ctrl,treatment=treatment,delta_mae=diff.delta.mean(),improved_cases=int((diff.groupby('case').delta.mean()<-1e-10).sum()),
                improved_seeds=int((diff.groupby('seed').delta.mean()<-1e-10).sum())))
    contrast=pd.DataFrame(contrasts);contrast.to_csv(ROOT/'contrasts.csv',index=False)
    pd.DataFrame(audits).to_csv(ROOT/'run_audit.csv',index=False)
    pd.DataFrame(coverage).to_csv(ROOT/'condition_coverage.csv',index=False)
    write_json(ROOT/'audit.json',dict(passed=True,dynamics_runs=24,epochs_per_run=500,total_epochs=12000,inner_filter_sequences=len(score),
        outer_targets=138,checkpoint_reload_max_error_mm=maxreload,quadrature4_vs8_max_error_mm=maxquad,
        causal_prefix_max_error=maxprefix,dynamics_prefix_max_error_mm=maxdynamicprefix,heldout_labels_do_not_change_observation_fit=True,test_labels_do_not_change_filter=True,
        future_sensor_do_not_change_prefix=True,shared_training_mask=True,uncertainty_calibrated=False))
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,fold in zip(axs,['fold_a','fold_b']):
        q=curves[(curves.fold==fold)&(curves.role=='inner')].groupby('epoch')[['train_mae','validation_mae']].mean()
        q.plot(ax=ax);ax.set(title=fold+' dynamics',ylabel='Case-macro MAE (mm)');ax.grid(alpha=.2)
    fig.savefig(ROOT/'dynamics_learning.png',dpi=150);plt.close(fig)
    q=case[case.policy=='fixed500'].groupby(['case','name']).mae.mean().unstack()[list(LABELS)]
    q.to_csv(ROOT/'case_mae_readout.csv');fig,ax=plt.subplots(figsize=(15,5),layout='constrained');q.rename(columns=LABELS).plot.bar(ax=ax)
    ax.set(ylabel='MAE (mm)',title='Fixed 500 epochs, mean of five seeds');ax.grid(axis='y',alpha=.2);fig.savefig(ROOT/'case_mae.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(4,4,figsize=(16,13),layout='constrained')
    for ax,(caseid,g) in zip(axs.ravel(),traj[traj.policy=='fixed500'].groupby('case')):
        true=g.drop_duplicates('run').sort_values('run');ax.scatter(true.time,true.target,c='black',s=15,label='Measured VB',zorder=4)
        for name in ['dynamics_prior','sensor_filter','background_filter']:
            v=g[g.name==name].groupby('time').prediction.mean();ax.plot(v.index,v.values,label=LABELS[name])
        ax.set(title=f'Case {caseid}',xlabel='Time',ylabel='VB (mm)');ax.grid(alpha=.2)
    for ax in axs.ravel()[14:]:ax.axis('off')
    axs.ravel()[0].legend(fontsize=7);fig.savefig(ROOT/'trajectories.png',dpi=145);plt.close(fig)
    print(summary.to_string(index=False));print(choices.to_string(index=False));print(contrast.to_string(index=False))
    write_json(ROOT/'status.json',dict(status='evaluated_and_audited',dynamics_runs=24,epochs=500))


if __name__=='__main__':main()
