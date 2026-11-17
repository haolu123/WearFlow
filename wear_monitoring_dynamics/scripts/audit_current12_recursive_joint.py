"""Independent audit and diagnostics for the completed recursive study; no training."""

import json,time

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from ..config import file_hash,write_json

from ..training.recursive_adaptation import prepare,layout,rollout,future,record,CONFIGS

from ..data.preprocessing import fit_scalers

from .run_psd_sweep import rank

from .report_current12_fusion import metrics,md

ROOT=Path('wear_monitoring_dynamics/outputs/current12_recursive_joint_20261014')

COLS=['epoch','split','fusion','task','case','origin_run','target_run','horizon','target_vb_mm','predicted_vb_mm']

KEYS=COLS[1:-1]

class Collector:

    def __init__(self):self.rows=[]

    def writerows(self,rows):self.rows.extend(rows)



def selected(frame,selection):

    parts=[]

    for r in selection[selection.selected].itertuples():

        q=frame[(frame.fold==r.fold)&(frame.method==r.method)].copy()

        q['method']='selected_'+r.family;parts.append(q)

    return pd.concat(parts,ignore_index=True)



def main():

    started=time.perf_counter();torch.set_num_threads(1)

    manifest=pd.read_csv(ROOT/'study_manifest.csv');assert len(manifest)==168

    sel=pd.read_csv(ROOT/'inner_selection.csv');pred=pd.read_csv(ROOT/'predictions.csv.gz')

    seen=set();common={};audits=[];trajectories=[];curves=[];oracle=[];rowcount=0

    for num,r in enumerate(manifest.itertuples(),1):

        d=Path(r.run_dir);identity=json.loads((d/'identity.json').read_text());task=identity['task']

        assert identity['code_hash']==file_hash('wear_monitoring_dynamics/training/recursive_adaptation.py')

        assert identity['start_hash']==file_hash(task['start_checkpoint']) and '/joint_l1/' in task['start_checkpoint'].replace('\\','/')

        assert json.loads((d/'status.json').read_text())['status']=='completed'

        h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,501)) and np.isfinite(h.to_numpy()).all()

        q=pd.read_csv(d/'epoch_predictions.csv.gz');rowcount+=len(q)

        assert np.isfinite(q.predicted_vb_mm).all()

        count=q.groupby('epoch').size();assert count.index.tolist()==list(range(501)) and count.nunique()==1

        # Order-independent key/target identity and uniqueness for every epoch, not just row counts.

        hashes=pd.util.hash_pandas_object(q[KEYS],index=False).to_numpy().reshape(501,-1)

        assert len(np.unique(hashes[0]))==len(hashes[0]) and np.all(hashes==hashes[0])

        q['ae']=abs(q.predicted_vb_mm-q.target_vb_mm)

        cm=q[q.epoch>0].groupby(['epoch','split','fusion','task','case']).ae.mean().groupby(['epoch','split','fusion','task']).mean().reset_index()

        for (split,config,typ),g in cm.groupby(['split','fusion','task']):

            np.testing.assert_allclose(g.sort_values('epoch').ae,h[f'{split}_{config}_{typ}_mae'],atol=1e-8,rtol=1e-6)

        zero=q[q.epoch==0];basekey=(r.fold,r.role,str(r.holdout),r.seed)

        if r.mode=='single':common[basekey]=zero

        else:

            base=common[basekey];base=base[base.fusion==zero.fusion.iloc[0]]

            np.testing.assert_allclose(zero.predicted_vb_mm,base.predicted_vb_mm,atol=1e-7,rtol=1e-6)

        model,b,t,parent=prepare(task,'cpu');sp=parent['split'];sc=parent['scalers'];cfg=parent['config']

        assert sc['wear'][1]>0 and sc['time_scale']>0

        assert not set(sp['train'])&set(sp['outer_test_reserved'])

        if r.role=='inner':assert r.holdout not in sp['train'] and r.holdout in sp['validation']

        splitkey=(r.fold,tuple(sp['train']))

        if splitkey not in seen:

            assert json.loads(json.dumps(fit_scalers(t,sp['train'],cfg['data']['sensor_preprocessing'])))==json.loads(json.dumps(sc))

            assert cfg['data']['sensor_quality']['rules']['train_cases']==sp['train']

            score,_=rank(t,sp['train']);names=cfg['data']['feature_archive']['names']

            order=sorted([i for i,n in enumerate(names) if n.startswith(('smcAC_','smcDC_'))],key=lambda i:(-score[i],names[i]))

            assert cfg['monitor']['selected_indices']==order[:12]

            fallback=torch.load(cfg['monitor']['fallback_checkpoint'],map_location='cpu',weights_only=True)

            assert fallback['split']==sp and fallback['config']['experiment']['seed']==r.seed

            seen.add(splitkey)

        structures={name:layout(t,cases,'cpu') for name,cases in [('train',sp['train']),('validation',sp['validation']),('test_diagnostic',sp['outer_test_reserved'])] if cases}

        configs=CONFIGS if r.mode=='single' else [(task['alpha'],task['clamp'])]

        maxerr=0

        for epoch in [0,500]:

            if epoch==500:

                state=torch.load(d/'last.pt',map_location='cpu',weights_only=True)

                assert state['split']==sp and state['scalers']==sc and state['epoch']==500

                model.load_state_dict(state['model'])

            c=Collector();record(model,b,t,sc,structures,configs,epoch,c)

            expected=pd.DataFrame(c.rows,columns=COLS).sort_values(KEYS)

            logged=q[q.epoch==epoch].sort_values(KEYS)

            assert expected[KEYS].reset_index(drop=True).equals(logged[KEYS].reset_index(drop=True))

            err=np.max(abs(expected.predicted_vb_mm.to_numpy()-logged.predicted_vb_mm.to_numpy()));maxerr=max(maxerr,err)

            assert err<3e-6

        grad=json.loads((d/'gradient_audit.json').read_text());assert grad['monitor']>0 and grad['ode']>0 and grad['no_future_sensor'] and grad['endpoints']

        # Recheck end-of-training full-case gradient and label-independent causal behavior.

        st=structures['train'];model.zero_grad(set_to_none=True);raw=model.estimate(b,np.arange(len(t['case'])))

        states=rollout(model,b,raw,st,configs);last=int(st['seqs'][0][-1]);first=int(st['seqs'][0][0])

        initial_grad=float(torch.autograd.grad(states[last].sum(),raw,retain_graph=True)[0][first].abs().sum())

        (states[st['ids']]-b['target'][st['ids']]).square().mean().backward()

        norms={name:sum(float(p.grad.square().sum()) for p in branch.parameters() if p.grad is not None)**.5 for name,branch in [('monitor',model.monitor),('ode',model.dynamics)]}

        assert np.isfinite(list(norms.values())+[initial_grad]).all()

        with torch.no_grad():

            modified=raw.detach().clone();modified[last]+=100

            altered=rollout(model,b,modified,st,configs);torch.testing.assert_close(altered[st['seqs'][0][:-1]],states.detach()[st['seqs'][0][:-1]],rtol=0,atol=0)

            zero_state=rollout(model,b,raw.detach(),st,[(0.,False)])

            one=rollout(model,b,raw.detach(),st,[(1.,False)])

            valid=st['ids'][t['sensor_valid'][st['ids']]];torch.testing.assert_close(one[valid],raw.detach()[valid],atol=1e-6,rtol=1e-5)

            bad=dict(b);bad['sensor_valid']=torch.zeros_like(b['sensor_valid'])

            skipped=rollout(model,bad,raw.detach(),st,configs)

            torch.testing.assert_close(skipped,zero_state.expand_as(skipped),atol=2e-6,rtol=1e-5)

            for prev,ids in st['steps']:

                torch.testing.assert_close(zero_state[ids],model.dynamics.propagate(zero_state[prev],b['condition'][ids],b['time'][ids]-b['time'][prev]))

        if r.role=='final':

            assert len(list((d/'epochs').glob('epoch_*.pt')))==500

            for (split,config,typ),g in cm.groupby(['split','fusion','task']):

                curves.append(g.assign(fold=r.fold,seed=r.seed,method=r.mode+'_'+config))

            st=structures['test_diagnostic']

            for family,epoch in ([(r.mode,500),('frozen',0)] if r.mode=='single' else [(r.mode,500)]):

                if epoch==0:model.load_state_dict(torch.load(task['start_checkpoint'],map_location='cpu',weights_only=True)['model'])

                model.eval()

                with torch.no_grad():

                    raw=model.estimate(b,np.arange(len(t['case'])));states=rollout(model,b,raw,st,configs)

                    a,zz=st['pairs'][:,0],st['pairs'][:,1]

                    oy=model.dynamics.propagate(b['target'][a],b['condition'][a],b['time'][zz]-b['time'][a]).numpy().ravel()*sc['wear'][1]+sc['wear'][0]

                    for k,(alpha,clamp) in enumerate(configs):

                        method=f'{family}_a{alpha:g}_m{int(clamp)}'

                        oracle.append(pd.DataFrame(dict(case=t['case'][a],origin_run=t['run'][a],target_run=t['run'][zz],horizon=st['pairs'][:,2],target_vb_mm=t['VB'][zz],predicted_vb_mm=oy)).assign(method=method,fold=r.fold,seed=r.seed))

                        for seq in st['seqs']:

                            old=states[seq[:-1],k:k+1]

                            prior=model.dynamics.propagate(old,b['condition'][seq[1:]],b['time'][seq[1:]]-b['time'][seq[:-1]])

                            pre=prior+alpha*b['sensor_valid'][seq[1:],None]*(raw[seq[1:]]-prior)

                            post=states[seq[1:],k:k+1]

                            if clamp:assert torch.all(post>=old-1e-7)

                            y=states[seq,k].numpy()*sc['wear'][1]+sc['wear'][0]

                            elapsed=t['time'][seq]-t['time'][seq[0]]

                            trajectories.append(pd.DataFrame(dict(case=t['case'][seq],run=t['run'][seq],time=t['time'][seq],elapsed=elapsed,half=np.where(elapsed<=elapsed[-1]/2,'first','second'),initial=np.arange(len(seq))==0,sensor_valid=t['sensor_valid'][seq],target_vb_mm=t['VB'][seq],predicted_vb_mm=y,raw_monitor_mm=raw[seq,0].numpy()*sc['wear'][1]+sc['wear'][0],delta_mm=np.r_[np.nan,np.diff(y)],clamp_trigger=np.r_[False,(clamp&(pre<old-1e-7)).numpy().ravel()],clamp_amount_mm=np.r_[0,(post-pre).numpy().ravel()*sc['wear'][1]])).assign(method=method,fold=r.fold,seed=r.seed))

        audits.append(dict(id=r.id,epochs=500,rows=len(q),reload_error=maxerr,initial_gradient_start=grad['initial_to_last_gradient'],initial_gradient_end=initial_grad,monitor_gradient_end=norms['monitor'],ode_gradient_end=norms['ode']))

        if num%14==0:print(f'AUDIT {num}/168',flush=True)

    for s in sel[sel.selected].itertuples():

        v=pred[(pred.role=='inner')&(pred.fold==s.fold)&(pred.family==s.family)&(pred.task=='current')]

        scores=v.assign(ae=abs(v.predicted_vb_mm-v.target_vb_mm)).groupby(['method','case']).ae.mean().groupby('method').mean()

        assert scores.sort_index().idxmin()==s.method

    external=pred[pred.role=='final']

    for typ,n in [('monitor',138),('current',128)]:

        for _,g in external[external.task==typ].groupby(['method','seed']):assert len(g)==n

    for _,g in external.groupby(['seed','task']):

        keys=[set(map(tuple,x[['case','origin_run','target_run','horizon']].to_numpy())) for _,x in g.groupby('method')];assert all(x==keys[0] for x in keys)

    selected_pred=selected(external,sel);selected_pred.to_csv(ROOT/'selected_predictions.csv.gz',index=False)

    tr=pd.concat(trajectories);tr.to_csv(ROOT/'all_record_trajectories.csv.gz',index=False);ts=selected(tr,sel)

    ts.to_csv(ROOT/'selected_trajectories.csv.gz',index=False)

    halves=[]

    for half,g in ts[(~ts.initial)&ts.target_vb_mm.notna()].groupby('half'):

        _,sm=metrics(g,['method']);halves.append(sm.assign(half=half))

    pd.concat(halves).to_csv(ROOT/'trajectory_halves.csv',index=False)

    changes=tr[~tr.initial].copy();changes['drop']=changes.delta_mm< -1e-7;changes['drop_mm']=(-changes.delta_mm).clip(lower=0)

    diag=changes.groupby(['method','fold','seed','case']).agg(drop_fraction=('drop','mean'),mean_drop_mm=('drop_mm','mean'),max_drop_mm=('drop_mm','max'),clamp_fraction=('clamp_trigger','mean'),mean_clamp_mm=('clamp_amount_mm','mean')).reset_index()

    diag.to_csv(ROOT/'monotonic_case_diagnostics.csv',index=False);selected(diag,sel).groupby('method')[['drop_fraction','mean_drop_mm','max_drop_mm','clamp_fraction','mean_clamp_mm']].mean().to_csv(ROOT/'selected_monotonic_diagnostics.csv')

    oo=pd.concat(oracle);oo=pd.concat([oo,selected(oo,sel)]);per,sm=metrics(oo,['method','horizon']);sm.to_csv(ROOT/'oracle_summary.csv',index=False);per.to_csv(ROOT/'oracle_case_metrics.csv',index=False)

    cc=selected(pd.concat(curves),sel);cc.to_csv(ROOT/'selected_epoch_curves.csv.gz',index=False)

    # Paired stratified case bootstrap, averaging seeds before resampling cases.

    boot=[];rng=np.random.default_rng(20261015)

    for (typ,horizon),g in selected_pred.groupby(['task','horizon']):

        for subset,mask in [('overall',np.ones(len(g),bool)),('fold_a',g.fold=='fold_a'),('fold_b',g.fold=='fold_b'),('fold_a_in',(g.fold=='fold_a')&(g.target_vb_mm<=.74)),('fold_a_out',(g.fold=='fold_a')&(g.target_vb_mm>.74))]:

            x=g[mask].assign(ae=lambda x:abs(x.predicted_vb_mm-x.target_vb_mm))

            pivot=x.groupby(['fold','case','seed','method']).ae.mean().groupby(['fold','case','method']).mean().unstack('method')

            for test,ref in [('selected_recursive','selected_single'),('selected_recursive','selected_frozen'),('selected_single','selected_frozen')]:

                delta=pivot[test]-pivot[ref];samples=[]

                for fold,a in delta.groupby(level=0):

                    v=a.to_numpy();samples.append(rng.choice(v,(5000,len(v)),replace=True).sum(1))

                draws=sum(samples)/len(delta);lo,hi=np.quantile(draws,[.025,.975])

                boot.append(dict(task=typ,horizon=horizon,subset=subset,test=test,reference=ref,delta_mae=delta.mean(),ci_low=lo,ci_high=hi,cases=len(delta)))

    pd.DataFrame(boot).to_csv(ROOT/'case_bootstrap.csv',index=False)

    for typ in ['monitor','current','future']:

        s=pd.read_csv(ROOT/f'{typ}_summary.csv');s[s.method.str.startswith('selected_')].to_csv(ROOT/f'{typ}_selected_summary.csv',index=False)

        per=pd.read_csv(ROOT/f'{typ}_case_metrics.csv');per[per.method.str.startswith('selected_')].to_csv(ROOT/f'{typ}_selected_cases.csv',index=False)

    plots(cc,ts)

    pd.DataFrame(audits).to_csv(ROOT/'independent_run_audit.csv',index=False)

    write_json(ROOT/'independent_audit.json',dict(status='passed',runs=len(audits),epochs=84000,raw_prediction_rows=rowcount,training_splits=len(seen),monitor_targets=138,current_targets=128,audit_seconds=time.perf_counter()-started))

    print('AUDIT PASSED',flush=True)



def plots(cc,ts):

    import matplotlib

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    fig,axs=plt.subplots(2,3,figsize=(15,8))

    for col,subset in enumerate(['overall','fold_a','fold_b']):

        for row,typ in enumerate(['current','future']):

            q=cc[(cc.task==typ)&((cc.fold==subset) if subset!='overall' else True)]

            for (method,split),g in q.groupby(['method','split']):

                if split not in ['train','test_diagnostic']:continue

                y=g.groupby('epoch').ae.mean();axs[row,col].plot(y.index,y,label=method.replace('selected_','')+'/'+split,ls='--' if split=='train' else '-')

            axs[row,col].set(title=f'{subset}: {typ}',xlabel='Additional epoch',ylabel='Case MAE (mm)');axs[row,col].grid(alpha=.2);axs[row,col].legend(fontsize=7)

    fig.suptitle('Diagnostic only: final epoch 500, configuration selected by inner validation');fig.tight_layout();fig.savefig(ROOT/'epoch_curves.png',dpi=140);plt.close(fig)

    cases=sorted(ts.case.unique());fig,axs=plt.subplots(4,4,figsize=(16,13))

    for ax,case in zip(axs.flat,cases):

        q=ts[ts.case==case];truth=q.drop_duplicates('run').sort_values('time');ax.plot(truth.time,truth.target_vb_mm,'ko',ms=3,label='Measured VB')

        for method,g in q.groupby('method'):

            z=g.groupby('time').predicted_vb_mm.agg(['mean','std']);ax.plot(z.index,z['mean'],label=method.replace('selected_',''));ax.fill_between(z.index,z['mean']-z['std'],z['mean']+z['std'],alpha=.12)

        ax.set(title=f'Case {case}',xlabel='Machining time',ylabel='VB (mm)');ax.grid(alpha=.2)

    for ax in list(axs.flat)[len(cases):]:ax.set_visible(False)

    axs.flat[0].legend(fontsize=7);fig.suptitle('Inner-selected methods; band = seed SD, not a case confidence interval');fig.tight_layout();fig.savefig(ROOT/'case_trajectories.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()

