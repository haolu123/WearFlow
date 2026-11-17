"""Replay existing checkpoints for a common-target current-VB learning-curve panel.
No training, epoch selection, or new hyperparameter search is performed.
"""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.func import functional_call,vmap
from ..config import write_json,file_hash
from ..training.single_shot_adaptation import load_parent
from ..training.recursive_adaptation import layout,rollout
from .plot_current12_fusion_epochs import predictions

BASE=Path('wear_monitoring_dynamics/outputs/current12_fusion_20261013')
JOINT=BASE.with_name('current12_single_joint_20261014')
OUT=BASE.with_name('model_learning_curves_20261018')
LABELS={'ct_direct':'工况＋时间：直接估计','sensor_direct':'Sensor monitor：直接估计','sensor_ode':'Sensor初始化＋纯ODE','ct_fusion':'工况/时间观测融合＋截断','sensor_fusion':'Sensor观测融合＋截断','joint_fusion':'Sensor融合：联合微调中','before_joint':'Sensor融合：微调前冻结参考'}

class Propagator(torch.nn.Module):
    def __init__(self,dynamics):
        super().__init__();self.dynamics=dynamics
    def forward(self,wear,condition,dt):
        return self.dynamics.propagate(wear,condition,dt)

@torch.no_grad()
def parallel_rollout(model,batch,raw,st,alpha,clamp,weights=None,ct=False):
    # E checkpoint models x N records x 1; each checkpoint has its own ODE weights.
    values=raw.clone();wrapper=Propagator(model.dynamics)
    for prev,idx in st['steps']:
        old=values[:,prev];c=batch['condition'][idx];dt=batch['time'][idx]-batch['time'][prev]
        if weights is None:
            e,n,_=old.shape
            prior=model.dynamics.propagate(old.reshape(-1,1),c.repeat(e,1),dt.repeat(e,1)).reshape(e,n,1)
        else:
            prior=vmap(lambda p,w: functional_call(wrapper,p,(w,c,dt)),in_dims=(0,0))(weights,old)
        valid=torch.ones(len(idx),dtype=torch.bool) if ct else batch['sensor_valid'][idx]
        gain=alpha*valid[None,:,None];post=prior+gain*(raw[:,idx]-prior)
        values[:,idx]=torch.maximum(old,post) if clamp else post
    return values

def case_rows(values,t,st,sc,epochs,method,partition,fold,seed):
    y=values.numpy()[:,:,0]*sc['wear'][1]+sc['wear'][0];rows=[]
    for seq in st['seqs']:
        ids=seq[np.isin(seq,st['ids'])];err=abs(y[:,ids]-t['VB'][ids])
        rows.extend(dict(method=method,partition=partition,fold=fold,seed=seed,case=int(t['case'][seq[0]]),epoch=int(e),mae=float(v),points=len(ids)) for e,v in zip(epochs,err.mean(1)))
    return rows

def main():
    start=time.perf_counter();torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    monitors=pd.read_csv(BASE/'study_manifest.csv');odes=pd.read_csv(BASE/'reused_ode_manifest.csv');jm=pd.read_csv(JOINT/'study_manifest.csv');audits=[]
    for r in monitors[(monitors.type=='monitor')&(monitors.role=='final')].itertuples():
        dest=OUT/f'{r.fold}_seed{r.seed}.csv'
        if dest.exists():continue
        od=odes[(odes.fold==r.fold)&(odes.seed==r.seed)].iloc[0].run_dir
        model,b,t,parent=load_parent(r.run_dir,od,'cpu');sc=parent['scalers'];sp=parent['split']
        cfg,_,sensor=predictions(r.run_dir);_,_,ct=predictions(Path(cfg['monitor']['fallback_checkpoint']).parent)
        raw=torch.tensor((sensor-sc['wear'][0])/sc['wear'][1],dtype=torch.float32)[:,:,None]
        ctraw=torch.tensor((ct-sc['wear'][0])/sc['wear'][1],dtype=torch.float32)[:,:,None]
        alpha=1. if r.fold=='fold_a' else .25;ctalpha=.75 if r.fold=='fold_a' else 0.
        rows=[];structures={p:layout(t,sp[p],'cpu') for p in ['train','test']}
        for partition,st in structures.items():
            for name,values in [('ct_direct',ctraw),('sensor_direct',raw),('sensor_ode',parallel_rollout(model,b,raw,st,0.,False)),('sensor_fusion',parallel_rollout(model,b,raw,st,alpha,True))]:
                rows+=case_rows(values,t,st,sc,range(1,501),name,partition,r.fold,r.seed)
            # CT observation, but shared sensor initial state at every checkpoint epoch.
            obs=ctraw.clone();initial=list(st['initial']);obs[:,initial]=raw[:,initial]
            values=parallel_rollout(model,b,obs,st,ctalpha,True,ct=True)
            rows+=case_rows(values,t,st,sc,range(1,501),'ct_fusion',partition,r.fold,r.seed)
        joint=jm[(jm.fold==r.fold)&(jm.role=='final')&(jm.seed==r.seed)&(jm.name=='joint_l1')].iloc[0]
        d=Path(joint.run_dir);original={k:v.clone() for k,v in model.state_dict().items()};rawlist=[];odeweights=[]
        for epoch in range(501):
            state=original if epoch==0 else torch.load(d/'epochs'/f'epoch_{epoch:04d}.pt',map_location='cpu',weights_only=True)
            model.load_state_dict(state);model.eval()
            with torch.no_grad():rawlist.append(model.estimate(b,np.arange(len(t['case']))))
            odeweights.append({k:v for k,v in state.items() if k.startswith('dynamics.')})
        rawjoint=torch.stack(rawlist);weights={k:torch.stack([p[k] for p in odeweights]) for k in odeweights[0]}
        maxdiff=0.
        for partition,st in structures.items():
            values=parallel_rollout(model,b,rawjoint,st,alpha,True,weights)
            rows+=case_rows(values,t,st,sc,range(501),'joint_fusion',partition,r.fold,r.seed)
            # Compare batched replay with the existing scalar-checkpoint rollout at both ends.
            for epoch,state in [(0,original),(500,torch.load(d/'last.pt',map_location='cpu',weights_only=True)['model'])]:
                model.load_state_dict(state)
                with torch.no_grad():expected=rollout(model,b,rawjoint[epoch],st,[(alpha,True)])
                err=float(abs(expected-values[epoch]).max())*sc['wear'][1];maxdiff=max(maxdiff,err);assert err<5e-6
        frame=pd.DataFrame(rows)
        for part in ['train','test']:
            before=frame[(frame.method=='sensor_fusion')&(frame.epoch==500)&(frame.partition==part)].sort_values('case')
            zero=frame[(frame.method=='joint_fusion')&(frame.epoch==0)&(frame.partition==part)].sort_values('case')
            np.testing.assert_allclose(before.mae,zero.mae,atol=3e-6,rtol=1e-5)
        frame.to_csv(dest,index=False)
        audit=dict(fold=r.fold,seed=r.seed,scalar_replay_max_error_mm=maxdiff,monitor_checkpoint=file_hash(Path(r.run_dir)/'last.pt'),joint_checkpoint=file_hash(d/'last.pt'),joint_method='joint_l1',sensor_alpha=alpha,ct_alpha=ctalpha)
        write_json(OUT/f'{r.fold}_seed{r.seed}_audit.json',audit);print(f'{r.fold} seed {r.seed} complete',flush=True)
    curves=pd.concat([pd.read_csv(p) for p in OUT.glob('fold_*_seed*.csv')]);curves.to_csv(OUT/'epoch_case_metrics.csv.gz',index=False)
    # Check 500-epoch endpoints against independent prior reports (common 128 targets).
    old=pd.read_csv(BASE/'current_case_metrics.csv');old=old[old.start_kind=='first_record']
    for new,prior in [('sensor_ode','sensor_a0_m0'),('ct_fusion','selected_ct_m1'),('sensor_fusion','selected_sensor_m1'),('ct_direct','ct_a1_m0'),('sensor_direct','sensor_a1_m0')]:
        a=curves[(curves.method==new)&(curves.partition=='test')&(curves.epoch==500)]
        z=a.merge(old[old.method==prior],on=['fold','seed','case'],validate='one_to_one',suffixes=('_replay','_old'));assert len(z)==70
        np.testing.assert_allclose(z.mae_replay,z.mae_old,atol=5e-6,rtol=1e-5)
    for _,g in curves[curves.partition=='test'].groupby(['method','epoch','seed']):assert g.points.sum()==128 and len(g)==14
    summary=[]
    for subset in ['overall','fold_a','fold_b']:
        q=curves if subset=='overall' else curves[curves.fold==subset]
        ps=q.groupby(['method','partition','epoch','seed']).mae.mean().reset_index()
        summary.append(ps.groupby(['method','partition','epoch']).mae.agg(['mean','std']).reset_index().assign(subset=subset))
    summary=pd.concat(summary);summary.to_csv(OUT/'epoch_summary.csv',index=False)
    plot(summary)
    endpoints=summary[(summary.epoch==500)&(summary.partition=='test')];endpoints.to_csv(OUT/'epoch500_summary.csv',index=False)
    write_json(OUT/'audit.json',dict(status='passed',targets_per_seed=128,seeds=5,base_epochs=500,joint_epochs_including_zero=501,seconds=time.perf_counter()-start,training_performed=False))
    print(endpoints[['method','subset','mean']].to_string(index=False),flush=True)

def plot(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    methods=['ct_direct','sensor_direct','sensor_ode','ct_fusion','sensor_fusion'];colors=plt.cm.tab10.colors
    for begin in [1,50]:
        fig,axs=plt.subplots(5,3,figsize=(15,16),sharex=True)
        for row,method in enumerate(methods):
            for col,subset in enumerate(['overall','fold_a','fold_b']):
                ax=axs[row,col]
                for part,label,color in [('train','训练集','#2474b5'),('test','测试集','#db6336')]:
                    q=summary[(summary.method==method)&(summary.subset==subset)&(summary.partition==part)&(summary.epoch>=begin)]
                    ax.plot(q.epoch,q['mean'],label=label,color=color);ax.fill_between(q.epoch,q['mean']-q['std'],q['mean']+q['std'],color=color,alpha=.15)
                ax.set(title=f'{LABELS[method]}\n{dict(overall="总体",fold_a="Fold A",fold_b="Fold B")[subset]}',ylabel='Case 等权 MAE（mm）',xlabel='Monitor 预训练 epoch');ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.suptitle('当前VB：统一128个测试标签；ODE固定最终模型；融合权重固定；阴影为5 seed ±1 SD',fontsize=13)
        fig.tight_layout(rect=[0,0,1,.98]);fig.savefig(OUT/f'base_epoch{begin}_500.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.6))
    for ax,subset in zip(axs,['overall','fold_a','fold_b']):
        for part,label,color in [('train','训练集','#2474b5'),('test','测试集','#db6336')]:
            q=summary[(summary.method=='joint_fusion')&(summary.subset==subset)&(summary.partition==part)]
            ax.plot(q.epoch,q['mean'],color=color,label=label+'：微调中');ax.fill_between(q.epoch,q['mean']-q['std'],q['mean']+q['std'],color=color,alpha=.15)
            ax.axhline(q[q.epoch==0]['mean'].iloc[0],color=color,ls='--',label=label+'：微调前')
        ax.set(title=dict(overall='总体',fold_a='Fold A',fold_b='Fold B')[subset],xlabel='额外联合微调 epoch',ylabel='Case 等权 MAE（mm）');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Sensor观测融合＋单调截断：固定 joint_l1；α沿用微调前 A=1、B=0.25；ODE和monitor同时更新')
    fig.tight_layout();fig.savefig(OUT/'joint_before_after.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.6))
    for ax,subset in zip(axs,['overall','fold_a','fold_b']):
        for method,color in zip(methods,colors):
            q=summary[(summary.method==method)&(summary.subset==subset)&(summary.partition=='test')&(summary.epoch>=50)]
            ax.plot(q.epoch,q['mean'],label=LABELS[method],color=color)
        ax.set(title=subset,xlabel='Monitor 预训练 epoch',ylabel='测试集 Case MAE（mm）');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('五类方法的测试曲线对照（epoch 50—500）；仅供诊断，不按测试选择epoch')
    fig.tight_layout();fig.savefig(OUT/'test_comparison.png',dpi=160);plt.close(fig)
if __name__=='__main__':main()

