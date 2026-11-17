"""Replay saved MLP epochs with frozen final ODE and validation-selected gains."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .report_current12_fusion import monitor
from ..data.monitor_batch import make_monitor_batch
from ..models.registry import build_model,build_dynamics
from ..evaluation.fixed_fusion import rollout

ROOT=Path('wear_monitoring_dynamics/outputs/current12_fusion_20261013')
OUT=ROOT/'epoch_curves'

def predictions(directory):
    directory=Path(directory)
    cfg,table,_=monitor(directory)
    sc=json.loads((directory/'scalers.json').read_text())
    batch=make_monitor_batch(cfg,table,sc,'cpu')
    model=build_model(cfg).eval()
    values=[]
    with torch.no_grad():
        for epoch in range(1,501):
            ck=torch.load(directory/'epochs'/f'epoch_{epoch:04d}.pt',map_location='cpu',weights_only=True)
            assert ck['epoch']==epoch
            model.load_state_dict(ck['model'])
            values.append(model.estimate(batch,np.arange(len(table['case']))).numpy().ravel()*sc['wear'][1]+sc['wear'][0])
    return cfg,table,np.asarray(values)

def main():
    start=time.perf_counter();torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    manifest=pd.read_csv(ROOT/'study_manifest.csv');odes=pd.read_csv(ROOT/'reused_ode_manifest.csv')
    selected=pd.read_csv(ROOT/'selected_alpha.csv');selected=selected[(selected.start_kind=='first_record')&selected.monotone]
    for r in manifest[(manifest.type=='monitor')&(manifest.role=='final')].itertuples():
        dest=OUT/f'{r.fold}_seed{r.seed}_cases.csv'
        if dest.exists():continue
        cfg,t,sensor=predictions(r.run_dir)
        _,cttable,ct=predictions(Path(cfg['monitor']['fallback_checkpoint']).parent)
        assert np.array_equal(t['case'],cttable['case']) and np.array_equal(t['run'],cttable['run'])
        od=Path(odes[(odes.fold==r.fold)&(odes.seed==r.seed)].iloc[0].run_dir)
        oc=json.loads((od/'provenance.json').read_text())['config'];sc=json.loads((od/'scalers.json').read_text())
        ode=build_dynamics(oc['dynamics']).eval();ode.load_state_dict(torch.load(od/'last.pt',map_location='cpu',weights_only=True)['model'])
        condition=np.column_stack([(np.column_stack([t['DOC'],t['feed']])-np.asarray(sc['condition'][0]))/np.asarray(sc['condition'][1]),t['material']==1,t['material']==2])
        split=json.loads((Path(r.run_dir)/'split.json').read_text());rows=[]
        gains=[float(selected[(selected.fold==r.fold)&(selected.source==src)].iloc[0].alpha) for src in ['ct','sensor']]
        for partition in ['train','test']:
            for case in split[partition]:
                ids=np.flatnonzero(t['case']==case);ids=ids[np.argsort(t['time'][ids])]
                @torch.no_grad()
                def propagate(values,dt):
                    v=torch.tensor(values.reshape(-1,1),dtype=torch.float32)
                    c=torch.tensor(condition[ids[0]],dtype=torch.float32).reshape(1,-1).expand(len(v),-1)
                    delta=torch.full_like(v,float(dt)/sc['time_scale'])
                    return ode.propagate((v-sc['wear'][0])/sc['wear'][1],c,delta).numpy().ravel()*sc['wear'][1]+sc['wear'][0]
                obs=np.concatenate([ct[:,ids].T,sensor[:,ids].T],axis=1)
                valid=np.concatenate([np.ones((len(ids),500),bool),np.broadcast_to(t['sensor_valid'][ids,None],(len(ids),500))],axis=1)
                _,post=rollout(t['time'][ids],obs,valid,np.repeat(gains,500),np.ones(1000,bool),np.tile(sensor[:,ids[0]],2),propagate)
                good=np.isfinite(t['VB'][ids]);good[0]=False
                err=post[good]-t['VB'][ids[good],None]
                for k,src in enumerate(['ct','sensor']):
                    mae=np.abs(err[:,k*500:(k+1)*500]).mean(axis=0)
                    rows.extend(dict(fold=r.fold,seed=r.seed,partition=partition,case=case,method=f'selected_{src}_m1',epoch=e+1,mae=float(v),points=int(good.sum())) for e,v in enumerate(mae))
        pd.DataFrame(rows).to_csv(dest,index=False)
        print(f'{r.fold} seed {r.seed} done, {time.perf_counter()-start:.1f}s',flush=True)
    cases=pd.concat([pd.read_csv(p) for p in OUT.glob('fold_*_seed*_cases.csv')],ignore_index=True)
    cases.to_csv(OUT/'epoch_case_metrics.csv.gz',index=False)
    summaries=[]
    for subset in ['overall','fold_a','fold_b']:
        q=cases if subset=='overall' else cases[cases.fold==subset]
        perseed=q.groupby(['method','partition','epoch','seed']).mae.mean().reset_index()
        summaries.append(perseed.groupby(['method','partition','epoch']).mae.agg(['mean','std']).reset_index().assign(subset=subset))
    summary=pd.concat(summaries);summary.to_csv(OUT/'epoch_summary.csv',index=False)
    original=pd.read_csv(ROOT/'current_case_metrics.csv')
    original=original[(original.start_kind=='first_record')&original.method.isin(['selected_ct_m1','selected_sensor_m1'])]
    last=cases[(cases.epoch==500)&(cases.partition=='test')].merge(original,on=['fold','seed','case','method'],suffixes=('_new','_old'),validate='one_to_one')
    assert len(last)==140
    discrepancy=float(abs(last.mae_new-last.mae_old).max());assert discrepancy<1e-5,discrepancy
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    for zoom in [False,True]:
        fig,axs=plt.subplots(2,3,figsize=(15,8),sharex=True)
        for row,(method,label) in enumerate([('selected_ct_m1','工况＋时间观测融合＋单调截断'),('selected_sensor_m1','Sensor观测融合＋单调截断')]):
            for col,(subset,title) in enumerate([('overall','总体'),('fold_a','Fold A'),('fold_b','Fold B')]):
                ax=axs[row,col]
                for part,name,color in [('train','训练集','#2474b5'),('test','测试集','#db6336')]:
                    q=summary[(summary.method==method)&(summary.subset==subset)&(summary.partition==part)&(summary.epoch>=(50 if zoom else 1))]
                    ax.plot(q.epoch,q['mean'],label=name,color=color,lw=1.5)
                    ax.fill_between(q.epoch,q['mean']-q['std'],q['mean']+q['std'],alpha=.15,color=color)
                ax.set_title(f'{label}\n{title}',fontsize=11);ax.grid(alpha=.2);ax.set_ylabel('Case 等权 MAE（mm）');ax.set_xlabel('MLP checkpoint epoch');ax.legend()
        fig.suptitle('逐 epoch 融合评估：ODE 固定为第500轮，α固定；阴影为5 seed ±1标准差',fontsize=13)
        fig.tight_layout();fig.savefig(OUT/('curves_epoch50_500.png' if zoom else 'curves_epoch1_500.png'),dpi=160);plt.close(fig)
    info=dict(seconds=time.perf_counter()-start,epoch500_max_case_mae_difference=discrepancy,checkpoints=10000,epochs=500)
    (OUT/'audit.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text('''# 两种单调融合方法的逐epoch评估曲线

这不是融合网络的联合训练loss。用保存的第1—500轮MLP checkpoint重新计算当前VB递推MAE，没有重新训练。

- ODE固定为原先训练完成的第500轮；alpha固定为原先内层验证选择值。工况时间融合：fold A=0.75、fold B=0；sensor融合：fold A=1、fold B=0.25。
- 每轮两方法都用该轮sensor12 MLP初始化。ct观测使用该轮ct MLP，sensor观测使用该轮sensor12 MLP。sensor12无效输入后备仍为训练协议中的冻结最终ct模型。因此横轴同时改变共享初值与观测模型，不单独隔离其中一项作用。
- first_record主协议，真实时间间隔；每次校正后max(上一后验,当前更新)。无效sensor跳过sensor校正，ct观测仍可用。初始化点不计分。
- 训练/测试各case先计算MAE，再等权平均；实线5seed均值，阴影±1标准差，不是独立case置信区间。
- fold B工况时间融合alpha=0，所以该曲线只由当轮sensor初值和冻结ODE决定，不受当轮ct观测影响。
- 训练集曲线是事后递推评估，不是优化器实际训练目标。逐epoch测试仅供诊断，不据此重新选择epoch或alpha。
- 第500轮逐case结果已与原始报告核对，误差见audit.json。

完整曲线：curves_epoch1_500.png；放大第50—500轮：curves_epoch50_500.png。原始数据：epoch_case_metrics.csv.gz；汇总：epoch_summary.csv。
''',encoding='utf-8')
    print(json.dumps(info),flush=True)

if __name__=='__main__':main()

