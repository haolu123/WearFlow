"""Audit all epoch archives; aggregate test tools without test-driven selection."""
import json
import time
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_protocol import ROOT,records,pairs
from run_formal import OUT, identity, save_json

def main():
    started=time.time();manifest=pd.read_csv(ROOT/'study_manifest.csv');frame,_,_=records();pp=pairs(frame)
    tables=[];predictions=[];audit=[];elapsed=0.;initial_states={}
    for row in manifest.itertuples():
        folder=OUT/row.split/f'{row.method}_seed{row.seed}'
        done=json.loads((folder/'complete.json').read_text())
        assert done['identity']==identity(row.split,row.method,row.seed)
        elapsed+=done['seconds']
        state=torch.load(folder/'last.pt',weights_only=False)
        assert state['epoch']==100
        if row.method!='time':
            key=(row.split,row.seed)
            if key in initial_states:
                assert all(torch.equal(v,initial_states[key][k]) for k,v in state['initial'].items())
            else:initial_states[key]=state['initial']
        for epoch in range(101):
            path=folder/f'epoch_{epoch:03}.npz'
            with np.load(path) as a:
                assert a['current_mm'].shape==(len(frame),)
                assert a['future_mm'].shape==(len(pp),)
                assert np.isfinite(a['current_mm']).all() and np.isfinite(a['future_mm']).all()
                if row.method in ['local','integral']:assert np.isfinite(a['oracle_mm']).all()
                if epoch==100 and row.split.endswith('final'):
                    tool=int(row.split.split('_')[0][4:]);mask=frame.tool.to_numpy()==tool
                    current=frame.loc[mask,['filename','tool','run','cumulated_tool_contact_time','VB_mm']].copy()
                    current['prediction']=a['current_mm'][mask];current['method']=row.method;current['seed']=row.seed;current['task']='current';current['horizon']=0
                    predictions.append(current)
                    for name in ['future_mm','oracle_mm','linear_mm']:
                        if name not in a or not np.isfinite(a[name]).all():continue
                        m=pp.tool.to_numpy()==tool
                        future=pp.loc[m].copy();future['VB_mm']=frame.VB_mm.to_numpy()[future.target]
                        future['origin_VB_mm']=frame.VB_mm.to_numpy()[future.origin]
                        future['prediction']=a[name][m];future['method']=row.method;future['seed']=row.seed;future['task']=name;future['horizon']=future.requested_seconds
                        predictions.append(future)
            item=json.loads((folder/f'epoch_{epoch:03}.json').read_text())
            assert item['epoch']==epoch
            if row.split.endswith('final'):
                for metric in item['metrics']:
                    if metric['role']=='test':tables.append({'method':row.method,'seed':row.seed,'epoch':epoch,**metric})
        audit.append({'split':row.split,'method':row.method,'seed':int(row.seed),'epoch_archives':101,'initialization_shared':row.method!='time','passed':True})
    for split in pd.read_csv(ROOT/'gpr_manifest.csv').split:
        folder=OUT/split/'gpr';done=json.loads((folder/'complete.json').read_text())
        assert done['identity']==identity(split,'gpr',0);elapsed+=done['seconds']
        if split.endswith('final'):
            tool=int(split.split('_')[0][4:]);m=frame.tool.to_numpy()==tool
            with np.load(folder/'predictions.npz') as a:pred=a['current_mm'][m]
            error=pred-frame.VB_mm.to_numpy()[m]
            tables.append({'method':'gpr','seed':0,'epoch':100,'role':'test','tool':tool,'task':'current','horizon':0,'n':int(m.sum()),'mae':float(abs(error).mean()),'rmse':float(np.sqrt(np.mean(error**2)))})
            current=frame.loc[m,['filename','tool','run','cumulated_tool_contact_time','VB_mm']].copy()
            current['prediction']=pred;current['method']='gpr';current['seed']=0;current['task']='current';current['horizon']=0;predictions.append(current)
    report=ROOT/'reports';report.mkdir(exist_ok=True)
    full=pd.DataFrame(tables);full.to_csv(report/'test_metrics_all_epochs.csv',index=False)
    raw=pd.concat(predictions,ignore_index=True);raw.to_csv(report/'test_predictions_epoch100.csv.gz',index=False)
    final=full[full.epoch==100]
    tools=final.groupby(['method','task','horizon','tool'],as_index=False)[['mae','rmse','n']].mean()
    tools.to_csv(report/'tool_metrics_seed_mean.csv',index=False)
    summary=tools.groupby(['method','task','horizon'],as_index=False).agg(mae=('mae','mean'),rmse_mean=('rmse','mean'),tools=('tool','size'),targets=('n','sum'))
    summary.to_csv(report/'summary.csv',index=False)
    # Tool is the resampling unit; seeds have already been averaged.
    rng=np.random.default_rng(20261111);contrasts=[]
    for task in ['current','future_mm','oracle_mm']:
        for h in ([0] if task=='current' else [240,480,720]):
            pivot=tools[(tools.task==task)&(tools.horizon==h)].pivot(index='tool',columns='method',values='mae')
            for reference in (['local','current','time'] if task!='oracle_mm' else ['local']):
                delta=(pivot['integral']-pivot[reference]).dropna().to_numpy()
                bootstrap=rng.choice(delta,size=(10000,len(delta)),replace=True).mean(axis=1)
                contrasts.append({'task':task,'horizon':h,'contrast':f'integral minus {reference}','tools':len(delta),'mean_difference':delta.mean(),'lower95':np.quantile(bootstrap,.025),'upper95':np.quantile(bootstrap,.975)})
    pd.DataFrame(contrasts).to_csv(report/'paired_tool_bootstrap.csv',index=False)
    # Curves are descriptive fixed-budget results, never epoch-selection evidence.
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for method in ['time','current','local','integral']:
        for ax,task,h in [(axs[0],'current',0),(axs[1],'future_mm',720)]:
            z=full[(full.method==method)&(full.task==task)&(full.horizon==h)].groupby(['epoch','tool']).mae.mean().groupby('epoch').mean()
            ax.plot(z.index,z,label=method)
    for ax,title in zip(axs,['Current VB','Future VB: 720 seconds']):
        ax.set(title=title,xlabel='Epoch',ylabel='Tool-macro MAE (mm)');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(report/'training_curves.png',dpi=180);plt.close(fig)
    # Predeclared examples, not tools chosen for favorable test differences.
    fig,axs=plt.subplots(2,3,figsize=(14,7))
    for column,tool in enumerate([1,5,8]):
        g=frame[frame.tool==tool]
        for ax in axs[:,column]:
            ax.plot(g.cumulated_tool_contact_time,g.VB_mm,color='black',label='Published VB',linewidth=2)
            ax.set(xlabel='Cumulative contact time (s)',ylabel='VB (mm)');ax.grid(alpha=.2)
        for method in ['time','current','gpr','local','integral']:
            z=raw[(raw.tool==tool)&(raw.method==method)&(raw.task=='current')].groupby('cumulated_tool_contact_time').prediction.mean()
            axs[0,column].plot(z.index,z,label=method,alpha=.8)
        for method in ['time','current','local','integral']:
            z=raw[(raw.tool==tool)&(raw.method==method)&(raw.task=='future_mm')&(raw.horizon==720)].groupby('target').prediction.mean()
            axs[1,column].plot(frame.cumulated_tool_contact_time.to_numpy()[z.index.to_numpy(int)],z,label=method)
        axs[0,column].set_title(f'Tool {tool}: current')
        axs[1,column].set_title(f'Tool {tool}: future 720 s')
    axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(report/'example_trajectories.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for method in ['local','integral']:
        z=tools[(tools.method==method)&(tools.task=='future_mm')&(tools.horizon==720)]
        axs[0].plot(z.tool,z.mae,'o-',label=method)
        z=raw[(raw.method==method)&(raw.task=='oracle_mm')&(raw.horizon==720)].groupby(['tool','origin','target'])[['prediction','VB_mm','origin_VB_mm']].mean()
        axs[1].scatter(z.VB_mm-z.origin_VB_mm,z.prediction-z.origin_VB_mm,s=3,alpha=.2,label=method)
    axs[0].set(xlabel='Held-out tool',ylabel='MAE (mm)',title='Future 720 s: tool-level comparison')
    axs[1].set(xlabel='Published label increment (mm)',ylabel='Integrated increment (mm)',title='Published-label initial state, 720 s')
    lo,hi=axs[1].get_xlim();axs[1].plot([lo,hi],[lo,hi],'k--',linewidth=1)
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(report/'mechanism_comparison.png',dpi=180);plt.close(fig)
    text=['# Stiehl 外部验证结果','', '固定末轮 100 epoch；9 折留一刀，五 seed 先在刀具内平均，再计算刀具宏平均。GPR 为每折一次确定性拟合。','', '## 当前 VB','', '| 方法 | MAE (mm) |','|---|---:|']
    for r in summary[summary.task=='current'].itertuples():text.append(f'| {r.method} | {r.mae:.6f} |')
    text+=['','## 未来 VB 与发布标签初值诊断','','| 方法 | 任务 | 请求秒数 | MAE (mm) |','|---|---|---:|---:|']
    for r in summary[summary.task!='current'].itertuples():text.append(f'| {r.method} | {r.task} | {r.horizon} | {r.mae:.6f} |')
    text+=['','current 的 future_mm 是 persistence；linear_mm 使用过去两次因果预测斜率，首点退化为 persistence。time 的未来是在目标时间直接查询，不使用未来 sensor。local/integral 的 future_mm 是从估计初值积分，oracle_mm 从发布标签初值积分。oracle 终点 MAE 与区间增量 MAE 数学相同。', '', '## 机制比较：Integral − Local（负值有利于 Integral）','']
    for r in contrasts:
        if r['contrast']=='integral minus local':text.append(f"- {r['task']}，{r['horizon']} 秒：{r['mean_difference']:+.6f} mm；刀具配对 bootstrap 95% 区间 [{r['lower95']:+.6f}, {r['upper95']:+.6f}]。")
    text+=['','不预设积分监督一定优于局部约束；以配对差异及逐刀结果判断，不按外层测试挑模型、维度或训练轮数。区间包括零时，不能声称九刀上有稳定统计优势。', '', '## 覆盖、预算和边界','',f'216 次网络训练全部 100 epoch，另 18 次 GPR。网络累计运行及逐轮评估、GPR 累计耗时 {elapsed/3600:.2f} 小时（不含准备及最终报告）。', '每个 seed 当前 6,409 点；240/480/720 秒未来分别 6,007/5,626/5,256 对；各方法共同目标一致。逐轮原始预测在 formal/<split>/<method_seed>/epoch_*.npz；目标键与 role 由 causal_targets.csv、forecast_pairs.csv、splits/*.json 唯一确定。', 'GPR 每训练刀按时间均匀最多 128 点，神经网络使用全训练目标，预算不同。时间 MLP 不使用 sensor；其他神经网络共享训练内前 12 维特征。','', '标签含插值且缺少原始实测 mask，结果仅检验发布轨迹。固定切削条件不能验证条件泛化；9 刀、3 台已见机床，机床/批次混杂。Local 约束是固定 sensor 的偏导，不是真实轨迹全导数。两类 ODE 冷启动，无未来监督预训练。', '', '统计与图表源数据：summary.csv、tool_metrics_seed_mean.csv、paired_tool_bootstrap.csv、test_metrics_all_epochs.csv、test_predictions_epoch100.csv.gz。']
    (ROOT/'RESULTS.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    save_json(report/'audit.json',{'passed':True,'runs':audit,'nn_runs':len(audit),'gpr_fits':18,'raw_epoch_archives':len(audit)*101,'report_seconds':time.time()-started})
    save_json(ROOT/'status.json',{'stage':'complete','formal_nn_runs_completed':216,'gpr_completed':18,'audit_passed':True,'report':'RESULTS.md','training_and_epoch_evaluation_hours':elapsed/3600})
    save_json(OUT/'status.json',json.loads((ROOT/'status.json').read_text()))
    print('Report and archive audit complete.',flush=True)

if __name__=='__main__':main()
