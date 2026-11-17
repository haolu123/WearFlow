"""Inner-only epoch selection and paired audits for initial-context experiments."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import write_json
from .run_initial_context import ROOT, METHODS, make_model, build_inputs, initial_to_delta_weights
from .run_sensor_mechanisms import OLD, context, predict_raw
from .diagnose_sensor_learning import md


def main():
    mf=pd.read_csv(ROOT/'study_manifest.csv');assert len(mf)==120 and (~mf.reused).sum()==72
    curve=[];audit=[]
    for r in mf.itertuples():
        d=Path(r.run_dir);h=pd.read_csv(d/'history.csv');complete=json.loads((d/'complete.json').read_text())
        assert h.epoch.tolist()==list(range(1,501)) and complete['reload_verified']
        curve.append(h.assign(name=r.name,fold=r.fold,role=r.role,seed=r.seed,holdout=r.holdout))
        a=np.load(d/'epoch_predictions.npz');assert a['predictions'].shape[0]==500 and np.isfinite(a['predictions']).all()
        ident=json.loads((d/'identity.json').read_text());ids=ident['train_indices']
        assert set(a['case'][ids])==set(ident['split']['train']) and a['valid'][ids].all()
        assert not set(a['case'][ids]).intersection(ident['split']['validation']+ident['split']['outer_test_reserved'])
        audit.append(dict(name=r.name,fold=r.fold,role=r.role,holdout=r.holdout,seed=r.seed,reused=r.reused,
            epochs=500,parameters=complete['parameters'],seconds=complete['seconds'],prediction_values=a['predictions'].size))
    curves=pd.concat(curve,ignore_index=True);curves.to_csv(ROOT/'epoch_metrics.csv.gz',index=False)
    inner=curves[curves.role=='inner'].groupby(['name','fold','epoch']).validation_mae.mean().reset_index()
    chosen=inner.sort_values(['validation_mae','epoch']).groupby(['name','fold']).first().reset_index()
    chosen.to_csv(ROOT/'selected_epochs.csv',index=False)
    # Freeze all epoch choices before aggregating outer results.
    write_json(ROOT/'selection.json',dict(rule='mean inner held-out case MAE, seed0; earlier epoch breaks ties',choices=chosen.to_dict('records')))
    anchors=pd.read_csv(ROOT/'anchor_records.csv');records=[]
    for r in mf[mf.role=='final'].itertuples():
        d=Path(r.run_dir);a=np.load(d/'epoch_predictions.npz');ident=json.loads((d/'identity.json').read_text())
        mask=np.isin(a['case'],ident['split']['outer_test_reserved'])&np.isfinite(a['target'])
        anchor=anchors[(anchors.fold==r.fold)&(anchors.role=='final')&(anchors.seed==r.seed)].set_index('case').anchor_run.to_dict()
        selected=int(chosen[(chosen.name==r.name)&(chosen.fold==r.fold)].epoch.iloc[0])
        for policy,epoch in [('fixed500',500),('inner_epoch',selected)]:
            p=a['predictions'][epoch-1,mask];target=a['target'][mask];case=a['case'][mask].astype(int);run=a['run'][mask].astype(int)
            records.append(pd.DataFrame(dict(name=r.name,policy=policy,epoch=epoch,fold=r.fold,seed=r.seed,case=case,run=run,
                anchor_run=[anchor[c] for c in case],target=target,prediction=p,ae=abs(p-target),se=(p-target)**2)))
    rec=pd.concat(records,ignore_index=True);rec['post_anchor']=rec.run>rec.anchor_run
    rec.to_csv(ROOT/'test_predictions.csv',index=False)
    ref=None
    for key,q in rec.groupby(['name','policy','seed']):
        assert len(q)==138 and not q.duplicated(['case','run']).any()
        labels=q.set_index(['case','run']).target.sort_index()
        if ref is None:ref=labels
        else:pd.testing.assert_series_equal(ref,labels)
    allcase=[]
    for subset,q in [('all',rec),('post_anchor',rec[rec.post_anchor])]:
        c=q.groupby(['name','policy','fold','seed','case']).agg(mae=('ae','mean'),mse=('se','mean'),n=('ae','size')).reset_index();c['subset']=subset;allcase.append(c)
    case=pd.concat(allcase,ignore_index=True);case.to_csv(ROOT/'case_metrics.csv',index=False)
    summary=case.groupby(['name','policy','subset']).agg(mae=('mae','mean'),mse=('mse','mean')).reset_index()
    folds=case.groupby(['name','policy','subset','fold']).mae.mean().unstack().reset_index()
    summary=summary.merge(folds,on=['name','policy','subset']);summary.to_csv(ROOT/'summary.csv',index=False)
    seed=case.groupby(['name','policy','subset','seed']).mae.mean().reset_index();seed.to_csv(ROOT/'seed_metrics.csv',index=False)
    seed.groupby(['name','policy','subset']).mae.agg(['mean','std','min','max']).to_csv(ROOT/'seed_summary.csv')
    contrasts=[]
    for control,treatment in [('current_only','current_initial'),('current_only','current_delta'),('current_only','current_duplicate'),
        ('current_duplicate','current_initial'),('current_duplicate','current_delta'),('current_initial','current_delta'),
        ('ct_baseline','current_initial'),('ct_baseline','current_delta')]:
        for policy in ['fixed500','inner_epoch']:
            for subset in ['all','post_anchor']:
                a=case[(case.name==control)&(case.policy==policy)&(case.subset==subset)].set_index(['fold','seed','case']).mae
                b=case[(case.name==treatment)&(case.policy==policy)&(case.subset==subset)].set_index(['fold','seed','case']).mae
                delta=(b-a).rename('delta_mae').reset_index();delta['control']=control;delta['treatment']=treatment;delta['policy']=policy;delta['subset']=subset;contrasts.append(delta)
    contrasts=pd.concat(contrasts,ignore_index=True);contrasts.to_csv(ROOT/'paired_case_seed_deltas.csv',index=False)
    bycase=contrasts.groupby(['control','treatment','policy','subset','case']).delta_mae.mean().reset_index()
    contrast=bycase.groupby(['control','treatment','policy','subset']).agg(delta_mae=('delta_mae','mean'),improved_cases=('delta_mae',lambda x:int((x<0).sum())),cases=('delta_mae','size')).reset_index()
    contrast.to_csv(ROOT/'contrasts.csv',index=False)
    contrasts.groupby(['control','treatment','policy','subset','seed']).delta_mae.mean().reset_index().to_csv(ROOT/'seed_contrasts.csv',index=False)
    # Independent checkpoint/input audit and identical initialization across augmented arms.
    original=pd.read_csv(OLD/'reused_sensor_manifest.csv');torch.set_num_threads(1)
    maxreload=maxequiv=0.
    for r in original.itertuples():
        c=context(r);xs,anchor=build_inputs(c)
        group=mf[(mf.fold==r.fold)&(mf.role==r.role)&(mf.seed==r.seed)]
        group=group[group.holdout.isna()] if pd.isna(r.holdout) else group[group.holdout==r.holdout]
        assert len(group)==5;initial_ref=None
        for row in group.itertuples():
            d=Path(row.run_dir);ident=json.loads((d/'identity.json').read_text())
            assert ident['train_indices']==c['train'].tolist() and ident['split']==c['split']
            if row.reused:continue
            init=torch.load(d/'initial.pt',weights_only=True)
            if initial_ref is None:initial_ref=init
            else:
                for k in init:assert torch.equal(init[k],initial_ref[k])
            model=make_model(int(r.seed));model.load_state_dict(torch.load(d/'last.pt',weights_only=True))
            archive=np.load(d/'epoch_predictions.npz');np.testing.assert_array_equal(archive['anchor'],anchor)
            error=float(abs(predict_raw(model,xs[row.name],c)-archive['predictions'][-1]).max());maxreload=max(maxreload,error);assert error<2e-6
            if row.name=='current_initial':
                mapped=initial_to_delta_weights(model)
                with torch.no_grad():error=float((model(xs['current_initial'])-mapped(xs['current_delta'])).abs().max())
                maxequiv=max(maxequiv,error);assert error<1e-5
    pd.DataFrame(audit).to_csv(ROOT/'run_audit.csv',index=False)
    post_count=int(rec[(rec.name=='current_only')&(rec.policy=='fixed500')&(rec.seed==0)].post_anchor.sum())
    write_json(ROOT/'audit.json',dict(passed=True,new_runs=72,reused_runs=48,new_epochs=36000,
        new_prediction_values=sum(x['prediction_values'] for x in audit if not x['reused']),paired_outer_targets=138,post_anchor_targets=post_count,
        input_audit_passed=json.loads((ROOT/'input_audit.json').read_text())['passed'],shared_training_population=True,
        identical_initial_weights_across_new_arms=True,checkpoint_reload_max_error=maxreload,trained_function_reparameterization_max_error=maxequiv,
        note='Post-anchor is a secondary subgroup on the same already-observed outer cases, not an independent test set.'))
    labels={'ct_baseline':'Condition + time','current_only':'Current only','current_initial':'Current + initial','current_delta':'Current + change','current_duplicate':'Current duplicated'}
    colors={'ct_baseline':'black','current_only':'tab:blue','current_initial':'tab:orange','current_delta':'tab:green','current_duplicate':'tab:purple'}
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,fold in zip(axs,['fold_a','fold_b']):
        for name in labels:
            q=inner[(inner.name==name)&(inner.fold==fold)];ax.plot(q.epoch,q.validation_mae,label=labels[name],color=colors[name])
        ax.set(title=fold+' inner validation',xlabel='Epoch',ylabel='Case-macro MAE (mm)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(ROOT/'inner_learning_curves.png',dpi=160);plt.close(fig)
    maincase=case[(case.policy=='fixed500')&(case.subset=='all')].groupby(['case','name']).mae.mean().unstack()
    maincase=maincase[list(labels)].rename(columns=labels);maincase.to_csv(ROOT/'case_mae_readout.csv')
    fig,ax=plt.subplots(figsize=(14,5),layout='constrained');maincase.plot.bar(ax=ax,color=list(colors.values()))
    ax.set(title='Fixed epoch 500, mean of five seeds',ylabel='MAE (mm)');ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'case_mae.png',dpi=160);plt.close(fig)
    main=summary[(summary.policy=='fixed500')&(summary.subset=='all')]
    secondary=summary[(summary.policy=='inner_epoch')&(summary.subset=='all')]
    body='''# 当前特征＋初始特征 / 当前特征＋变化量

两种请求的输入及同维度重复输入对照已完成。72次新增训练（每种24次），复用24次仅当前sensor和24次工况时间模型。每次500轮；每fold七个inner holdout使用seed0，final使用seed0–4。

## 结论

本次两种追加表示均没有改善总体外层MAE。固定500轮：仅当前特征0.09188，当前＋初始0.13528，当前＋变化量0.11230；同维度重复当前特征0.09158，基本不变。两种追加表示在五个seed的总体配对差值均为退化，因此不能只归因于一次不利初始化。

退化主要集中在fold B：当前＋初始0.17557、当前＋变化量0.13043，对照仅当前0.07882；fold A反而由0.10495改善到约0.0950/0.0942。输入的效用仍明显依赖训练/测试case组合。

较严重例子为case1：仅当前0.09358、加初始0.37465、加变化0.23704；case3分别为0.06921、0.19943、0.14873。也有改善case，例如case9和11。当前＋初始改善4/14个case，当前＋变化量改善5/14个case。

学习曲线显示追加表示在后期有明显验证退化。按内层选择epoch后，两种表示的外层MAE降到0.09919/0.09972，与同样按内层选epoch的仅当前0.09917接近，仍未显示总体优势；无sensor基线0.08267。不能跨轮数政策把数值混为同一比较。

fold B训练MAE由仅当前0.01837降至约0.01486/0.01490，而外层明显上升，符合过拟合迹象。初始特征可能让模型依赖case特有背景，但尚未通过独立干预证明其充当case标识。去掉参考记录后的127个测试目标上结论不变，不是只由初始记录本身驱动。

当前建议不把这两个简单拼接版本作为主模型改进，也不根据已看的测试case继续扩参寻找有利结果。

## 固定500轮主结果

MAE为14个case等权、五seed平均，单位mm；MSE另列。所有选择只用内层验证，主结果固定500轮。

'''+md(main)+'''

## 内层选择epoch的次要结果

'''+md(secondary)+'''\n\n'''+md(chosen)+'''

## 去掉参考记录后的次要结果

参考记录本身的delta为0；为避免结果仅由这些记录驱动，另统计首条有效sensor之后的同一测试记录子集。该子集不是新独立测试集。

'''+md(summary[(summary.policy=='fixed500')&(summary.subset=='post_anchor')])+'''

## 成对比较

delta_mae为treatment减control，负数表示改善。改善case数按五seed平均后计数，不把seed当作独立case。

'''+md(contrast[(contrast.policy=='fixed500')&(contrast.subset=='all')])+'''

## 输入和初始化协议

- 当前18维特征名单和clip1%/99%＋z-score沿用各split训练内结果。初始值使用同一变换，变化量定义为current_z−initial_z，不额外裁剪或重新缩放差分，以保留两种表示的线性可逆关系。
- 初始参考为该case首条质量有效的sensor记录，不要求VB已知。参考不可供更早记录使用，后续值不能改变初始参考。不输入初始VB、case编号或未来轨迹统计。
- 三种新增模型输入均41维：18当前＋18附加＋4工况＋1时间，隐藏层64/32，4801参数；同split/seed起始权重逐张量完全相同。仅当前sensor为3649参数；工况时间2497参数。
- current_duplicate附加当前特征的复制，无新信息；它控制输入维度/名义参数量，但冗余参数化仍可改变有效正则和优化，不能将其看作完全相同的优化过程。
- AdamW学习率0.001、weight decay 0.0001，点平均标准化VB的MSE。CPU单线程确定性训练。所有训练标签、后备模型和测试目标配对一致。
- current_only复用上一轮sensor_standard（约0.09188），与更早0.08950的sensor模型随机数调用顺序不同；旧顺序已在上一轮独立重放复现，不能混作同一组初始权重。

## 两种表示包含相同信息

对于x=current、x0=initial、d=x−x0，x0=x−d。第一层在[x,x0]上的权重[A,B]可改写为[x,d]上的[A+B,−B]。已对初始和训练后网络验证函数一致性。因此两种表示若成绩不同，不能解释为某一种包含更多信息；同数值初始权重对应的初始函数不同，AdamW及权重衰减也不对这种坐标变化保持不变。

## 保存、审计与限制

保存36,000个新增epoch的逐轮指标和逐记录预测、initial.pt与last.pt；没有声称保存每轮checkpoint。审计包括：训练/验证case分离、138个外层目标键与标签一致、首条参考因果可用性、无VB输入、相同训练掩码、三种新增模型相同初始化权重、checkpoint重载、两种输入的函数等价重参数化。另有三项单元测试和3模型×3轮短运行。

首条sensor不等于零磨损，case2/12从run2取参考，case13/14/16参考记录无VB标签；这些都不妨碍输入构造，也没有把其VB设为0。测试此前已查看，结果为探索性；不按测试重新选择特征、epoch或模型网格。

![逐case误差](case_mae.png)

![内层学习曲线](inner_learning_curves.png)

```text
python -m wear_monitoring_dynamics.scripts.run_initial_context
python -m wear_monitoring_dynamics.scripts.summarize_initial_context
python -m unittest wear_monitoring_dynamics.tests.test_initial_context -v
```

[完整汇总](summary.csv) · [逐case配对差值](paired_case_seed_deltas.csv) · [五seed结果](seed_metrics.csv) · [初始记录](anchor_records.csv) · [审计](audit.json)
'''
    (ROOT/'RESULTS.md').write_text(body,encoding='utf-8')
    write_json(ROOT/'status.json',dict(status='completed_and_audited',new_runs=72,reused_runs=48,epochs=500))
    print(main.to_string(index=False));print(secondary.to_string(index=False));print(contrast[(contrast.policy=='fixed500')&(contrast.subset=='all')].to_string(index=False))


if __name__=='__main__':main()
