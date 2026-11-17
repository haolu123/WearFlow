"""Evaluate preregistered mechanism controls, selecting only on inner cases."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import write_json
from .diagnose_sensor_learning import ROOT, md
from .run_sensor_mechanisms import MODELS, ALPHAS

OUT=ROOT/'step23_controlled'


def macro_mae(pred,target,cases):
    return float(np.mean([np.mean(abs(pred[cases==c]-target[cases==c])) for c in np.unique(cases)]))


def main():
    mf=pd.read_csv(OUT/'study_manifest.csv');assert len(mf)==264
    inner=[];curves=[];audit=[]
    for r in mf.itertuples():
        d=Path(r.run_dir);ident=json.loads((d/'identity.json').read_text());complete=json.loads((d/'complete.json').read_text())
        h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,501))
        assert complete['reload_verified'] and complete['epochs']==500
        a=np.load(d/'epoch_predictions.npz');p=a['predictions'];assert p.shape[0]==500 and np.isfinite(p).all()
        traincases=set(a['case'][ident['train_indices']].tolist())
        assert traincases==set(ident['split']['train'])
        assert all(a['valid'][ident['train_indices']])
        assert not traincases.intersection(ident['split']['validation']+ident['split']['outer_test_reserved'])
        audit.append(dict(name=r.name,fold=r.fold,role=r.role,seed=r.seed,holdout=r.holdout,parameters=complete['parameters'],seconds=complete['seconds'],rows=p.size))
        if r.role=='inner':
            curves.append(h.assign(name=r.name,fold=r.fold,holdout=r.holdout))
            mask=(a['case']==r.holdout)&np.isfinite(a['target'])
            for alpha in (ALPHAS if MODELS[r.name]['kind']=='residual' else [1.]):
                v=a['base']+alpha*(p[-1]-a['base']) if len(a['base']) else p[-1]
                inner.append(dict(name=r.name,fold=r.fold,holdout=r.holdout,alpha=alpha,mae=macro_mae(v[mask],a['target'][mask],a['case'][mask])))
    inner=pd.DataFrame(inner);inner.to_csv(OUT/'inner_alpha_scores_by_case.csv',index=False)
    scores=inner.groupby(['name','fold','alpha']).mae.mean().reset_index()
    chosen=scores.sort_values(['mae','alpha']).groupby(['name','fold']).first().reset_index()
    chosen.to_csv(OUT/'selected_alphas.csv',index=False)
    allcurves=pd.concat(curves,ignore_index=True);allcurves.to_csv(OUT/'inner_learning_curves.csv.gz',index=False)
    cv=allcurves.groupby(['name','fold','epoch']).validation_mae.mean().reset_index()
    ep=cv.sort_values(['validation_mae','epoch']).groupby(['name','fold']).first().reset_index()
    ep.to_csv(OUT/'inner_selected_epochs.csv',index=False)
    # Selection is now frozen before any final test aggregation.
    write_json(OUT/'selection.json',dict(epochs=ep.to_dict('records'),alphas=chosen.to_dict('records'),rule='inner cases only'))
    records=[]
    for r in mf[mf.role=='final'].itertuples():
        d=Path(r.run_dir);ident=json.loads((d/'identity.json').read_text());a=np.load(d/'epoch_predictions.npz')
        mask=np.isin(a['case'],ident['split']['outer_test_reserved'])&np.isfinite(a['target'])
        parameters=chosen[(chosen.name==r.name)&(chosen.fold==r.fold)].iloc[0]
        if MODELS[r.name]['kind']=='residual':
            choices=[('selected_alpha',500,float(parameters.alpha))]+[(f'alpha_{alpha:g}',500,alpha) for alpha in ALPHAS]
        else:
            epoch=int(ep[(ep.name==r.name)&(ep.fold==r.fold)].epoch.iloc[0])
            choices=[('fixed500',500,1.),('inner_epoch',epoch,1.)]
        for policy,epoch,alpha in choices:
            p=a['predictions'][epoch-1]
            if len(a['base']):p=a['base']+alpha*(p-a['base'])
            pred=p[mask];target=a['target'][mask]
            records.append(pd.DataFrame(dict(name=r.name,policy=policy,epoch=epoch,alpha=alpha,fold=r.fold,seed=r.seed,
                    case=a['case'][mask].astype(int),run=a['run'][mask].astype(int),valid=a['valid'][mask],
                    target=target,prediction=pred,ae=abs(pred-target),se=(pred-target)**2)))
    rec=pd.concat(records,ignore_index=True);rec.to_csv(OUT/'test_predictions.csv',index=False)
    ref=None
    for key,q in rec.groupby(['name','policy','seed']):
        assert len(q)==138 and not q.duplicated(['case','run']).any()
        labels=q.set_index(['case','run']).target.sort_index()
        if ref is None:ref=labels
        else:pd.testing.assert_series_equal(ref,labels)
    case=rec.groupby(['name','policy','fold','seed','case']).agg(mae=('ae','mean'),mse=('se','mean'),n=('ae','size')).reset_index()
    case.to_csv(OUT/'case_metrics.csv',index=False)
    seed=case[case.policy.isin(['fixed500','selected_alpha'])].groupby(['name','seed']).mae.mean().reset_index()
    seed.to_csv(OUT/'seed_metrics.csv',index=False)
    seed.groupby('name').mae.agg(['mean','std','min','max']).to_csv(OUT/'seed_summary.csv')
    summary=case.groupby(['name','policy']).agg(mae=('mae','mean'),mse=('mse','mean')).reset_index()
    folds=case.groupby(['name','policy','fold']).mae.mean().unstack('fold').reset_index()
    summary=summary.merge(folds,on=['name','policy']);summary.to_csv(OUT/'summary.csv',index=False)
    valid=rec[rec.valid].groupby(['name','policy','fold','seed','case']).agg(mae=('ae','mean'),mse=('se','mean')).reset_index()
    valid.groupby(['name','policy']).agg(mae=('mae','mean'),mse=('mse','mean')).reset_index().to_csv(OUT/'valid_only_summary.csv',index=False)
    pairs=[('ct_small','sensor_small'),('ct_standard','sensor_standard'),('ct_standard','sensor_matched'),
           ('ct_continue','sensor_warm'),('sensor_standard','sensor_warm'),('ct_standard','ct_continue')]
    deltas=[]
    for control,treatment in pairs:
        for policy in ['fixed500','inner_epoch']:
            a=case[(case.name==control)&(case.policy==policy)].set_index(['fold','seed','case']).mae
            b=case[(case.name==treatment)&(case.policy==policy)].set_index(['fold','seed','case']).mae
            delta=(b-a).rename('delta').reset_index()
            delta['control']=control;delta['treatment']=treatment;delta['policy']=policy;deltas.append(delta)
    for control,treatment in [('resid_ct_in','resid_sensor_in'),('resid_ct_oof','resid_sensor_oof'),
                              ('ct_standard','resid_sensor_in'),('ct_standard','resid_sensor_oof')]:
        a=case[(case.name==control)&(case.policy==('fixed500' if control=='ct_standard' else 'selected_alpha'))].set_index(['fold','seed','case']).mae
        b=case[(case.name==treatment)&(case.policy=='selected_alpha')].set_index(['fold','seed','case']).mae
        delta=(b-a).rename('delta').reset_index();delta['control']=control;delta['treatment']=treatment;delta['policy']='selected_alpha';deltas.append(delta)
    deltas=pd.concat(deltas,ignore_index=True);deltas.to_csv(OUT/'paired_deltas_by_case_seed.csv',index=False)
    deltas[deltas.policy.isin(['fixed500','selected_alpha'])].groupby(['control','treatment','policy','seed']).delta.mean().reset_index().to_csv(OUT/'seed_contrasts.csv',index=False)
    bycase=deltas.groupby(['control','treatment','policy','case']).delta.mean().reset_index()
    contrast=bycase.groupby(['control','treatment','policy']).agg(delta_mae=('delta','mean'),cases_improved=('delta',lambda x:int((x<0).sum())),cases=('delta','size')).reset_index()
    contrast.to_csv(OUT/'contrasts.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'run_audit.csv',index=False)
    # Audit shared populations, function-preserving starts, frozen alpha=0 and OOF provenance.
    for keys,g in mf.groupby(['fold','role','holdout','seed'],dropna=False):
        population=None;base_pred=None
        for r in g.itertuples():
            d=Path(r.run_dir);ids=json.loads((d/'identity.json').read_text())['train_indices']
            if population is None:population=ids
            else:assert ids==population
            if r.name=='ct_standard':base_pred=np.load(d/'epoch_predictions.npz')['predictions'][-1]
        assert base_pred is not None
        for r in g.itertuples():
            if MODELS[r.name]['kind']!='scratch':
                assert np.max(abs(np.load(Path(r.run_dir)/'epoch0.npy')-base_pred))<2e-6
    cfrefs=0
    for p in (OUT/'oof_targets').rglob('audit.json'):
        for v in json.loads(p.read_text()):
            assert v['holdout'] not in v['train_cases']
            s=json.loads((Path(v['directory'])/'scalers.json').read_text())
            assert s['train_cases']==v['train_cases'];cfrefs+=1
    for r in mf[mf.name.str.startswith('resid_') & (mf.role=='final')].itertuples():
        q=rec[(rec.name==r.name)&(rec.policy=='alpha_0')&(rec.fold==r.fold)&(rec.seed==r.seed)].sort_values(['case','run'])
        b=rec[(rec.name=='ct_standard')&(rec.policy=='fixed500')&(rec.fold==r.fold)&(rec.seed==r.seed)].sort_values(['case','run'])
        np.testing.assert_allclose(q.prediction,b.prediction,atol=1e-7)
    unique_cf=len(list((OUT/'crossfit_baselines').glob('*/complete.json')))
    write_json(OUT/'audit.json',dict(passed=True,primary_runs=len(mf),epochs=len(mf)*500,per_epoch_prediction_values=sum(v['rows'] for v in audit),
        crossfit_model_references=cfrefs,unique_crossfit_runs=unique_cf,paired_outer_targets=138,
        baseline_identity=True,shared_training_population=True,alpha_zero_identity=True,checkpoint_reload=True))
    main=summary[(summary.policy=='fixed500') | (summary.policy=='selected_alpha')]
    fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
    ax.bar(main.name,main.mae);ax.set(ylabel='Case-macro MAE (mm)',title='Fixed epoch 500; residual alpha selected by inner validation')
    ax.tick_params(axis='x',rotation=40);ax.grid(axis='y',alpha=.2);fig.savefig(OUT/'primary_mae.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,fold in zip(axs,['fold_a','fold_b']):
        for name in ['ct_standard','sensor_standard','ct_continue','sensor_warm']:
            z=cv[(cv.fold==fold)&(cv.name==name)];ax.plot(z.epoch,z.validation_mae,label=name)
        ax.set(title=fold+' inner validation',xlabel='Epoch in this stage',ylabel='Case-macro MAE (mm)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'warm_start_curves.png',dpi=150);plt.close(fig)
    report='# 第二、三步：容量、初始化与冻结残差对照\n\n'
    report+='264次主训练，每次500轮；另有'+str(unique_cf)+'次唯一的嵌套交叉预测基线训练。CPU单线程确定性执行。先冻结选择规则，再汇总外层结果。\n\n'
    report+='## 主结果\n\n'+md(main)+'\n\n## 成对对照\n\n负delta表示treatment更好；case数按五seed平均后计数，不把seed当独立case。\n\n'+md(contrast)
    report+='\n\n## 内层选出的残差收缩强度\n\n'+md(chosen[chosen.name.str.startswith('resid_')])
    report+='\n\n## 次要结果：仅内层选择epoch\n\n'+md(summary[summary.policy=='inner_epoch'])
    report+='\n\n## 解释边界\n\n所有新模型使用相同有效sensor训练标签；无效sensor记录统一回退至原冻结ct。逐记录核对发现实际监督标签与旧研究没有差异。新旧sensor的初始化顺序不同：旧训练器先随机构造fallback网络再加载权重，消耗了一段随机数；本轮每个预测器直接以seed初始化。初始化审计重放旧顺序后，十次500轮CPU重训与原GPU预测最大差约6.6e-7 mm，复现原总体MAE0.08950。因此旧sensor与本轮sensor_standard的数值差别不能说成硬件误差，也不能混作同一组模型。详见../initialization_audit/。\n\nct_standard为2497参数，sensor_matched为2501参数，匹配不是结构完全相同。small为32/16，standard为64/32。\n\nct_continue与sensor_warm均从本轮ct_standard的500轮模型开始，再训练500轮，均重置AdamW。二者比较控制了预训练和额外预算；sensor_warm对sensor_standard的比较同时改变初始化和总预算，不能独立归因于初始化。\n\n残差分支32/16、末层零初始化，冻结基线。in用当前训练拟合残差；oof用训练case内部再留一case预测得到的残差，c/t/VB缩放在子训练集重拟合。两个版本各含ct-only残差分支作为容量对照。外推时oof修正叠加在完整训练基线上，存在子模型与完整模型误差分布不同的限制。alpha只从0/.25/.5/1按内层MAE选，固定500轮；alpha=0精确返回本轮基线。选择包含零不保证外层非劣。\n\nOOF只指基线目标的交叉预测；残差学习器的18维名单由当前完整训练split选出，最终以该split的独立留出case评估，不把训练残差拟合成绩当无偏性能。\n\n保存全部逐epoch指标和逐记录预测、epoch0预测及末轮checkpoint。本轮没有保存每轮checkpoint；可凭预测数组复算任何epoch的指标。主指标为case平均MAE，MSE另列。学习目标仍为点平均MSE，未声称隔离了损失权重影响。原始信号物理对齐未新验证；未增加测试驱动的参数搜索。两个fold且测试已查看，全部结论为探索性。\n'
    (OUT/'RESULTS.md').write_text(report,encoding='utf-8')
    write_json(OUT/'status.json',dict(status='completed_and_audited',runs=len(mf),epochs=500,crossfit_runs=unique_cf))
    print(main.to_string(index=False));print(contrast.to_string(index=False));print(chosen[chosen.name.str.startswith('resid_')].to_string(index=False))


if __name__=='__main__':main()
