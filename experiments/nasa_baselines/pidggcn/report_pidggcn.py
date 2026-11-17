"""Report fixed100 current wear, keep printed/corrected variants separate."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).parent/'formal'


def main():
    rows=[];audits=[];prior_rows=[];rate_rows=[];seen_splits=set()
    folders=sorted(ROOT.glob('fold_*/*/seed_*/*'))
    complete=[p for p in folders if (p/'completed.json').exists()]
    assert len(complete)==48,f'Only {len(complete)}/48 completed'
    for folder in complete:
        done=json.loads((folder/'completed.json').read_text());identity=json.loads((folder/'identity.json').read_text())
        a=dict(np.load(folder/'epoch_predictions.npz'))
        assert a['pred_z'].shape==(101,len(a['case']))
        history=pd.read_csv(folder/'history.csv');assert history.epoch.tolist()==list(range(101))
        splitpath=Path(__file__).resolve().parent.parent/'data'/folder.parents[2].name/folder.parents[1].name
        meta=json.loads((splitpath/'metadata.json').read_text());mu,sigma=meta['scalers']['wear']
        splitkey=str(splitpath.relative_to(Path(__file__).resolve().parent.parent/'data'))
        if splitkey not in seen_splits:
            seen_splits.add(splitkey)
            priors=json.loads((folder/'physical_priors.json').read_text())
            with np.load(splitpath/'arrays.npz') as archive:source=dict(archive)
            for group in priors['groups']:
                aa,bb,cc=group['abc'];T=group['time_scale']
                prior_rows.append(dict(split=splitkey,condition=str(group['condition']),A=aa,B=bb,C=cc,time_scale=T,labels=group['group_labels'],training_rmse_mm=group['training_rmse_mm'],fallback_reason=group['fallback_reason'],fit_source_labels=group['n_labels']))
                mask=source['train_mask']&np.all(source['condition_raw']==group['condition'],axis=1)
                for idx in np.flatnonzero(mask):
                    u=source['time'][idx]/T;exp=np.exp(aa+bb*u+cc*u*u)
                    rate_rows.append(dict(split=splitkey,case=int(source['case'][idx]),run=int(source['run'][idx]),time=source['time'][idx],paper_rate_mm_per_time=(1+bb*u+cc*u*u)*exp/T,corrected_rate_mm_per_time=(1+bb*u+2*cc*u*u)*exp/T))
        for key in ('case','run','time','sensor_valid','train_mask','validation_mask','test_mask'):
            with np.load(splitpath/'arrays.npz') as reference:np.testing.assert_array_equal(a[key],reference[key])
        for split in ('train','validation','test'):
            mask=a[split+'_mask']
            assert np.isfinite(a['pred_z'][:,mask]).all()
            if mask.any():
                errors=abs(a['pred_z'][:,mask]*sigma+mu-a['target_mm'][mask]);np.testing.assert_allclose(errors.mean(1),history[split+'_mae_mm'],atol=1e-12)
        audits.append(dict(run=str(folder.relative_to(ROOT)),epochs=done['epochs'],seconds=done['seconds'],initial_hash=done['initial_hash'],reload=done['reload_exact']))
        if folder.parents[1].name!='final':continue
        for i in np.flatnonzero(a['test_mask']):
            rows.append(dict(method='chen2025_pidggcn_causal_'+folder.name,fold=folder.parents[2].name,seed=int(folder.parent.name.split('_')[-1]),epoch=100,split='test',case=int(a['case'][i]),run=int(a['run'][i]),time=a['time'][i],target_mm=a['target_mm'][i],pred_mm=a['pred_z'][-1,i]*sigma+mu,sensor_valid=bool(a['sensor_valid'][i])))
    frame=pd.DataFrame(rows);frame['ae']=abs(frame.pred_mm-frame.target_mm)
    for (method,seed),g in frame.groupby(['method','seed']):assert len(g)==138
    frame.to_csv(ROOT/'current_predictions.csv',index=False)
    c=frame.groupby(['method','fold','seed','case']).ae.mean().reset_index();c.to_csv(ROOT/'case_seed_metrics.csv',index=False)
    c=c.groupby(['method','fold','case']).ae.mean().reset_index()
    summary=[]
    for method,part in frame.groupby('method'):
        for scope in ('overall','fold_a','fold_b','fold_a_in','fold_a_out'):
            g=part
            if scope.startswith('fold_a'):g=g[g.fold=='fold_a']
            if scope=='fold_b':g=g[g.fold=='fold_b']
            if scope=='fold_a_in':g=g[g.target_mm<=.74]
            if scope=='fold_a_out':g=g[g.target_mm>.74]
            mean=g.groupby(['fold','case','seed']).ae.mean().groupby(['fold','case']).mean()
            summary.append(dict(method=method,scope=scope,mae_mm=mean.mean(),cases=len(mean)))
    pd.DataFrame(summary).to_csv(ROOT/'summary.csv',index=False)
    pd.DataFrame(audits).to_csv(ROOT/'run_audit.csv',index=False)
    pd.DataFrame(prior_rows).to_csv(ROOT/'physical_prior_groups.csv',index=False)
    pd.DataFrame(rate_rows).to_csv(ROOT/'printed_vs_corrected_prior_rates.csv',index=False)
    for parent in ROOT.glob('fold_*/*/seed_*'):
        packs=[dict(np.load(parent/v/'epoch_predictions.npz')) for v in ('paper_printed','calculus_corrected')]
        np.testing.assert_array_equal(packs[0]['pred_z'][0],packs[1]['pred_z'][0])
    result=dict(status='passed',completed_runs=len(audits),epochs_each=100,prediction_epochs=101,outer_current_targets_per_method_seed=138,variants_not_test_selected=True,summed_run_seconds=sum(x['seconds'] for x in audits))
    (ROOT/'report_audit.json').write_text(json.dumps(result,indent=2))
    table=pd.DataFrame(summary)
    lines=['# PIDGGCN NASA因果适配结果','',
      '两种导数公式均为事先固定版本，不按测试择优。原文当前监测任务，不支持无未来sensor的未来VB输出；未添加未来预测头。',
      '48次各100轮，14个inner seed0仅诊断，2fold各5seed为final。每case先计算各seed MAE再平均，最后case等权；外层每seed138标签。',
      '原文5000轮缩为100轮；前50轮权重均1，52起每2轮按训练loss比例半衰减。clip_zscore/top12、timestamp、AD对角偏导、prefix特征图和分工况训练prior均是显式NASA适配，详见../FULLTEXT_DESIGN.md。','',
      '| method | scope | MAE mm | cases |','|---|---|---:|---:|']
    for row in table.itertuples():lines.append(f'| {row.method} | {row.scope} | {row.mae_mm:.6f} | {row.cases} |')
    fallback_count=sum(x['fallback_reason'] is not None for x in prior_rows)
    lines+=['',f'训练物理prior共{len(seen_splits)}个split、{len(prior_rows)}工况组，{fallback_count}组按预设规则回退训练pool；逐组见physical_prior_groups.csv，印刷导数与解析导数的真实单位差异见printed_vs_corrected_prior_rates.csv。',
      f'训练加逐epoch评价记录累计{result["summed_run_seconds"]:.2f}秒；当前表各版本固定epoch100，不宣称达到原文5000轮收敛水平。',
      'current_predictions.csv为外层原始预测；case_seed_metrics.csv保留逐case/seed。完整每轮预测在各运行epoch_predictions.npz，逐轮指标history.csv，checkpoint保留优化器及RNG可恢复。',
      '该报告完成基础键、每轮指标反算及共同epoch0检查；仍需独立审计。已反复观察NASA测试，结果属于探索性。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf8')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':main()
