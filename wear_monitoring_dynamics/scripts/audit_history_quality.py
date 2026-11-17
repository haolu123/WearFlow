"""Audit quality-skip training and create matched, descriptive comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..config import project_path, write_json, digest
from ..data.records import load_table
from ..data.preprocessing import fit_scalers
from ..data.sensor_quality import apply_quality
from ..data.monitor_batch import make_monitor_batch
from ..data.splits import make_split
from .run_train_only_coverage import FEATURES, rank_features
from .run_history_phase13 import load_monitor

ROOT=project_path('wear_monitoring_dynamics/outputs/history_quality_skip_20261008')
OLD=ROOT.with_name('history_phase13_20261008')

def markdown(frame):
    lines=['| '+' | '.join(frame.columns)+' |','|'+'|'.join(['---']*len(frame.columns))+'|']
    for row in frame.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.5f}' if isinstance(v,(float,np.floating)) else str(v) for v in row)+' |')
    return '\n'.join(lines)

def main():
    torch.set_num_threads(1)
    m=pd.read_csv(ROOT/'study_manifest.csv');assert len(m)==322 and m.id.nunique()==322
    cfg=json.loads((Path(m.iloc[0].run_dir)/'provenance.json').read_text())['config']
    table,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))
    rank_cache={};audits=[];starts=[];ends=[];coverage=[]
    for r in m.itertuples():
        d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config']
        split=make_split(r.fold,r.role,None if pd.isna(r.holdout) else int(r.holdout))
        assert split==json.loads((d/'split.json').read_text())
        qt=apply_quality(table,cfg['data']['sensor_quality'])
        assert cfg['data']['sensor_quality']['rules']['train_cases']==split['train']
        scaler=json.loads((d/'scalers.json').read_text())
        assert digest(fit_scalers(qt,split['train']))==digest(scaler)
        if r.type=='monitor' and r.name!='ct':
            key=tuple(split['train'])
            if key not in rank_cache:
                f=pd.DataFrame({k:v for k,v in qt.items() if k!='sensor'});f[FEATURES]=qt['sensor']
                rank_cache[key]=rank_features(f[f.case.isin(key)&f.sensor_valid]).head(18).feature.tolist()
            assert cfg['monitor']['selected_features']==rank_cache[key]
            fallback=Path(cfg['monitor']['fallback_checkpoint'])
            fc=json.loads((fallback.parent/'provenance.json').read_text())['config']
            assert fc['experiment']['seed']==r.seed and fc['split']==cfg['split'] if 'split' in cfg else fc['experiment']['seed']==r.seed
            batch=make_monitor_batch(cfg,qt,scaler,'cpu')
            ix=batch['history_indices'].numpy();mask=batch['history_mask'].numpy()
            for i in range(len(qt['case'])):
                past=ix[i][mask[i]]
                assert qt['sensor_valid'][past].all()
                assert (qt['case'][past]==qt['case'][i]).all() and (qt['time'][past]<=qt['time'][i]).all()
                usable=np.flatnonzero((qt['case']==qt['case'][i])&qt['sensor_valid'])
                end=np.searchsorted(usable,i,side='right');start=max(0,end-cfg['monitor']['history_length'])
                np.testing.assert_array_equal(past,usable[start:end])
                gaps=np.r_[0.,np.diff(qt['time'][usable])]/scaler['time_scale']
                np.testing.assert_allclose(batch['history_delta'][i,:,0].numpy()[mask[i]],gaps[start:end],rtol=1e-6)
            if r.role=='final':
                pred=load_monitor(d,table,torch.device('cpu'))
                ct=load_monitor(fallback.parent,table,torch.device('cpu'))
                np.testing.assert_allclose(pred[~qt['sensor_valid']],ct[~qt['sensor_valid']],atol=1e-6)
        h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,501))
        assert h.filter(regex='^(train|validation|test_diagnostic)_').notna().all().all()
        metrics=json.loads((d/'metrics.json').read_text());assert metrics['epochs']==500 and metrics['reload_verified']
        if r.type=='monitor':assert metrics['selected_epoch']==500 and metrics['selected_checkpoint']=='last.pt'
        if r.type=='monitor' and r.role=='final':assert len(list((d/'epochs').glob('epoch_*.pt')))==500
        p=pd.read_csv(d/'epoch_predictions.csv.gz');assert np.isfinite(p[['target_vb_mm','predicted_vb_mm']]).all().all()
        for (group,task),q in p.groupby(['split','task']):
            assert q.epoch.nunique()==500 and q.groupby('epoch').size().nunique()==1
            expected=split['outer_test_reserved'] if group=='test_diagnostic' else split[group]
            assert set(q.case)==set(expected)
        if r.type=='monitor':
            group='validation' if r.role=='inner' else 'test_diagnostic'
            p500=p[(p.epoch==500)&(p.split==group)&(p.task=='monitoring')]
            actual=abs(p500.predicted_vb_mm-p500.target_vb_mm).groupby(p500.case).mean().mean()
            assert abs(actual-metrics['monitoring']['case_macro_mae'])<1e-6
        audits.append(dict(id=r.id,prediction_rows=len(p),seconds=metrics['wall_seconds']))
        starts.append((d/'config_resolved.yaml').stat().st_mtime);ends.append((d/'status.json').stat().st_mtime)
    audit=pd.DataFrame(audits);audit.to_csv(ROOT/'run_audit.csv',index=False)
    selected=json.loads((ROOT/'history_selection.json').read_text());scores=pd.read_csv(ROOT/'inner_validation_scores.csv')
    for fold,name in selected.items():
        q=scores[(scores.fold==fold)&scores.name.str.endswith(('_L3','_L5','_L10'))]
        assert name==q.sort_values(['mae','name']).iloc[0]['name']
    current=pd.read_csv(ROOT/'phase1_test_predictions.csv');previous=pd.read_csv(OLD/'phase1_test_predictions.csv')
    keys=['name','fold','seed','case','run']
    matched=current.merge(previous,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(matched)==len(current)==len(previous)
    np.testing.assert_array_equal(matched.vb_mm_new,matched.vb_mm_old)
    assert len(current[['fold','case','run']].drop_duplicates())==138
    matched.to_csv(ROOT/'matched_test_predictions.csv.gz',index=False)
    summaries=[]
    for root,label in [(OLD,'old'),(ROOT,'new')]:
        ss=json.loads((root/'history_selection.json').read_text())
        pp=pd.read_csv(root/'phase1_test_predictions.csv')
        chosen=pd.concat([pp[(pp.fold==fold)&(pp.name==name)] for fold,name in ss.items()])
        for subset,mask in dict(overall=np.ones(len(chosen),bool),fold_a=chosen.fold=='fold_a',fold_b=chosen.fold=='fold_b',fold_a_in_range=(chosen.fold=='fold_a')&(chosen.vb_mm<=.74),fold_a_out_of_range=(chosen.fold=='fold_a')&(chosen.vb_mm>.74)).items():
            summaries.append(dict(version=label,subset=subset,mae=chosen[mask].groupby(['fold','seed','case']).absolute_error.mean().mean()))
    sel=pd.DataFrame(summaries);sel.to_csv(ROOT/'selected_history_comparison.csv',index=False)
    old=pd.read_csv(OLD/'phase1_summary.csv');new=pd.read_csv(ROOT/'phase1_summary.csv')
    comp=old.merge(new,on=['name','subset'],suffixes=('_old','_new'));comp['delta_mae']=comp.mae_new-comp.mae_old
    comp.to_csv(ROOT/'phase1_before_after.csv',index=False)
    case_old=pd.read_csv(OLD/'phase1_case_metrics.csv');case_new=pd.read_csv(ROOT/'phase1_case_metrics.csv')
    case_new.merge(case_old,on=['name','fold','seed','case'],suffixes=('_new','_old'),validate='one_to_one').to_csv(ROOT/'phase1_case_before_after.csv',index=False)
    qc=pd.read_csv(ROOT/'quality_decisions.csv');sens=pd.read_csv(ROOT/'quality_threshold_sensitivity.csv')
    assert qc.groupby(['fold','role','holdout'],dropna=False).ngroups==16
    for _,q in qc.groupby(['fold','role','holdout'],dropna=False):assert set(map(tuple,q[~q.sensor_valid][['case','run']].values))=={(2,1),(12,1)}
    for _,q in sens.groupby(['fold','role','holdout','multiplier'],dropna=False):assert set(map(tuple,q[~q.sensor_valid][['case','run']].values))=={(2,1),(12,1)}
    for fold in selected:
        q=qc[(qc.fold==fold)&(qc.role=='final')]
        test=make_split(fold,'final')['test'];q=q[q.case.isin(test)]
        retained=q[~((q.case==2)&(q.run==1))]
        coverage.append(dict(fold=fold,retained_records=len(retained),valid_sensor=int(retained.sensor_valid.sum()),fallback_records=int((~retained.sensor_valid).sum()),labeled_fallback=0))
    cv=pd.DataFrame(coverage);cv.to_csv(ROOT/'quality_coverage.csv',index=False)
    p3=[]
    for root,label in [(OLD,'old'),(ROOT,'new')]:
        q=pd.read_csv(root/'phase3_summary.csv');ss=json.loads((root/'history_selection.json').read_text())
        q['method']=q.apply(lambda r:'selected_history' if r['method']==ss[r['fold']] else r['method'],axis=1)
        p3.append(q.rename(columns={'mae':'mae_'+label}))
    dynamics=p3[0].merge(p3[1],on=['fold','method','sigma','start_kind','task'],validate='one_to_one')
    dynamics.to_csv(ROOT/'phase3_before_after.csv',index=False)
    lines=['# 跳过无效 sensor：完成结果','',
        '主指标为 case-macro MAE（mm），越小越好；固定第500轮，各最终模型五个seed。结论仅针对本批Phase1/3。',
        '','## 主要结论','',
        '质量处理消除了历史MLP/CNN由异常输入引起的极端误差，但没有证明历史sensor稳定优于当前sensor或工况＋时间基线。MLP_L3整体0.09435，优于自身L1的0.09673；fold A改善，fold B略差。GRU历史在fold B改善但fold A未改善。无sensor基线整体0.08349，仍优于本批所有固定sensor结构。',
        '内层选择改变为fold A GRU_L3、fold B MLP_L3。选择流程的前后差异同时包含输入处理和结构选择变化，不能归因于某个固定网络。',
        '','## 当前VB：固定结构修复前后','',
        markdown(comp[['name','subset','mae_old','mae_new','delta_mae']]),
        '','### 内层选出的历史模型','',markdown(sel),
        '','## 未来VB与完整递推','',
        '下表selected_history在前后各自遵循内层选择，其他方法含义不变；forecast汇总h1..3，逐horizon见phase3_forecast_horizons.csv。',
        '',markdown(dynamics[['task','fold','method','sigma','start_kind','mae_old','mae_new']]),
        '', '首记录初始化的当前MLP在fold A从8.57101降至0.22650，严重异常已消失；这一改进包括跳过坏窗口、重拟合scaler/特征和冻结ct后备的整体作用，不能只归于某一个机制。',
        'oracle与噪声初值结果不变，说明ODE训练协议保持一致。oracle整条开环递推fold A仍0.22101、fold B仍0.08895；输入修复没有解决动力学分布差异。真实初值不保证每条轨迹上误差最小，估计初值可能偶然补偿ODE偏差，不能由部分sensor递推优于oracle推断sensor更准确。',
        'first_record和first_labeled的起点及目标集合不同，不能直接用其差值量化纯漂移。ODE随后不读取sensor，也不以真实VB重置；初始点不计误差。16次噪声重复先平均，不作为独立case。',
        '','## 质量覆盖与审计','',markdown(cv),
        '', '16个划分及系数4/6/8均仅标记case2/run1和case12/run1。case2/run1沿用既有排除；新增跳过的是无标签case12/run1。138个外层有标签点与旧实验逐点一致，未删除困难测试标签。fold A首记录case12由ct后备初始化。',
        f'322次训练全部500epoch，共161000epoch，逐轮预测共{int(audit.prediction_rows.sum())}行。校验逐epoch指标/预测有限、训练case划分、质量规则训练case、过滤后排名/scaler、有效因果窗口/实际间隔、冻结ct后备一致性、内层选择、末轮评分及最终monitor的65000份逐轮checkpoint。',
        f'累计训练时间{audit.seconds.sum()/60:.2f}分钟；首个运行配置至最后训练结束约{(max(ends)-min(starts))/60:.2f}分钟（报告与审计另计）。',
        '','## 限局与复现','',
        '规则受已观察到的case12异常启发。数值校准虽仅用训练case，本次仍是已知数据上的探索性修复，而非完全未触及测试集的确认性验证。规则敏感性结果一致不等于已证明能检测所有异常。',
        '固定结构、fold、seed和目标配对结果见phase1_case_before_after.csv及matched_test_predictions.csv.gz；不把五seed当成独立case，不根据外层测试挑选最佳网络。',
        '运行协议见README.md；审计命令：`python -m wear_monitoring_dynamics.scripts.audit_history_quality`。没有启动Phase2/4或融合。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result=dict(passed=True,runs=322,epochs=161000,prediction_rows=int(audit.prediction_rows.sum()),training_seconds=float(audit.seconds.sum()),elapsed_seconds=max(ends)-min(starts),history_selection=selected)
    write_json(ROOT/'completion_audit.json',result);print(json.dumps(result));print(sel.to_string(index=False))

if __name__=='__main__':main()
