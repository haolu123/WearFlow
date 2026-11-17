"""Read-only training audit and explicitly labeled input sensitivity diagnostic."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from ..config import project_path, write_json, digest
from ..data.records import load_table
from ..data.preprocessing import fit_scalers
from ..data.datasets import tensors
from ..data.history import add_history
from ..data.splits import make_split
from ..models.registry import build_model
from ..training.checkpoint import load_checkpoint
from .run_history_phase13 import ROOT
from .run_train_only_coverage import FEATURES, rank_features


def main():
    m=pd.read_csv(ROOT/'study_manifest.csv');assert len(m)==322 and m.id.nunique()==322
    first=json.loads((Path(m.iloc[0].run_dir)/'provenance.json').read_text())['config']
    table,_=load_table(project_path(first['data']['path']),project_path(first['data']['cache']))
    frame=pd.DataFrame({k:v for k,v in table.items() if k!='sensor'});frame[FEATURES]=table['sensor']
    rank_cache={};audits=[];starts=[];ends=[]
    for r in m.itertuples():
        d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config']
        split=make_split(r.fold,r.role,None if pd.isna(r.holdout) else int(r.holdout))
        assert split==json.loads((d/'split.json').read_text())
        assert digest(fit_scalers(table,split['train']))==digest(json.loads((d/'scalers.json').read_text()))
        if r.type=='monitor' and r.name!='ct':
            key=tuple(split['train'])
            if key not in rank_cache:rank_cache[key]=rank_features(frame[frame.case.isin(key)]).head(18).feature.tolist()
            assert cfg['monitor']['selected_features']==rank_cache[key]
        h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,501))
        assert h.filter(regex='^(train|validation|test_diagnostic)_').notna().all().all()
        metrics=json.loads((d/'metrics.json').read_text());assert metrics['epochs']==500 and metrics['reload_verified']
        if r.type=='monitor':assert metrics['selected_epoch']==500 and metrics['selected_checkpoint']=='last.pt'
        if r.type=='monitor' and r.role=='final':assert len(list((d/'epochs').glob('epoch_*.pt')))==500
        p=pd.read_csv(d/'epoch_predictions.csv.gz')
        assert np.isfinite(p[['target_vb_mm','predicted_vb_mm']]).all().all()
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
    selected=json.loads((ROOT/'history_selection.json').read_text())
    scores=pd.read_csv(ROOT/'inner_validation_scores.csv')
    for fold,name in selected.items():
        q=scores[(scores.fold==fold)&scores.name.str.endswith(('_L3','_L5','_L10'))]
        assert name==q.sort_values(['mae','name']).iloc[0]['name']
    predictions=pd.read_csv(ROOT/'phase1_test_predictions.csv')
    chosen=pd.concat([predictions[(predictions.fold==fold)&(predictions.name==name)] for fold,name in selected.items()])
    rows=[]
    for label,mask in dict(overall=np.ones(len(chosen),bool),fold_a=chosen.fold=='fold_a',fold_b=chosen.fold=='fold_b',
        fold_a_in_range=(chosen.fold=='fold_a')&(chosen.vb_mm<=.74),fold_a_out_of_range=(chosen.fold=='fold_a')&(chosen.vb_mm>.74)).items():
        q=chosen[mask];rows.append(dict(subset=label,mae=q.groupby(['fold','seed','case']).absolute_error.mean().mean()))
    selected_metrics=pd.DataFrame(rows);selected_metrics.to_csv(ROOT/'selected_history_metrics.csv',index=False)
    # Sensitivity ONLY: replace one suspicious sensor record with training-mean
    # features (zero standardized input), retaining time/condition/scalers/model.
    device=torch.device('cuda');torch.set_num_threads(1);sensitivity=[];maxima=[]
    bad=np.flatnonzero((table['case']==12)&(table['run']==1)).item()
    idx=np.flatnonzero((table['case']==12)&np.isfinite(table['VB']))
    for r in m[(m.role=='final')&(m.fold=='fold_a')&m.name.isin(['mlp_L3','cnn_L3'])].itertuples():
        d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config'];scaler=json.loads((d/'scalers.json').read_text())
        batch=tensors(table,scaler,device);add_history(batch,table,3,scaler['time_scale'],device)
        model=build_model(cfg).to(device);load_checkpoint(d/'last.pt',model,device);model.eval()
        maxima.append(float(batch['sensor'][bad,model.monitor.selected_indices].abs().max()))
        with torch.no_grad():
            before=model.estimate(batch,idx).flatten().cpu().numpy()*scaler['wear'][1]+scaler['wear'][0]
            batch['sensor'][bad]=0
            after=model.estimate(batch,idx).flatten().cpu().numpy()*scaler['wear'][1]+scaler['wear'][0]
        affected=((batch['history_indices'][idx]==bad)&batch['history_mask'][idx]).any(1).cpu().numpy()
        np.testing.assert_allclose(before[~affected],after[~affected],atol=1e-7,rtol=0)
        for j,row in enumerate(idx):sensitivity.append(dict(name=r.name,seed=r.seed,case=12,run=table['run'][row],affected=bool(affected[j]),
            target=table['VB'][row],original=float(before[j]),neutralized=float(after[j])))
    diagnostic=pd.DataFrame(sensitivity);diagnostic.to_csv(ROOT/'case12_run1_input_sensitivity.csv',index=False)
    affected=diagnostic[diagnostic.affected].copy()
    affected['original_error']=abs(affected.original-affected.target);affected['neutralized_error']=abs(affected.neutralized-affected.target)
    ds=affected.groupby('name')[['original_error','neutralized_error']].mean()
    ds.to_csv(ROOT/'case12_run1_sensitivity_summary.csv')
    case=pd.read_csv(ROOT/'phase1_case_metrics.csv');case.groupby(['name','fold','case']).mae.mean().to_csv(ROOT/'phase1_case_mean.csv')
    # Fair diagnostic on six identical cases, not a replacement for primary results.
    excluded=case[(case.fold=='fold_a')&(case.case!=12)].groupby('name').mae.mean()
    excluded.to_csv(ROOT/'fold_a_without_case12_diagnostic.csv')
    summary=pd.read_csv(ROOT/'phase1_summary.csv').pivot(index='name',columns='subset',values='mae')
    lines=['','## 第一批结论与异常诊断','',
        '本轮历史信息没有呈现稳定的精度收益。fold B中三种结构所有L>1均不如自身L1；fold A的历史MLP/CNN出现严重极端值，不能仅据总体分数推断历史信息无用。GRU未出现同量级极端值，但历史窗口也未优于GRU_L1。',
        '内层验证选出的历史模型：fold A为MLP_L3，fold B为GRU_L3；合并结果约0.51624mm，明显落后当前MLP和无sensor基线。选择本身未使用外层测试。', '',
        '### 固定结构、窗口的主要结果','',
        '| 模型 | overall | fold A | fold B | A in-range | A out-of-range |','|---|---:|---:|---:|---:|---:|']
    for name,q in summary.iterrows():lines.append('| '+name+' | '+' | '.join(f'{q[k]:.5f}' for k in ['overall','fold_a','fold_b','fold_a_in_range','fold_a_out_of_range'])+' |')
    lines+=['','### Case 12 / run 1','',
        '该无VB标签记录的异常幅值此前已标记，本轮遵守原QC阈值保留。单点有标签评估不直接评分它，但历史窗口会把它传给后续目标；从首条sensor记录初始化的递推也会受到它影响。',
        f'在fold A训练标准化后，该记录选中特征的最大绝对值达到约 {max(maxima):.1f} 个训练标准差。',
        '固定网络和scaler，仅将该记录sensor替换为训练均值，MLP_L3/CNN_L3受影响窗口的误差如下；窗口不含该记录的预测保持不变。该实验只定位输入敏感性，不是新的正式成绩，也没有重新训练。','',
        '| 模型 | 原受影响点MAE | 中性输入后的MAE |','|---|---:|---:|']
    for name,q in ds.iterrows():lines.append(f'| {name} | {q.original_error:.5f} | {q.neutralized_error:.5f} |')
    lines+=['','另保存了统一排除case12后的六case诊断，但不能把它替换原七case主结果。',
        '从case12首次记录初始化时，当前MLP给出约−172至−319mm的VB，导致ODE进入训练范围之外。这是严重的初值/输入异常，不能解释为对0.005–0.03mm小噪声的敏感性。sensor估计初值未做非负截断；高斯扰动初值做了非负截断，两者协议不同，已分别标明。', '',
        '### Phase 3解释','',
        'oracle h1..3 MAE：fold A约0.09683，fold B约0.05897；sigma=0.03mm时为0.10222、0.06382。平均小噪声影响有限，不能推广为任意大/有偏初值下鲁棒。',
        '从首个实测VB开始且不再修正，整条轨迹MAE为fold A约0.22101、fold B约0.08895。整条轨迹与h1..3具有不同起点/预测距离/目标加权，不能将两者差值直接称作纯漂移增量；它们显示长期开环预测仍有明显误差。',
        '从首次可用sensor初始化时，fold A出现极端误差，主要涉及case12的异常首记录；从首个有标签点对齐初始化时应单独比较。16个扰动重复先平均，不能当成16倍独立样本。', '',
        '### 审计','',
        f'322次训练全部500epoch，共161000epoch；检查了每轮训练/验证/测试诊断预测完整且有限、真实case划分、训练内特征名单、scaler、末轮评分、65,000个final sensor checkpoint及重载标志。累计训练{audit.seconds.sum()/60:.2f}分钟。',
        '这些检查确认实验按协议执行，不代表输入质量良好；本轮明确暴露了无标签异常sensor进入历史窗口的风险。',
        '没有启动Phase2/4或修改正式数据。建议后续先确定统一、可部署且不依赖测试标签的sensor质量处理规则，再决定是否进入融合实验。',
        '复现审计：`python -m wear_monitoring_dynamics.scripts.audit_history_phase13`。']
    p=ROOT/'RESULTS.md';text=p.read_text(encoding='utf-8').split('\n## 第一批结论与异常诊断')[0]
    p.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    write_json(ROOT/'completion_audit.json',dict(passed=True,runs=322,epochs=161000,prediction_rows=int(audit.prediction_rows.sum()),
        training_seconds=float(audit.seconds.sum()),elapsed_seconds=max(ends)-min(starts),history_selection=selected,
        severe_input_sensitivity_case=12,severe_input_sensitivity_run=1))
    print(ds.to_string());print(excluded.to_string());print(selected_metrics.to_string(index=False))
    print(json.loads((ROOT/'completion_audit.json').read_text()))


if __name__=='__main__':main()
