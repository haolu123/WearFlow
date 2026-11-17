"""Audit completed direct/ODE runs and extend the readout; no model training."""

import json

from pathlib import Path

import numpy as np

import pandas as pd

from ..config import write_json,digest

from .run_direct_vs_ode import ROOT,PREVIOUS





def main():

    manifest=pd.read_csv(ROOT/'study_manifest.csv')

    assert len(manifest)==150 and not manifest.duplicated(['model','kind','fold','seed']).any()

    pred=pd.read_csv(ROOT/'epoch500_predictions.csv')

    configs={};parents=set();audits=[];starts=[];ends=[]

    previous=json.loads((PREVIOUS/'groups.json').read_text())

    expected={g['fold']:g['features'] for g in previous if g['study']=='train_only'}

    for row in manifest.itertuples():

        d=Path(row.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config']

        m=json.loads((d/'metrics.json').read_text());assert m['epochs']==m['selected_epoch']==500 and m['reload_verified']

        assert json.loads((d/'status.json').read_text())['status']=='completed'

        assert cfg['training']['selection_epoch']==500 and cfg['loss']['forecasting_weight']==1 and cfg['loss']['derivative_weight']==0

        assert cfg['dynamics']['transition']==row.kind

        if row.model!='time_ct_mlp':assert cfg['monitor']['selected_features']==expected[row.fold]

        h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,501))

        assert h.filter(regex='^(train|test_diagnostic)_').notna().all().all()

        assert len(list((d/'epochs').glob('epoch_*.pt')))==500

        raw=pd.read_csv(d/'epoch_predictions.csv.gz')

        assert np.isfinite(raw[['target_vb_mm','predicted_vb_mm']].to_numpy()).all()

        for _,g in raw.groupby(['split','task']):

            assert g.epoch.nunique()==500 and g.groupby('epoch').size().nunique()==1

        q=pred[(pred.model==row.model)&(pred.kind==row.kind)&(pred.fold==row.fold)&(pred.seed==row.seed)&(pred.initial=='estimated')]

        for task,newtask,keys in [('monitoring','current',['case','origin_run']),('forecasting','future',['case','origin_run','target_run','horizon'])]:

            saved=raw[(raw.epoch==500)&(raw.split=='test_diagnostic')&(raw.task==task)].sort_values(keys)

            calculated=q[(q.task==newtask)&(q.horizon<=3)].sort_values(keys)

            cols=keys+['target_vb_mm','predicted_vb_mm']

            np.testing.assert_allclose(saved[keys+['target_vb_mm']],calculated[keys+['target_vb_mm']],atol=1e-12,rtol=0)

            # h1..5 is evaluated in a different GPU batch than the saved h1..3.

            np.testing.assert_allclose(saved.predicted_vb_mm,calculated.predicted_vb_mm,atol=1e-6,rtol=0)

        parent=Path(cfg['dynamics']['checkpoint']).parent;parents.add(parent)

        pcfg=json.loads((parent/'provenance.json').read_text())['config']

        assert pcfg['dynamics']['transition']==row.kind and pcfg['experiment']['fold']==row.fold

        for file in ['scalers.json','split.json']:

            assert json.loads((parent/file).read_text())==json.loads((d/file).read_text())

        assert pcfg['data']==cfg['data']

        norm=json.loads(json.dumps(cfg));norm['experiment'].pop('name');norm['dynamics'].pop('transition');norm['dynamics'].pop('checkpoint')

        key=(row.model,row.fold,row.seed)

        if key in configs:assert configs[key]==norm

        else:configs[key]=norm

        starts.append((d/'config_resolved.yaml').stat().st_mtime);ends.append((d/'status.json').stat().st_mtime)

        audits.append(dict(model=row.model,kind=row.kind,fold=row.fold,seed=row.seed,rows=len(raw),seconds=m['wall_seconds']))

    assert len(parents)==6

    pre=[]

    for d in parents:

        m=json.loads((d/'metrics.json').read_text());assert m['epochs']==100 and m['reload_verified']

        h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,101))

        raw=pd.read_csv(d/'epoch_predictions.csv.gz')

        assert raw.epoch.nunique()==100 and np.isfinite(raw.predicted_vb_mm).all()

        cfg=json.loads((d/'provenance.json').read_text())['config']

        pre.append(dict(kind=cfg['dynamics']['transition'],fold=cfg['experiment']['fold'],seconds=m['wall_seconds']))

    audit=pd.DataFrame(audits);audit.to_csv(ROOT/'epoch_record_audit.csv',index=False)

    pre=pd.DataFrame(pre);pre.to_csv(ROOT/'pretraining_audit.csv',index=False)

    timing=audit.groupby('kind').seconds.agg(['sum','mean']);timing['pretraining_seconds']=pre.groupby('kind').seconds.sum()

    timing.to_csv(ROOT/'timing_summary.csv')

    diag=pd.read_csv(ROOT/'transition_diagnostics.csv').groupby('kind')[['semigroup_gap_mm','decline_fraction','transition_parameters']].mean()

    summary=pd.read_csv(ROOT/'summary.csv')

    lines=['','## 结果解读与完整性核查','',

           '**本实验没有支持 ODE 积分传播在预测精度上优于直接状态映射。** 合并两个 fold、未来 h=1..3 的部署式估计初值结果中，五种 monitoring 结构都是 direct 最低、direct_positive 次之、ODE 更高。',

           '实测初值 oracle 下仍是直接映射误差更低，因此差距不能完全归因于当前 VB 估计误差。不过各分支与监测器联合训练，oracle 结果仍不等同于独立识别的真实动力学。',

           '当前 VB 方面，ODE 在无 sensor、时间编码、FiLM、attention 四种结构上略优于两种直接映射；拼接 MLP 的非负直接映射略好。当前估计与未来预测的收益并不一致。',

           '五种结构的 ODE−direct 和 ODE−direct_positive 的合并 h1..3 case 配对 bootstrap 95% 区间都大于零，但这些区间未校正多重比较，且数据只有14个独立测试case。',

           '直接模型在这批测试起点及 h1..5 上没有预测负增量（包括未加非负约束的 direct）；这不是对任意输入的保证。因此 ODE 的误差差距不能简单解释为直接模型通过非物理下降取得优势。','',

           '### 分预测步数的部署式误差','',

           'h1/2/3 参与训练；h4/5 未参与训练，但真实时间跨度可能与训练范围重叠。每列对可评估的 case 平均，远期因样本不足可能减少 case，比较时不要忽略人群变化。','',

           '| model | kind | fold | h1 | h2 | h3 | h4 | h5 |','|---|---|---|---:|---:|---:|---:|---:|']

    future=summary[(summary.task=='future')&(summary.initial=='estimated')&(summary.horizon>0)]

    for (model,kind,fold),g in future.groupby(['model','kind','fold']):

        vals=g.set_index('horizon').mae

        lines.append('| '+model+' | '+kind+' | '+fold+' | '+' | '.join(f'{vals.loc[h]:.5f}' for h in range(1,6))+' |')

    lines+=['','### 配对 case bootstrap 区间','',

            'Δ=ODE−对应对照；正值表示 ODE 误差更高，单位 mm。下表合并 h1..3、两个fold，以case为抽样单位；先对seed平均，未校正多重比较。','',

            '| model | initial | reference | Δ MAE | 95% lower | 95% upper |','|---|---|---|---:|---:|---:|']

    for r in pd.read_csv(ROOT/'paired_case_comparisons.csv').itertuples():

        lines.append(f'| {r.model} | {r.initial} | {r.reference} | {r.delta_mae:+.5f} | {r.ci_low:+.5f} | {r.ci_high:+.5f} |')

    lines+=['','### 计算成本与传播一致性','',

            '| kind | propagation params | mean train seconds/run | all pretrain seconds | split-step gap mm |','|---|---:|---:|---:|---:|']

    for kind in ['direct','direct_positive','ode']:

        lines.append(f'| {kind} | {diag.loc[kind,"transition_parameters"]:.0f} | {timing.loc[kind,"mean"]:.2f} | {timing.loc[kind,"pretraining_seconds"]:.2f} | {diag.loc[kind,"semigroup_gap_mm"]:.8f} |')

    elapsed=max(ends)-min(starts);total=audit.seconds.sum()+pre.seconds.sum()

    lines += ['',

              '成本包含每轮诊断和模型保存，是实际实验运行成本，不是纯前向推理 benchmark。直接分支比 ODE rate 多32个参数，因额外接收 Δt。',

              'ODE 一次长步与两次半步的结果几乎一致；直接映射存在约0.015–0.016 mm的平均差异。这里对 h1..5 的case内所有pair取均值，再对运行取均值，不能与case-macro MAE直接等同。恒定不变的persistence同样完全一致，因此一致性本身不等于预测准确。',

              f'150次新训练共75000 epoch，6次预训练各100 epoch；所有每轮训练/测试指标、原始预测和75000个checkpoint核查通过。训练累计 {total/60:.2f} 分钟，首个训练配置到最后训练完成区间 {elapsed/60:.2f} 分钟（不含本次核查）。',

              '模型间配置除传播分支、名称和匹配的预训练checkpoint外一致；训练特征名单、预训练数据/标准化一致；重新评估的当前与未来 h1..3 预测逐条匹配原始500轮记录。未来sensor/真实标签输入隔离已由单元测试验证。',

              '论文应如实写成：本设定下，ODE表现出传播组合一致性，但没有取得直接状态映射的预测精度优势，也付出了更高计算成本。不能把本实验描述为证明积分传播提高预测准确率。',

              '复现核查：`python -m wear_monitoring_dynamics.scripts.audit_direct_vs_ode`。']

    path=ROOT/'RESULTS.md';existing=path.read_text(encoding='utf-8').split('\n## 结果解读与完整性核查')[0]

    path.write_text(existing+'\n'.join(lines)+'\n',encoding='utf-8')

    write_json(ROOT/'completion_audit.json',dict(passed=True,new_runs=150,pretrain_runs=6,pure_references=50,

        epochs=75000,prediction_rows=int(audit.rows.sum()),training_seconds=float(total),elapsed_seconds=elapsed))

    print(timing.to_string());print(json.loads((ROOT/'completion_audit.json').read_text()))

    print(future[future.horizon<=3].pivot(index=['model','kind','fold'],columns='horizon',values='mae').to_string())





if __name__=='__main__':main()

