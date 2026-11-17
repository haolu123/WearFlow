"""Post-run audit and publication-facing comparisons; never retrains models."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import project_path, write_json, digest
from ..data.records import load_table
from ..data.subsets import restrict_training_records
from ..data.preprocessing import fit_scalers
from ..data.splits import POOLS
from .run_train_only_coverage import ROOT, FEATURES, rank_features
from .run_fixed_conditioning import MODELS
from .compare_fixed500_ranges import calculate_metrics


def main():
    manifest = pd.read_csv(ROOT/'study_manifest.csv')
    assert len(manifest) == 250 and not manifest.duplicated(['study','group','model','mode','fold','seed']).any()
    frames, audit, parents, starts, ends = [], [], set(), [], []
    table = None
    for run in manifest.itertuples():
        d = Path(run.run_dir)
        info = json.loads((d/'provenance.json').read_text())
        cfg = info['config']
        split = json.loads((d/'split.json').read_text())
        metrics = json.loads((d/'metrics.json').read_text())
        assert metrics['epochs'] == metrics['selected_epoch'] == 500 and metrics['reload_verified']
        assert json.loads((d/'status.json').read_text())['status'] == 'completed'
        h = pd.read_csv(d/'history.csv')
        assert h.epoch.tolist() == list(range(1,501))
        assert h.filter(regex='^(train|test_diagnostic)_').notna().all().all()
        assert len(list((d/'epochs').glob('epoch_*.pt'))) == 500
        if table is None:
            table,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))
        sub = restrict_training_records(table,split['train'],cfg['data'].get('train_records'))
        assert digest(fit_scalers(sub,split['train'])) == digest(json.loads((d/'scalers.json').read_text()))
        expected_train = {(int(c),int(r)) for c,r,v in zip(sub['case'],sub['run'],sub['VB']) if c in split['train'] and np.isfinite(v)}
        pred = pd.read_csv(d/'epoch_predictions.csv.gz')
        assert np.isfinite(pred[['target_vb_mm','predicted_vb_mm']].to_numpy()).all()
        mon = pred[pred.task=='monitoring']
        for label,cases in [('train',split['train']),('test_diagnostic',split['test'])]:
            q=mon[mon.split==label]
            assert q.epoch.nunique()==500 and q.groupby('epoch').size().nunique()==1
            assert set(q.case.unique())==set(cases)
        q=mon[(mon.split=='train') & (mon.epoch==500)]
        assert set(zip(q.case,q.origin_run))==expected_train
        for label, group in pred.groupby(['split','task']):
            assert group.epoch.nunique()==500 and group.groupby('epoch').size().nunique()==1
        final = mon[(mon.epoch==500)&(mon.split=='test_diagnostic')].copy()
        meta={k:getattr(run,k) for k in ['study','group','model','mode','fold','seed']}
        frames.append(final.assign(**meta))
        actual=(final.predicted_vb_mm-final.target_vb_mm).abs().groupby(final.case).mean().mean()
        assert abs(actual-h.iloc[-1].test_diagnostic_monitoring_case_macro_mae)<1e-7
        if cfg['dynamics']['checkpoint']:
            parent=Path(cfg['dynamics']['checkpoint']).parent
            parents.add(parent)
            pc=json.loads((parent/'provenance.json').read_text())['config']
            assert pc['data'].get('train_records')==cfg['data'].get('train_records')
            assert json.loads((parent/'scalers.json').read_text())==json.loads((d/'scalers.json').read_text())
        starts.append((d/'config_resolved.yaml').stat().st_mtime);ends.append((d/'status.json').stat().st_mtime)
        audit.append(dict(**meta,rows=len(pred),wall_seconds=metrics['wall_seconds'],run_dir=str(d)))
    assert len(parents)==9
    pretime=0
    for d in parents:
        m=json.loads((d/'metrics.json').read_text());assert m['epochs']==100 and m['reload_verified']
        h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,101))
        p=pd.read_csv(d/'epoch_predictions.csv.gz');assert p.epoch.nunique()==100
        pretime+=m['wall_seconds']
    # Audit feature ranking using only declared training rows.
    raw=pd.DataFrame({k:v for k,v in table.items() if k!='sensor'});raw[FEATURES]=table['sensor']
    groups=json.loads((ROOT/'groups.json').read_text())
    for g in groups:
        training=raw[raw.case.isin(POOLS[g['fold']])]
        if g['study']=='coverage':training=training[training.VB.notna() & (training.VB<=.74)]
        assert rank_features(training).head(18).feature.tolist()==g['features']
    full=pd.concat(frames,ignore_index=True)
    full.to_csv(ROOT/'epoch500_test_predictions.csv',index=False)
    pd.DataFrame(audit).to_csv(ROOT/'run_audit.csv',index=False)
    old=pd.read_csv(project_path('wear_monitoring_dynamics/outputs/fixed500_baseline_ranges_20260928/epoch500_test_predictions.csv'))
    training=full[full.study=='train_only']
    new_metrics,_=calculate_metrics(training)
    old_metrics,_=calculate_metrics(old)
    comp=new_metrics.merge(old_metrics,on=['model','mode','subset'],suffixes=('_train_only','_global'))
    comp['delta']=comp.case_macro_mae_train_only-comp.case_macro_mae_global
    comp.to_csv(ROOT/'feature_selection_comparison.csv',index=False)
    # Test identities are identical across every group and seed.
    for _,g in full.groupby(['study','group','model','mode','fold','seed']):
        ref=old[(old.model=='time_ct_mlp')&(old['mode']=='pure')&(old.seed==0)&(old.fold==g.fold.iloc[0])]
        cols=['case','origin_run','target_vb_mm']
        np.testing.assert_allclose(g.sort_values(cols[:2])[cols],ref.sort_values(cols[:2])[cols],atol=1e-12,rtol=0)
    coverage=full[full.study=='coverage'].copy()
    coverage['error']=(coverage.predicted_vb_mm-coverage.target_vb_mm).abs()
    cases=coverage.groupby(['group','model','mode','seed','case']).error.mean().reset_index()
    cases.to_csv(ROOT/'coverage_case_seed_mae.csv',index=False)
    summary=cases.groupby(['group','model','mode']).error.mean().unstack('group')
    summary['matched_minus_low']=summary.matched65-summary.low65
    summary.to_csv(ROOT/'coverage_comparison.csv')
    labels={name:label for name,label,_,_ in MODELS}
    lines=['# 两组补充实验结果（固定 500 epoch）','',
           '250 次 monitoring 与 9 次 ODE 预训练全部完成。主指标为当前 VB 的 case-macro MAE，单位 mm；对 case 与五次运行等权平均，越小越好。', '',
           '## 实验一：特征只在训练 case 内筛选','',
           '下表单元格为“此前全局筛选 → 本次训练内筛选”。无 sensor 行没有特征筛选，作为复现实验基线。', '']
    subsets=['overall','fold_a','fold_b','fold_a_in_range','fold_a_out_of_range']
    for mode,title in [('pure','纯监督'),('ode','加 ODE')]:
        lines += [f'### {title}','','| 模型 | 总体 | fold A | fold B | A in-range | A out-of-range |','|---|---:|---:|---:|---:|---:|']
        for name,_,_,_ in MODELS:
            q=comp[(comp.model==name)&(comp['mode']==mode)].set_index('subset')
            vals=[f'{q.loc[s,"case_macro_mae_global"]:.5f} → {q.loc[s,"case_macro_mae_train_only"]:.5f}' for s in subsets]
            lines.append('| '+labels[name]+' | '+' | '.join(vals)+' |')
        lines.append('')
    lines += ['fold A in-range 为真实 VB ≤ 0.74 mm（7 case、65 观测），out-of-range 为 >0.74 mm（4 case、9 观测）。按范围内存在观测的 case 平均，两列不能直接加权恢复整体 macro。','',
              '训练内筛选后，FiLM＋ODE 在 fold B 仍优于无 sensor＋ODE，但优势由约 16.9% 缩小到约 6.8%（0.07724 vs 0.08287）。纯监督 FiLM 与 attention 在 fold B 均不再优于纯监督基线。总体仍是无 sensor 基线更好。','',
              '## 实验二：固定 fold B，控制磨损覆盖与样本数','',
              '| 模型 | 目标 | full74 | low65 | matched65 | matched − low |','|---|---|---:|---:|---:|---:|']
    for name,_,_,_ in MODELS:
        for mode in ['pure','ode']:
            r=summary.loc[(name,mode)]
            lines.append(f'| {labels[name]} | {mode} | {r.full74:.5f} | {r.low65:.5f} | {r.matched65:.5f} | {r.matched_minus_low:+.5f} |')
    lines += ['', 'full74：全部 74 个有标签训练观测；low65：仅 VB≤0.74 的 65 个；matched65：保留全部 9 个高磨损观测，并用它们替换相同 case 内的 9 个低磨损观测。后两组逐 case 数量完全一致，测试集相同。', '',
              '**本次对照不支持“见过高磨损是 fold B 更好表现的原因”：10 种模型/目标组合的 low65 均优于 matched65，也均优于 full74。** 这说明对当前低磨损测试范围，增加覆盖到更高磨损并没有自动改善估计；也可能涉及样本分布匹配、训练标准化或优化难度。不能由此断言高磨损数据普遍无用。','',
              'low65 中纯监督时间编码 MLP 为 0.06780、拼接 MLP 为 0.06927，优于纯监督无 sensor 的 0.07211。加入 ODE 在 low65 的 FiLM、attention 上改善，但拼接和编码 MLP 变差；无 sensor＋ODE 为 0.07087，低于四种 sensor＋ODE 的均值。','',
              '## 可比性与局限','',
              '- 实验一保留原协议的无标签训练记录；实验二三组统一只用有标签记录。实验二固定使用 low65 选出的同一组 top-18，因此其 full74 与实验一 fold B 不是同一个数据预处理/特征设置。应在各实验内部作匹配比较。',
              '- matched65 的五次运行同时改变固定抽样 seed 与网络初始化；其运行间波动不是纯初始化方差。高磨损也对应后期时间，且删点会改变 ODE 的相邻标签配对时间间隔，所以不是完全分离时间因素的因果证据。',
              '- 每组标准化仅拟合保留的训练记录，ODE 预训练也仅用相同记录。特征选择和测试评估严格隔离，但这仍是在反复查看过的同一测试集上的补充实验，不是新独立测试。',
              '- 本报告为五次运行的均值比较，未将小差异宣称为统计显著；独立测试 case 只有每 fold 七个。', '',
              '## 审计与耗时','',
              f'- 250 次训练共 125000 个 epoch，逐 epoch 训练/测试指标和预测完整，125000 个 epoch checkpoint 存在。',
              f'- 9 次 ODE 预训练各 100 epoch。训练累计用时 {(sum(a["wall_seconds"] for a in audit)+pretime)/3600:.3f} 小时；首个运行配置到最后运行完成的实际区间 {(max(ends)-min(starts))/3600:.3f} 小时。',
              '- 核查了所有运行的原始预测有限性、每轮记录数量、测试 case/run/真实标签一致性、500 轮结果与 history 一致性、checkpoint 重载、训练记录与 scaler、ODE 预训练子集，以及三套训练内特征排名。', '',
              '复现报告：`python -m wear_monitoring_dynamics.scripts.audit_train_only_coverage`。',
              '详细表见 feature_selection_comparison.csv、coverage_comparison.csv、coverage_case_seed_mae.csv、epoch500_test_predictions.csv、completion_audit.json。']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(ROOT/'completion_audit.json',dict(passed=True,monitor_runs=250,pretrain_runs=9,
        monitoring_epochs=125000,prediction_rows=sum(a['rows'] for a in audit),
        training_seconds=sum(a['wall_seconds'] for a in audit)+pretime,elapsed_seconds=max(ends)-min(starts)))
    print(summary.to_string())
    print(json.loads((ROOT/'completion_audit.json').read_text()))


if __name__=='__main__': main()

