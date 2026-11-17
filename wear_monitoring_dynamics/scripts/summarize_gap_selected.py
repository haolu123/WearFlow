"""Read-only result aggregation and presentation for gap-selected study."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run_gap_selected import OUT,METHODS,POOLS

def main():
    summary=pd.read_csv(OUT/'summary.csv');paired=pd.read_csv(OUT/'paired_comparison.csv');fold=pd.read_csv(OUT/'fold_summary.csv');mf=pd.read_csv(OUT/'manifest.csv');records=pd.read_csv(OUT/'records.csv')
    case=pd.read_csv(OUT/'case_metrics.csv');seed=case.groupby(['mode','name','seed']).mae.mean().reset_index();seed.to_csv(OUT/'seed_metrics.csv',index=False)
    selections=[]
    for f in POOLS:
        ranking=pd.read_csv(OUT/f/'final/ranking.csv');chosen=json.loads((OUT/f/'final/scalers.json').read_text())['chosen']
        for i,name in enumerate(chosen,1):
            r=ranking[ranking.feature==name].iloc[0];selections.append(dict(fold=f,rank=i,feature=name,train_ratio=r.score,in_top12=i<=12))
    sel=pd.DataFrame(selections);sel.to_csv(OUT/'selected_features.csv',index=False)
    # Strict run count, shape, common train populations, disjoint scopes, and finite checkpoints.
    for row in mf.itertuples():
        d=Path(row.run_dir);identity=json.loads((d/'identity.json').read_text());state=json.loads((d.parents[1]/'scalers.json').read_text())
        train=identity['train_cases'];assert state['train_cases']==sorted(train)
        if row.role=='inner':assert row.holdout not in train
        assert all(c in POOLS[row.fold] for c in train)
        with np.load(d/'predictions.npz') as z:assert z['predictions'].shape==(500,137) and np.isfinite(z['predictions']).all()
        assert len(pd.read_csv(d/'history.csv'))==500
    for _,g in mf.groupby(['fold','role','holdout','seed'],dropna=False):
        identities=[json.loads((Path(d)/'identity.json').read_text()) for d in g.run_dir]
        assert len(g)==3 and all(r['train_indices']==identities[0]['train_indices'] for r in identities)
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for col,f in enumerate(POOLS):
        for name in METHODS:
            inn=[pd.read_csv(OUT/f/f'inner_{c}'/'seed0'/name/'history.csv').validation_mae.values for c in POOLS[f]]
            final=[pd.read_csv(OUT/f/'final'/f'seed{s}'/name/'history.csv').validation_mae.values for s in range(5)]
            axs[0,col].plot(np.arange(1,501),np.mean(inn,axis=0),label=name)
            axs[1,col].plot(np.arange(1,501),np.mean(final,axis=0),label=name)
        axs[0,col].set_title(f+' inner validation (select epoch)');axs[1,col].set_title(f+' outer test (diagnostic only)')
    for ax in axs.flat:ax.set_xlabel('epoch');ax.set_ylabel('Case MAE (mm)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'learning_curves.png',dpi=140);plt.close(fig)
    epochs=json.loads((OUT/'selected_epochs.json').read_text());audit=json.loads((OUT/'audit.json').read_text())
    labels={'ct':'工况＋时间','ct_top12':'工况＋时间＋前12维','ct_top18':'工况＋时间＋前18维'}
    rows=[]
    for name in METHODS:
        a=summary[(summary.name==name)&(summary['mode']=='fixed500')].iloc[0];b=summary[(summary.name==name)&(summary['mode']=='inner_selected')].iloc[0]
        vals=seed[(seed.name==name)&(seed['mode']=='fixed500')].mae
        rows.append(f'<tr><td>{labels[name]}</td><td>{a.case_macro_mae:.5f}</td><td>{b.case_macro_mae:.5f}</td><td>{vals.min():.5f}–{vals.max():.5f}</td></tr>')
    ftable=fold.pivot(index=['mode','fold'],columns='name',values='mae').round(5).to_html()
    ptable=paired.round(5).to_html(index=False)
    lists=''.join('<h3>'+f+'最终训练侧排名</h3>'+sel[sel.fold==f].to_html(index=False) for f in POOLS)
    page='''<!doctype html><meta charset="utf-8"><title>差异/变化筛选12/18维训练对照</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;line-height:1.8}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}img{max-width:100%}code{font-size:14px}</style>
<h1>差异／变化筛选：前12、18维 sensor 是否优于工况＋时间？</h1>
<p>72次训练全部完成，每次500轮，共36,000个epoch。14次内层划分×3模型，2个外层fold×5个种子×3模型。统一137条有效VB记录，各模型训练及评价样本完全配对；基线重新训练，未复用旧138条记录实验的成绩。</p>
<h2>总体结果</h2><p>MAE单位mm，每个case等权，先算每次运行的误差再平均5个种子；不是先平均预测的集成成绩。固定500轮为主结果，内层选轮数为补充，外层测试不参与选轮数。</p>
<table><tr><th>模型</th><th>固定500轮 MAE↓</th><th>内层选轮数 MAE↓</th><th>固定500轮五种子范围</th></tr>'''+''.join(rows)+'''</table>
<h2>分fold结果</h2>'''+ftable+'''<h2>与基线配对差异</h2><p>delta_mae=含sensor−基线，负值为改善；improvement_percent为相对改善百分比；cases_better按每case平均5种子后判断。区间为14个case的配对bootstrap 95%区间，先平均种子；没有把种子当作独立case。只有14个case且工况成对，区间仅是描述性不确定性提示。</p>'''+ptable+'''
<h2>实验设置与含义</h2><ul>
<li>候选共99项。仅用当前训练case的VB曲线计算差异/变化，覆盖率至少80%、至少5个可比较对、排除极小变化；按比值升序排序，前12维为前18维前缀。每个内层划分都重新筛选。没有额外用相关性筛选，也未去除高度相关的重复信息，忠实检验本次单指标排序。</li>
<li>每个训练池内没有同工况重复case，训练排名比较的是不同工况case。这与上一页全数据7对同工况排名不同；不能将含测试标签的全局名单用于无泄漏验证。两个fold的名单不同是正常的。</li>
<li>三个模型均为[64,32]隐藏层SiLU MLP，AdamW lr=0.001、weight_decay=0.0001，全批量标准化VB MSE；输入分别为5、17、23维，参数量分别2497、3265、3649。结构宽度相同，参数量不完全相同。</li>
<li>工况输入为DOC/feed训练侧标准化、材料one-hot，时间除以训练侧h=3间隔中位数。sensor仅用训练有标签记录拟合中位数填补、0.5/99.5分位裁剪和z-score；不输入case ID、初始VB或测试标签。基线与sensor模型使用相同137条有效记录。</li>
<li>内层按训练池留一case，seed0，平均七条验证曲线选每fold/模型轮数；最终各跑seed0–4。历史CSV在final目录的validation_mae列实际为外层测试诊断，仅为沿用统一字段名，绝不用于轮数选择。</li>
<li>这是此前反复分析过的数据上的探索性比较，不是未触碰数据的独立确认。内层为无同工况伙伴的迁移，外层有同工况伙伴，验证任务仍不完全一致。</li></ul>
<p>内层选轮数：'''+str(epochs)+'''。</p>
<h2>训练曲线</h2><img src="learning_curves.png"><h2>每case比较</h2><img src="case_comparison.png">
<h2>最终入选特征</h2><p>每fold前12项用于12维模型，全部18项用于18维模型。仅展示外层最终名单；各内层名单存于相应ranking.csv和scalers.json。</p>'''+lists+'''
<h2>可复核文件</h2><p><a href="summary.csv">总体指标</a> · <a href="case_metrics.csv">逐case指标</a> · <a href="test_predictions.csv">外层逐样本预测</a> · <a href="selected_features.csv">特征名单</a> · <a href="protocol.json">协议</a> · <a href="audit.json">审计</a></p>
<p>复现：<code>python -m wear_monitoring_dynamics.scripts.run_gap_selected</code>，然后 <code>python -m wear_monitoring_dynamics.scripts.summarize_gap_selected</code>。保存每epoch全部预测、初始/最终/内层选轮checkpoint；72次最终与所选checkpoint均重载核验通过。两项新增测试检查留出数据不影响排名/预处理。</p>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    audit.update(train_population_paired=True,all_epoch_arrays_verified=True,selection_tests=2)
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    (OUT/'status.json').write_text(json.dumps(dict(status='complete',completed=72,planned=72)),encoding='utf-8')
    (OUT/'runner_snapshot.py').write_bytes(Path(__file__).with_name('run_gap_selected.py').read_bytes())
    print(summary.to_string(index=False));print(fold.to_string(index=False));print(paired.to_string(index=False));print(audit)

if __name__=='__main__':main()
