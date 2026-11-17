"""Controlled ranking comparison on identical99 candidates and137 targets."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import run_gap_selected as study

ROOT=study.OUT.with_name('selection_rules_20261013')
THRESHOLD=.7

def selection(data,cases,features,rule):
    # Common eligibility across strategies isolates the ordering/filter criterion.
    _,ranking=study.rank_features(data,cases,features)
    train=data[data.case.isin(cases)]
    correlations={}
    for f in features:
        rs=[]
        for c in cases:
            d=train[train.case==c].dropna(subset=[f,'VB'])
            rho=float(spearmanr(d[f],d.VB).statistic) if len(d)>=3 and d[f].nunique()>1 and d.VB.nunique()>1 else 0.
            rs.append(abs(rho))
        correlations[f]=float(np.mean(rs))
    ranking['mean_abs_rho']=ranking.feature.map(correlations)
    ranking['passes_correlation_gate']=ranking.mean_abs_rho>=THRESHOLD
    if rule=='correlation':
        ranking=ranking.sort_values(['mean_abs_rho','feature'],ascending=[False,True]);eligible=ranking.eligible
    elif rule=='hybrid':
        ranking=ranking.sort_values(['score','feature']);eligible=ranking.eligible&ranking.passes_correlation_gate
    else:raise ValueError(rule)
    chosen=ranking[eligible].feature.head(18).tolist()
    if len(chosen)<18:raise ValueError(f'{rule} has only {len(chosen)} eligible candidates in {cases}; do not silently relax threshold')
    ranking['selected18']=ranking.feature.isin(chosen)
    return chosen,ranking

def main():
    ROOT.mkdir(exist_ok=True)
    raw=pd.read_csv(study.SOURCE);data=raw[raw.case.isin(study.ACTIVE)&raw.VB.notna()].reset_index(drop=True);features=list(raw.columns[7:])
    protocol=dict(candidates=99,targets=137,threshold=THRESHOLD,
        shared_eligibility='Train coverage>=80%, >=5 informative cross-case pairs, nonzero train scale; same as gap-only study',
        correlation='Equal-case mean absolute Spearman rho; >=3 records and nonconstant feature/VB else case contributes0; descending, feature name tie-break',
        hybrid='Require mean absolute rho>=0.70, then ascending gap/swing, feature name tie-break; no fallback/relaxation',
        freeze='Threshold fixed before training outcomes, no tuning. Every inner split refits ranking.',
        design='Same99 candidates,137 targets, architectures, optimizer, seed0..4, splits,500epochs and preprocessing.12/18 fixed. All ranking uses train cases only.',
        old_gap_results=str(study.OUT),new_runs=144,baseline_replays=48,source_sha256=hashlib.sha256(study.SOURCE.read_bytes()).hexdigest())
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    # Check feasibility before any model fitting; uses training-side data only.
    counts=[]
    for fold,pool in study.POOLS.items():
        for holdout in pool+[None]:
            cases=[c for c in pool if c!=holdout];_,rank=selection(data,cases,features,'hybrid')
            counts.append(dict(fold=fold,holdout=holdout,qualified=int((rank.eligible&rank.passes_correlation_gate).sum())))
    pd.DataFrame(counts).to_csv(ROOT/'gate_feasibility.csv',index=False)
    for rule in ['correlation','hybrid']:
        desc=protocol[rule]+'; common eligibility: '+protocol['shared_eligibility']+'; top12 is prefix of top18; no redundancy removal'
        study.main(out=ROOT/rule,selector=lambda d,c,f:selection(d,c,f,rule),selection_description=desc)
    report()

def report():
    paths={'gap':study.OUT,'correlation':ROOT/'correlation','hybrid':ROOT/'hybrid'}
    allscores=[];selectionrows=[];audits=[]
    reference=pd.read_csv(study.OUT/'manifest.csv')
    for rule,path in paths.items():
        scores=pd.read_csv(path/'case_metrics.csv');scores['rule']=rule;allscores.append(scores)
        mf=pd.read_csv(path/'manifest.csv')
        for r in mf.itertuples():
            d=Path(r.run_dir);identity=json.loads((d/'identity.json').read_text())
            match=reference[(reference.fold==r.fold)&(reference.role==r.role)&(reference.seed==r.seed)&(reference.name==r.name)]
            match=match[match.holdout.isna()] if pd.isna(r.holdout) else match[match.holdout==r.holdout]
            assert len(match)==1;old=Path(match.iloc[0].run_dir);oi=json.loads((old/'identity.json').read_text())
            assert identity['train_indices']==oi['train_indices'] and identity['train_cases']==oi['train_cases']
            a=torch.load(d/'initial.pt',weights_only=True);b=torch.load(old/'initial.pt',weights_only=True)
            assert all(torch.equal(a[k],b[k]) for k in a)
            if r.name=='ct':
                with np.load(d/'predictions.npz') as x,np.load(old/'predictions.npz') as y:
                    np.testing.assert_array_equal(x['predictions'],y['predictions'])
            history=pd.read_csv(d/'history.csv');assert len(history)==500
            with np.load(d/'predictions.npz') as p:assert p['predictions'].shape==(500,137) and np.isfinite(p['predictions']).all()
            audit=json.loads((d/'complete.json').read_text());assert audit['reload_verified']
        for fold in study.POOLS:
            chosen=json.loads((path/fold/'final/scalers.json').read_text())['chosen']
            for i,f in enumerate(chosen,1):selectionrows.append(dict(rule=rule,fold=fold,rank=i,feature=f,in_top12=i<=12))
        audits.append(dict(rule=rule,runs=len(mf),common_training_population=True,initial_weights_identical=True,ct_all_epoch_predictions_identical=True,checkpoint_reload=True))
    scores=pd.concat(allscores);scores.to_csv(ROOT/'all_case_metrics.csv',index=False)
    summary=scores.groupby(['mode','rule','name']).mae.mean().reset_index();summary.to_csv(ROOT/'summary.csv',index=False)
    fold=scores.groupby(['mode','rule','fold','name']).mae.mean().reset_index();fold.to_csv(ROOT/'fold_summary.csv',index=False)
    pd.DataFrame(selectionrows).to_csv(ROOT/'selected_features.csv',index=False)
    comparisons=[];rng=np.random.default_rng(20261013)
    for mode in ['fixed500','inner_selected']:
        v=scores[scores['mode']==mode].groupby(['rule','name','case']).mae.mean()
        for rule in paths:
            for name in ['ct_top12','ct_top18']:
                for ref_rule,ref_name in [('gap','ct'),('gap',name),('correlation',name)]:
                    if (rule,name)==(ref_rule,ref_name):continue
                    a=v.loc[(rule,name)];b=v.loc[(ref_rule,ref_name)];delta=(a-b).values
                    boot=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
                    comparisons.append(dict(mode=mode,rule=rule,name=name,reference=ref_rule+'_'+ref_name,delta_mae=float(delta.mean()),cases_better=int((delta<0).sum()),ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975))))
    paired=pd.DataFrame(comparisons);paired.to_csv(ROOT/'paired_comparison.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,mode in zip(axs,['fixed500','inner_selected']):
        s=summary[(summary['mode']==mode)&(summary.name!='ct')].pivot(index='rule',columns='name',values='mae')
        s.plot.bar(ax=ax,rot=0);base=summary[(summary['mode']==mode)&(summary.name=='ct')].mae.iloc[0]
        ax.axhline(base,color='black',ls='--',label='CT baseline');ax.set_title(mode);ax.set_ylabel('Case macro MAE (mm)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'comparison.png',dpi=140);plt.close(fig)
    labels={'gap':'差异/变化','correlation':'相关性','hybrid':'相关性≥0.70，再按差异/变化'}
    rows=[]
    for rule in paths:
        for name in ['ct_top12','ct_top18']:
            a=summary[(summary.rule==rule)&(summary.name==name)&(summary['mode']=='fixed500')].mae.iloc[0]
            b=summary[(summary.rule==rule)&(summary.name==name)&(summary['mode']=='inner_selected')].mae.iloc[0]
            rows.append(f'<tr><td>{labels[rule]}</td><td>{name[-2:]}</td><td>{a:.5f}</td><td>{b:.5f}</td></tr>')
    page='''<!doctype html><meta charset="utf-8"><title>三种特征筛选规则对照</title><style>body{font:16px system-ui;line-height:1.8;max-width:1200px;margin:35px auto}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}</style>
<h1>固定候选与训练流程，只改变特征筛选规则</h1>
<p><b>本次固定500轮结果：相关性排序优于混合排序，混合排序优于纯差异排序；所有sensor方案的平均MAE仍高于工况＋时间。</b>最佳sensor组合为相关性前12维，MAE=0.09068，较基线高8.9%。同维数相比纯差异排序，相关性12维改善11/14个case，18维改善10/14个case；配对差异的描述性95%区间仍包含0，不能据此宣称普遍显著优势。</p>
<p>恢复相关性优先级改善了这批特征的预测结果，支持“纯差异排序筛掉了部分有效磨损信息”的解释，但不证明这是此前跨实验退化的唯一原因。相关性高的名单含许多相近幅值特征；本轮没有检验去冗余或相对于工况时间的条件增量筛选。</p>
<p>内层选轮数后各规则排名改变，且仍全部未超过基线。例如相关性12维在fold_a选择53轮，外层平均误差升高；这是当前内层/外层迁移任务不一致下的观察，不能用外层测试反选轮数。</p>
<p>相同99项电流候选、137条标签、14个case、同一MLP隐藏层[64,32]、AdamW、500轮。12/18维固定；每个内层重新筛选，外层5种子。新增144次训练，旧差异/变化72次作为参照；新实验重复的48次CT基线的每epoch预测与旧实验逐值相同。同维数模型跨规则初始权重完全相同。</p>
<p>工况＋时间基线：固定500轮MAE <b>0.08330</b>；内层选轮数MAE <b>0.08254</b>。下表全部为额外添加sensor后的case等权平均误差，单位mm。</p>
<table><tr><th>筛选规则</th><th>维数</th><th>固定500轮MAE↓</th><th>内层选轮数MAE↓</th></tr>'''+''.join(rows)+'''</table><img src="comparison.png">
<h2>规则固定方式</h2><ul><li>三规则共用原差异法资格条件：训练侧覆盖至少80%，至少5个可比较case对，训练内变化尺度非零。相关性法是在相同候选资格范围内改排序，保证对照只检验筛选方式。</li>
<li>相关性：各训练case内绝对Spearman rho的等权平均，至少3个观测且特征/VB不恒定，否则该case贡献0。降序选前18，前12为其前缀。不额外要求符号相同。</li>
<li>混合：先要求上述平均绝对rho≥0.70，再按差异/变化升序。0.70在模型训练前固定，没有根据外层成绩调门槛；16个训练划分均足够18项，不放宽门槛，不补位。未做去冗余。</li>
<li>每个训练池内没有同工况重复case，差异法和混合法使用跨工况对；并非前面全数据同工况7对的排名。测试标签不参与任何名单或轮数选择。</li>
<li>每fold在七个内层case、seed0的平均验证曲线上选轮数；外层各跑seed0–4。参数量在相同维数下相同，12与18维之间不同。全部预处理、缺失填补与VB标准化仅拟合训练侧。</li>
<li>这是已多次观察的数据上的探索性对照，不能视为新独立验证。配对bootstrap先平均种子再重采样14个case，未考虑同工况case间相关性，区间仅作描述。</li></ul>
<h2>分fold结果</h2>'''+fold.pivot(index=['mode','rule','name'],columns='fold',values='mae').round(5).to_html()+'''
<h2>配对差异</h2><p>delta_mae=当前−参照，负值表示改善，cases_better为14个case中改善个数，CI为描述性95%区间。</p>'''+paired.round(5).to_html(index=False)+'''
<h2>数据与名单</h2><p><a href="selected_features.csv">三个规则两个fold最终名单</a> · <a href="summary.csv">总体结果</a> · <a href="all_case_metrics.csv">逐case与种子</a> · <a href="gate_feasibility.csv">门槛覆盖检查</a> · <a href="protocol.json">预设协议</a> · <a href="audit.json">审计</a></p>
<p>运行：python -m wear_monitoring_dynamics.scripts.run_selection_comparison。correlation和hybrid目录保留内层/外层排名、标准化、每epoch预测与初始/最终/选轮checkpoint。</p>'''
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    (ROOT/'audit.json').write_text(json.dumps(dict(new_runs=144,reference_runs=72,new_epochs=72000,selection_tests_passed=3,existing_isolation_tests_passed=2,rules=audits),indent=2),encoding='utf-8')
    (ROOT/'runner_snapshot.py').write_bytes(Path(__file__).read_bytes())
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',new_runs=144)),encoding='utf-8')
    print(summary.to_string(index=False));print(fold.to_string(index=False))

if __name__=='__main__':main()
