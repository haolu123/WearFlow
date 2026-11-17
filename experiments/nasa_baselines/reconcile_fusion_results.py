"""Reconcile original complete fusion results and later component comparisons. No training."""
from wearflow_paths import ROOT as _WF_ROOT
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARENT = Path(str(_WF_ROOT / 'wear_monitoring_dynamics/outputs/current12_abcd_from100_20261019'))
OUT = ROOT / 'reconciliation_20261110'

def mae(q):
    return q.assign(ae=abs(q.pred_mm-q.target_mm)).groupby(['fold','case','seed']).ae.mean().groupby(['fold','case']).mean().mean()

def table(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{v:.5f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])

def main():
    OUT.mkdir(exist_ok=True)
    f=pd.read_csv(PARENT/'predictions.csv.gz')
    f=f[f.split=='test_diagnostic'].rename(columns={'target_vb_mm':'target_mm','predicted_vb_mm':'pred_mm'})
    names=dict(zip('ABCD',['independent_monitor100_ode500','current_only_extra100','frozen_ode_future_extra100','joint_future_extra100']))
    audits={}
    for task,filename in [('monitoring','ours_current_predictions.csv'),('forecasting','ours_future_predictions.csv')]:
        q=f[(f.protocol=='single_shot')&(f.task==task)].copy();q['method']=q.group.map(names)
        ref=pd.read_csv(ROOT/'references'/filename)
        keys=['method','fold','seed','case','origin_run','target_run','horizon']
        j=q.merge(ref,on=keys,suffixes=('_old','_export'),validate='one_to_one')
        assert len(j)==len(q)==len(ref)
        err=float(abs(j.pred_mm_old-j.pred_mm_export).max());assert err==0
        audits[task]={'rows':len(j),'max_prediction_difference':err}
    s=pd.read_csv(PARENT/'summary.csv');errors=[]
    for row in s[s.subset=='overall'].itertuples():
        q=f[(f.group==row.group)&(f.protocol==row.protocol)&(f.task==row.task)]
        if pd.notna(row.horizon):q=q[q.horizon==row.horizon]
        errors.append(abs(mae(q)-row.mae))
    audits['original_summary_max_error']=float(max(errors));assert max(errors)<1e-12
    original=[];bridge=[]
    for group in 'ABCD':
        raw=f[(f.group==group)&(f.task=='monitoring')];fusion=f[(f.group==group)&(f.task=='current')]
        key=['fold','seed','case','target_run'];matched=raw.merge(fusion[key],on=key,validate='one_to_one')
        bridge.append({'group':group,'raw138':mae(raw),'raw128_same_targets':mae(matched),'fusion128':mae(fusion)})
        row={'group':group,'current':mae(fusion)}
        for h in (1,2,3):row[f'h{h}']=mae(f[(f.group==group)&(f.task=='future')&(f.horizon==h)])
        original.append(row)
    original=pd.DataFrame(original);bridge=pd.DataFrame(bridge)
    original.to_csv(OUT/'original_complete_method.csv',index=False);bridge.to_csv(OUT/'current_protocol_bridge.csv',index=False)
    # Reproduce the later single-shot/common-pool figures separately.
    future=pd.read_csv(ROOT/'summary/future_predictions.csv')
    fk=['fold','seed','case','origin_run','target_run','horizon'];future_bridge=[]
    for h in (1,2,3):
        q=future[(future.horizon==h)&np.isfinite(future.pred_mm)]
        common=set.intersection(*[set(map(tuple,v[fk].to_numpy())) for _,v in q.groupby('method')])
        for group in 'ABCD':
            raw=f[(f.group==group)&(f.task=='forecasting')&(f.horizon==h)]
            fusion=f[(f.group==group)&(f.task=='future')&(f.horizon==h)]
            selected=raw[[tuple(k) in common for k in raw[fk].to_numpy()]]
            future_bridge.append({'group':group,'horizon':h,'fusion_original':mae(fusion),'single_shot_all':mae(raw),'single_shot_external_common':mae(selected),'fusion_pairs_per_seed':len(fusion)//5,'single_all_pairs_per_seed':len(raw)//5,'external_common_pairs_per_seed':len(selected)//5})
    fb=pd.DataFrame(future_bridge);fb.to_csv(OUT/'future_protocol_bridge.csv',index=False)
    audits['status']='passed';(OUT/'audit.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
    text='''# 结果口径更正：完整融合方法与组件对照

核查日期：2026-11-10。原ABCD表正确；后来的基线汇总使用同一批模型的另一种输出，不是原表的新版本。把组件对照简称为“我们的方法”是不准确的，完整方法主结果应保留原融合表。

原结果：sensor初始化、ODE逐记录传播、当前sensor monitor校正、单调截断。alpha沿用既有内层选择：fold A=1，fold B=.25。当前评价排除初始化，为128点/seed；未来从融合状态出发，排除初始化起点，h1/2/3为114/100/86对/seed。未来传播不使用未来sensor。

后来提取：extract_ours.py明确选择single_shot的monitoring/forecasting，未提取fusion current/future。当前为原始monitor的138点/seed，未来从每个当前monitor估计出发直接积分，不经过递推融合和截断；全部起点124/110/96对/seed。再与LSTM及Kim有限预测求共同目标，缩为100/86/73对/seed。因此同时改变了输出定义和评价集合。

逐键核查2760条当前及6600条未来导出记录，预测差为0，未发现重训、checkpoint替换或数值导出错误。原summary所有overall行从原始预测重算，最大误差小于1.2e-16；case/seed等权聚合方式一致。

## 完整方法主结果（保留原表）

'''+table(original)+'''

## 当前差异分解

raw128在与融合完全相同的128点评价，证明差异不仅来自删除初始化点。

'''+table(bridge)+'''

## 未来差异分解

'''+table(fb)+'''

## 后续表格使用规则

1. 原融合ABCD表继续作为完整方法结果；不以数值更小与否决定口径。
2. summary/RESULTS.md已有138点监测与单次传播表保留为组件比较，不能代替完整方法。
3. 对外部基线作完整方法主比较时，当前需将所有方法对齐同一128点；未来需把融合输出纳入共同目标重新汇总。不能把本表原数值直接拼到旧100/86/73目标的外部表，也不能拿128点融合与138点GPR直接排名。
4. 该更正不需要重训。主表应明确历史信息、初始化、截断、预算与覆盖；外部方法不必被强制放进同一滤波器。
'''
    (OUT/'RESULTS.md').write_text(text,encoding='utf-8')
    print(original.to_string(index=False));print(bridge.to_string(index=False));print(fb.to_string(index=False))

if __name__=='__main__':main()
