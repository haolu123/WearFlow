"""Complete epoch pairing checks and gate study figures/report."""
from wearflow_paths import ROOT as _WF_ROOT
import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wear_monitoring_dynamics.config import write_json
R=Path(str(_WF_ROOT / 'wear_monitoring_dynamics/outputs/current_gate_20261012'));m=pd.read_csv(R/'study_manifest.csv');p=pd.read_csv(R/'test_predictions.csv');reference=p[p.name=='ct_baseline'];ref=reference[reference.seed==0].set_index(['case','run']).vb_mm.to_dict();assert len(ref)==138
for (name,seed),q in p.groupby(['name','seed']):
 assert len(q)==138 and not q.duplicated(['case','run']).any();assert {(c,r):v for c,r,v in zip(q.case,q.run,q.vb_mm)}==ref
nrows=0
for row in m.itertuples():
 d=Path(row.run_dir);split=json.loads((d/'split.json').read_text());e=pd.read_csv(d/'epoch_predictions.csv.gz');nrows+=len(e)
 assert e.task.eq('monitoring').all() and (e.origin_run==e.target_run).all()
 for group,q in e.groupby('split'):
  cases=split['outer_test_reserved'] if group=='test_diagnostic' else split[group];target={key:v for key,v in ref.items() if key[0] in cases};first=q[q.epoch==1];assert {(c,r):v for c,r,v in zip(first.case,first.target_run,first.target_vb_mm)}==target
  assert not q.duplicated(['epoch','case','target_run']).any();assert q.groupby(['case','target_run']).size().eq(500).all();assert q.groupby(['case','target_run']).target_vb_mm.nunique().eq(1).all();assert set(q.epoch)==set(range(1,501))
write_json(R/'pairing_audit.json',dict(passed=True,runs=len(m),epochs=144000,prediction_rows=nrows,outer_targets=138))
g=pd.read_csv(R/'final_gate_values.csv.gz');sel=pd.read_csv(R/'selected_lambdas.csv');chosen=pd.concat([g[(g.fold==r.fold)&(g.name==r.name)].assign(kind=r.kind) for r in sel.itertuples() if r.kind in ['static','condition','time_condition']]);chosen=chosen[(chosen.split=='test')&chosen.sensor_valid]
stab=chosen.groupby(['kind','fold','case','run','feature']).gate.agg(['mean','std','min','max']).reset_index();stab.to_csv(R/'selected_gate_seed_variability.csv',index=False)
for (kind,fold),q in chosen.groupby(['kind','fold']):
 fig,axs=plt.subplots(6,3,figsize=(15,19),layout='constrained')
 for ax,(feature,z) in zip(axs.flat,q.groupby('feature')):
  for case,b in z.groupby('case'):
   v=b.groupby('time').gate.agg(['mean','std']).reset_index().sort_values('time');ax.plot(v.time,v['mean'],label=str(int(case)));ax.fill_between(v.time,v['mean']-v['std'],v['mean']+v['std'],alpha=.08)
  ax.set(title=feature,xlabel='Dataset time',ylabel='gate',ylim=(0,1));ax.grid(alpha=.2)
 axs.flat[0].legend(title='Test case',fontsize=7,ncol=2);fig.suptitle(f'{kind}, {fold}: validation-selected lambda; mean +/- seed SD');fig.savefig(R/f'gates_{kind}_{fold}.png',dpi=110);plt.close(fig)
rows=[]
for row in m[m.role=='final'].itertuples():
 h=pd.read_csv(Path(row.run_dir)/'history.csv');a=h.iloc[-1];rows.append(dict(name=row.name,fold=row.fold,seed=row.seed,mean_gate=a.gate_train_mean,fraction_lt01=a.gate_train_fraction_lt_01,input_weight_norm=a.gate_train_input_weight_norm,effective_weight_norm=a.gate_train_effective_input_weight_norm))
gs=pd.DataFrame(rows);gs.to_csv(R/'gate_training_summary.csv',index=False)
a=pd.read_csv(R/'feature_zeroing_sensitivity.csv');az=pd.concat([a[(a.fold==r.fold)&(a.name==r.name)].assign(kind=r.kind) for r in sel.itertuples() if r.kind in ['static','condition','time_condition']]);az.groupby(['kind','fold','feature']).delta_mae.mean().reset_index().to_csv(R/'selected_feature_sensitivity.csv',index=False)
case=pd.read_csv(R/'case_metrics.csv');case=case[case.name.isin(['ungated','ct_baseline','selected_static','selected_condition','selected_time_condition'])].groupby(['case','name']).mae.mean().unstack();case.to_csv(R/'selected_case_mae.csv')
fig,ax=plt.subplots(figsize=(14,5),layout='constrained');case.plot.bar(ax=ax);ax.set(ylabel='MAE (mm)',title='Fixed epoch 500; lambda selected by inner validation');ax.grid(axis='y',alpha=.2);fig.savefig(R/'selected_case_mae.png',dpi=150);plt.close(fig)
def md(df):
 return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])
summary=pd.read_csv(R/'summary.csv');brief=summary[summary.name.isin(['ungated','ct_baseline','selected_static','selected_condition','selected_time_condition','validation_selected_all'])].pivot(index='name',columns='subset',values='mae').reset_index();variability=stab.groupby('kind')['std'].agg(['mean','max']).reset_index();sens=az.groupby('kind').delta_mae.agg(['mean','min','max']).reset_index()
print(variability.to_string(index=False));print(sens.to_string(index=False))
text='''\n\n## 完成结论与审计

288次训练均500epoch，累计训练38.80分钟。配置/训练内scaler/同split18维及后备、不同gate和lambda同seed预测器初始化重建、checkpoint重载审计通过。额外核查全部144000epoch的逐点记录、每个epoch标签键与split一致、逐方法逐seed138外层目标配对通过（19872000行逐轮预测）。final共120次、60000个逐epochcheckpoint。初始化检查是按原调用顺序重建，不是保存epoch0快照。

三种gate在当前网格内均未超过原无gate电流模型。验证选择的总体静态0.09055、工况0.09559、时间工况0.09504，原电流0.08950，ct0.08349。12个固定配置的总体误差也均高于原电流；不能据此否定所有gate设计，但这版没有获得预期收益。跨架构含基线内层选择为fold A condition/lambda0、fold B static/lambda0.1，总体0.09705。内层偏好与外层最优并不一致。

'''+md(brief)+'''\n
## 正则强度与门控行为

动态gate在lambda0.1确实受到明显压缩：训练点上condition平均gate0.133，time_condition0.100，低于0.1的比例分别42.9%和64.6%。静态gate平均0.436，没有低于0.1的项，未形成明显稀疏选择。

time_condition从lambda0到0.1，平均输入权重范数4.600升至4.832，但gate乘平均输入权重的范数2.432降至0.749。这说明权重有一定增大，不能说惩罚完全被抵消；该指标仅第一层平均gate近似，不能排除更深层补偿。lambda0.1的动态结构比自身lambda0总体略好，但仍差于无gate模型；其他小lambda没有稳定收益。

'''+md(gs.groupby('name')[['mean_gate','fraction_lt01','input_weight_norm','effective_weight_norm']].mean().reset_index())+'''\n
## 逐case影响

对3/4/10/11，condition gate的MAE分别0.0685/0.0634/0.0885/0.0934，较原电流0.0686/0.0742/0.0902/0.0970有部分改善，但仍差于ct。收益被其他case退化抵消：condition下case13由0.1435到0.1808，case14由0.1446到0.1680，case16由0.1221到0.1426，case5由0.0602到0.0741。不是所有原来改善的case都保住优势。

'''+md(case.reset_index())+'''\n
## 门控图、seed稳定性与实际依赖

gates_{static,condition,time_condition}_{fold_a,fold_b}.png为验证选择模型的18维gate随真实记录时间曲线，每case一条线，阴影为五seed标准差。condition不读time，同case应为平线；static对全部case相同。曲线只作解释，不作为测试筛选依据。不同fold特征名单不同，不直接拼接gate。

以下为同case/记录/特征gate在五seed间的标准差汇总；不是统计置信区间：

'''+md(variability)+'''\n
逐特征关闭采用把标准化sensor输入设为0，保持原模型和其他输入不变；相当于该特征路径不再传递输入。完整逐case/seed结果见feature_zeroing_sensitivity.csv，delta_mae>0表示关闭后变差，<0表示关闭后反而改善。它是已有模型的敏感性，不是重新训练的消融，也不等于可迁移的因果重要性。不能只看gate值排名，应结合该文件和input_weight_norms.csv判断实际依赖。

未新增lambda、未依据测试更改特征、未启动下一批。所有结论为已接触外层测试后的探索性结果。
'''
path=R/'RESULTS.md';base=path.read_text(encoding='utf-8').split('\n\n## 完成结论与审计')[0];path.write_text(base+text,encoding='utf-8')
print('Report and plots complete')

