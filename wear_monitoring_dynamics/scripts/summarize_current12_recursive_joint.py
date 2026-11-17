"""Chinese narrative supplement; consumes completed audit, never trains."""

import json

import pandas as pd

from .audit_current12_recursive_joint import ROOT

from .report_current12_fusion import md



def main():

    status=json.loads((ROOT/'status.json').read_text());audit=json.loads((ROOT/'independent_audit.json').read_text());assert audit['status']=='passed'

    now=pd.read_csv(ROOT/'current_selected_summary.csv');fut=pd.read_csv(ROOT/'future_selected_summary.csv');mon=pd.read_csv(ROOT/'monitor_selected_summary.csv');sel=pd.read_csv(ROOT/'inner_selection.csv')

    boot=pd.read_csv(ROOT/'case_bootstrap.csv');halves=pd.read_csv(ROOT/'trajectory_halves.csv');oracle=pd.read_csv(ROOT/'oracle_summary.csv');di=pd.read_csv(ROOT/'monotonic_case_diagnostics.csv');curve=pd.read_csv(ROOT/'selected_epoch_curves.csv.gz')

    currentcase=pd.read_csv(ROOT/'current_selected_cases.csv').groupby(['fold','case','method']).mae.mean().unstack('method').reset_index()

    currentcase['recursive_minus_frozen']=currentcase.selected_recursive-currentcase.selected_frozen

    futurecase=pd.read_csv(ROOT/'future_selected_cases.csv').groupby(['fold','case','horizon','method']).mae.mean().unstack('method').reset_index()

    futurecase['recursive_minus_frozen']=futurecase.selected_recursive-futurecase.selected_frozen

    currentcase.to_csv(ROOT/'current_case_comparison.csv',index=False);futurecase.to_csv(ROOT/'future_case_comparison.csv',index=False)

    fixednow=pd.read_csv(ROOT/'current_summary.csv');fixedfuture=pd.read_csv(ROOT/'future_summary.csv')

    diag=di.groupby('method')[['drop_fraction','mean_drop_mm','max_drop_mm','clamp_fraction','mean_clamp_mm']].mean().reset_index()

    curve_end=curve[curve.epoch.isin([1,500])].groupby(['method','task','split','epoch']).ae.mean().reset_index()

    p=pd.read_csv(ROOT/'selected_predictions.csv.gz');counts=p[p.seed==0].groupby(['method','task','horizon']).size().reset_index(name='n')

    counts.to_csv(ROOT/'target_counts.csv',index=False)

    tr=pd.read_csv(ROOT/'selected_trajectories.csv.gz');tr=tr[(~tr.initial)&tr.target_vb_mm.notna()].copy();tr['error']=tr.predicted_vb_mm-tr.target_vb_mm

    drift=tr.groupby(['method','fold','seed','case','half']).error.mean().unstack('half').dropna().reset_index();drift['late_minus_early_bias_mm']=drift.second-drift['first'];drift.to_csv(ROOT/'trajectory_bias_drift_cases.csv',index=False)

    text='''# 完整 case 传播—校正联合训练结果



这轮实验不支持继续采用当前配置的递推联合微调。内层验证选出的 recursive，在当前 VB 和未来 h1/2/3 上均比冻结起点差，也比同预算 single 差；不能把“已经端到端”本身视为性能改进。原始 monitor 当前 VB 大体持平，主要退步发生在加入传播后的输出。结论限定于本批损失、初始化、训练预算和数据划分，不等于所有递推训练方案都无效。



## 方法与对照



- frozen：冻结上一批固定 joint_l1 的同 split/seed 模型，作为本批 epoch0；此前已做 500 轮预训练和 500 轮单次联合微调。

- single：同起点另训练 500 轮，训练中只做单次传播；训练后用相同递推校正规则评估六组 alpha/截断。

- recursive：同起点另训练 500 轮，训练中完整展开 case 的传播—校正，monitor 和 ODE 都更新，中间状态无 detach。



损失是 case 等权 raw monitor 当前监督＋输出当前监督＋输出未来 h1/2/3 监督，权重各 1。single 将输出状态替换为 raw monitor，用于控制当前监督权重与标签池；不是上一批 M+F 原损失直接续训。两类新训练更新次数相同，但计算量不同；frozen 没有额外训练预算。



所有观测均为工况＋时间＋训练内相关性前12维电流，clip_zscore、MLP[64,32]；质量规则、排名、scaler 和 CT 后备沿用同 split。first_record 初始化，无真实 VB 重置，无效 sensor 跳过校正。未来预测不读取未来 sensor。h1/2/3 指后续有标签记录，积分使用真实时间间隔。



当前融合评估每 seed 128 点，排除初始化；原始 monitor 138 点单列，不直接比较两个标签池的 MAE。未来各 horizon 也使用全方法一致的目标键。MAE 单位 mm，先 case 内平均，再 case 等权；不同 horizon 可用 case 数不同，总体不一定是 A/B 简单平均。



## 内层选择与主要结果



每个 family/fold 独立用七个内层验证 case 的当前融合 MAE 选择 alpha/截断；五个外层 seed 共用该选择，未来不参与选参。三类本次恰好都选 A: alpha=.75、不截断；B: alpha=.25、不截断，因此主要对照也具有相同融合配置。



'''+md(sel[sel.selected])+ '\n\n每seed配对数量（各方法一致）：\n\n'+md(counts[counts.method=='selected_frozen'][['task','horizon','n']])+ '\n\n### 当前融合 VB\n\n'+md(now[['method','subset','mae','bias']])+ '\n\n### 未来 VB\n\n'+md(fut[['method','horizon','subset','mae','bias']])+ '\n\n### 原始 monitor 当前 VB（138点）\n\n'+md(mon[['method','subset','mae']])+'''



recursive 的 raw monitor 总体 MAE 0.08778，与 frozen 0.08785 几乎相同；融合后却从 0.08972 升至 0.09414。不能因此认定 sensor 输入本身退化。未来 fold A 的高磨损部分退步尤其明显，例如 h3 从 0.51653 增至 0.81490；这些都是少量 case 的探索性结果。



## 不确定性与逐 case



以下为差值“test−reference”，正值代表退步；每个 case 先平均五 seed，再在 fold 内成对重采样 case 5000 次，报告百分位 95% 区间。没有把 seed 当独立样本，也没有做多重比较校正；区间覆盖0时不声称稳定差异。训练集在不同折之间重叠，因此区间也不是独立重复采样整个学习流程的严格总体推断。



'''+md(boot[(boot.subset=='overall')&(boot.task!='monitor')])+ '\n\n当前逐case，五seed平均：\n\n'+md(currentcase)+ '\n\n未来h3逐case：\n\n'+md(futurecase[futurecase.horizon==3])+'''



当前 VB 在14个case中的10个改善，但case5/13/16退步较大；未来h3退步主要集中case13和5。不能把总体退步描述成每个case都更差。



完整各 horizon、fold、范围分层区间见 case_bootstrap.csv；逐 seed 信息保留在 current_selected_cases.csv / future_selected_cases.csv。



## 轨迹后半段、偏移和单调截断



按每个 case 从首条到末条记录的真实时间中点划分前后半段，排除初始化，缺标签记录不计 MAE。两半段的目标分布不同，因此后半误差增大只作漂移诊断，不构成累积误差的因果证明。



'''+md(halves[halves.subset=='overall'][['method','half','mae','bias']])+ '\n\n后半减前半的有符号偏差，先每case计算再平均：\n\n'+md(drift.groupby('method').late_minus_early_bias_mm.mean().reset_index())+ '\n\n所有固定配置的全记录轨迹单调诊断（每case/seed等权）：\n\n'+md(diag)+'''



前半段MAE由0.06009降至0.05419，但后半段由0.11274升至0.12537，提示局部早段改善没有延续到完整轨迹。



下降幅度的均值包含未下降步的0；max_drop_mm为各case最大下降幅度的平均。clamp_fraction为实际触发投影的步比例。单调投影可以保证递推均值不下降，但不会纠正已经偏高的状态，也不保证准确率提升。本次全部内层选择均为不截断；固定截断对照完整保留，不能从测试选一个截断配置替换预先定义的内层选择。软/硬 max 可能屏蔽部分路径梯度，首末梯度均已保存，未强制每条路径非零。



![逐case轨迹](case_trajectories.png)



## 实测初值 ODE 诊断



用同一批未来配对的真实当前 VB 作为 ODE 初值（oracle），不做 sensor 校正。此诊断不可部署，只用于排查传播场是否改变；仍用内层当前 MAE 选定的模型，不按 oracle 挑模型。



'''+md(oracle[oracle.method.str.startswith('selected_')][['method','horizon','subset','mae','bias']])+'''



oracle总体h3从frozen的0.10531升至recursive的0.12767，说明用实测初值消除monitor初值误差后，传播性能仍退步；并非全部由sensor初值导致。它不证明唯一退化机制，未扫描新的正则或训练预算。



## 每轮曲线



递推模型的训练集当前MAE从第1轮0.01743降至第500轮0.01328，而测试当前从0.09023升至0.09414；训练未来平均误差从0.02258降至0.01564，测试从0.13466升至0.16141。这个方向符合训练拟合改善、泛化退化的表现，但不由此断言单一原因。



以下曲线只用于诊断，固定第500轮出结果，不按外层测试回选 epoch。未来曲线是各case全部horizon配对的平均，不等于各horizon MAE简单平均。frozen 对应共同 epoch0，可在 predictions.csv.gz 中追溯；曲线展示两个实际训练 family。



![逐轮曲线](epoch_curves.png)



'''+md(curve_end[curve_end.task.isin(['current','future'])])+ '\n\n## 固定网格（与内层选择分开）\n\n当前总体：\n\n'+md(fixednow[(fixednow.subset=='overall')&~fixednow.method.str.startswith('selected_')][['method','mae']])+ '\n\n未来总体：\n\n'+md(fixedfuture[(fixedfuture.subset=='overall')&~fixedfuture.method.str.startswith('selected_')][['method','horizon','mae']])+'''



其他分层见 current_summary.csv / future_summary.csv。固定网格不用于宣称外层最优配置；未因测试结果扩网格或启动下一阶段。



## 审计、耗时与文件



'''+f"168次训练、84000个更新epoch全部完成；另有每运行epoch0。独立核查{audit['raw_prediction_rows']:,}行逐轮原始预测：每epoch键/目标唯一且一致、有限预测、训练/验证/测试指标由原始记录重算一致；epoch0与500重载并重算全部split输出。24个共同起点下各配置epoch0一致，父checkpoint/执行源码hash未变。核查16个训练划分的前12维排名与scaler、CT同split/seed后备、内层排除、正尺度换算、因果、alpha0/1端点、强制无效sensor跳过、单调轨迹和全case梯度。外层每轮checkpoint共35000个（70次外层训练×500）。\n\n训练累计{status['training_seconds']/3600:.3f}小时（{status['training_seconds']/60:.2f}分钟），基础汇总{status['report_seconds']:.2f}秒，独立审计及附加评估{audit['audit_seconds']:.2f}秒。使用CPU单线程，未重训。独立结果见 independent_audit.json / independent_run_audit.csv；质量规则从原始数据重拟合以及物理单位RK4独立检查另见 input_quality_units_audit.json。single每次平均144.20秒，recursive每次平均271.92秒，因此同500更新不代表同墙钟成本。\n\n"+'''

复现审计：`python -m wear_monitoring_dynamics.scripts.audit_current12_recursive_joint`；输入审计 `python -m wear_monitoring_dynamics.scripts.audit_recursive_inputs`；再运行 `python -m wear_monitoring_dynamics.scripts.summarize_current12_recursive_joint` 生成本文。原入口 --report-only 会覆盖本文为基础摘要，随后需重新运行 summarize。



逐case轨迹含无标签记录见 all_record_trajectories.csv.gz；半段误差见 trajectory_halves.csv；原始每轮记录在各run/epoch_predictions.csv.gz，训练损失在 history.csv；选择清单 inner_selection.csv。所有结果属探索性，先前已多次查看测试集。保留阴性结果；当前没有证据支持用本批递推联合训练替换原冻结模型。下一阶段未启动。

'''

    (ROOT/'RESULTS.md').write_text(text,encoding='utf-8')

    print('Chinese report saved')

if __name__=='__main__':main()

