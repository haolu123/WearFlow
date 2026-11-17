from pathlib import Path
import pandas as pd,json
root=Path(__file__).parent/'results'; d=pd.read_csv(root/'summary.csv')
def val(arm,task,reg):return float(d[(d.arm==arm)&(d.task==task)&(d.region==reg)]['mean'].iloc[0])
def table(region):
    s='|方法|当前VB|未来h1|未来h2|未来h3|\n|---|---:|---:|---:|---:|\n'
    for arm in ('current_only','local_only','integral_only','both'):
        tasks=['current']+[(('persistence_h' if arm=='current_only' else 'h')+str(h)) for h in (1,2,3)]
        s+='|'+arm+('（未来为persistence）' if arm=='current_only' else '')+'|'+'|'.join(f'{val(arm,t,region):.5f}' for t in tasks)+'|\n'
    return s
text='''# 随机起点四臂机制消融：固定100epoch

96次训练全部完成。每臂同split/seed的monitor与rate随机初始化完全相同，不加载monitor100或ODE500。100轮AdamW lr=.001、wd=.0001，local与integral系数预设1；无外层挑参数。96×100训练及逐epoch评估累计553.15秒（9.22分钟），运行总墙钟以status.json为准；报告另耗约4秒。

## 主要结果
积分监督显著改善了未来传播，但没有改善当前VB。当前MAE以current-only最低；integral-only当前比它高约.00677mm，case配对95%区间跨0。不能写成“当前与未来均提升”。在本地偏导定义和固定权重下，integral-only的h1/h2/h3均优于local-only，配对区间均不跨0；both没有超过integral-only，尤其h2/h3更差。

current-only的rate没有训练，其随机rate积分只作诊断。本主表未来使用persistence（保持当前monitor估计不变）作为实际基准。其余三臂未来使用学习到的rate积分。单位mm，seed先case内平均再case等权；h3只有13case，因为case16没有第三个后续标签，h1/h2为14case。

'''
for region,title in [('overall','总体'),('fold_a','Fold A'),('fold_b','Fold B'),('A_in','Fold A 目标真实VB≤.74'),('A_out','Fold A 目标真实VB>.74')]:text+='## '+title+'\n\n'+table(region)+'\n'
text+='''## 配对证据与解释

bootstrap按fold内case重采样，先在case内平均5seed，10,000次、rng20261103。负差表示integral-only更低。

- 对local-only：h1差−.02045，95%区间[−.03651,−.00785]；h2差−.06650，[−.09554,−.03970]；h3差−.09290，[−.13060,−.05686]。
- 对current-only persistence：h1差−.02538，[−.04007,−.01237]；h2差−.09233，[−.11769,−.06754]；h3差−.14165，[−.17377,−.11078]。
- 当前对current-only：差+.00677，[−.00175,+.01650]；当前对local-only也没有可靠改善。

这里的local是固定sensor与condition的partial M/partial time，不是随sensor变化轨迹的全导。当前监督可以通过sensor输入表达磨损趋势，局部时间偏导不一定代表真实总磨损率。因此本实验支持“未来状态监督对本框架传播学习有价值”，不能推导所有PINN局部物理约束无效。相同lambda并非相同梯度强度；首末轮梯度与余弦已保存。

当前误差、未来误差可能存在折中，A超训练磨损范围误差仍大。真实初值oracle另列summary.csv的oracle_h*，不能混作实际预测。预测准确不保证rate唯一可识别；少量稀疏标签及monitor-rate补偿仍存在。已多次观察测试、质量规则亦为探索性方案，需如实声明。

## 审计与文件

- 96run、每run100次更新、epoch0..100共101份当前/未来/oracle预测；四臂初始化hash、epoch0预测、训练obs/pairs及目标键一致。
- 每seed外层当前138点；h1/h2/h3实际配对数由每run的pairs记录，不能按138套用。
- 训练局部偏导仅有效sensor记录；积分训练起点有效、终点真实标签，真实dt且同case/工况。无未来sensor输入。
- current-only rate精确不变，其余三臂更新；所有末轮current/future/oracle重载一致；独立agent重算逐epoch指标并重载96checkpoint通过。
- train/validation/test每epoch指标和原始预测均保留。CT后备来自共同已训练模型且hash记录；本外层138点均有效，主要结果不依赖后备预测。

`final_predictions.csv.gz`含fold/case/origin_run/target_run/horizon/dt/target_mm/pred_mm，task前缀明确oracle与persistence。`summary.csv`、`case_seed_metrics.csv`、`case_mean_metrics.csv`保存全部范围和细节；`case_paired_bootstrap.csv`含随机rate诊断和部署persistence对照的明确标签；`run_audit.csv`、`report_audit.json`、`status.json`保存审计与耗时。独立审计在suite/zhang目录。

warmstart ABCD研究适配收益，不能替代此随机起点机制对照，不能把两者混为相同预算。
'''
(root/'RESULTS.md').write_text(text,encoding='utf8')
print((root/'status.json').read_text())

