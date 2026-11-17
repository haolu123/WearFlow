# Sun 2020 LSTM 预测模块：NASA 适配与结果

已完成固定100epoch、2fold×5seed，共10次训练。本包是**预测模块迁移**，不是整篇 ResNet+LSTM+校正系统的复现。

## 来源

Huibin Sun, Jiduo Zhang, Rong Mo, Xianzhi Zhang. In-process tool condition forecasting based on a deep learning method. RCIM64(2020)101924. DOI10.1016/j.rcim.2019.101924。
https://www.sciencedirect.com/science/article/pii/S0736584519303655
作者仓库 https://github.com/kidozh/rnn_in_forecasting_tool_wear
固定commit：32bc9297003981d69faf6a54bd91f2a8122250db。source/保存4个原始文件，manifest.json含来源URL及SHA256。

## 保留与适配

保留source/model.py的encoder-decoder：LSTM64(sequence)→LSTM32(last)→Dense32→RepeatVector(H)→LSTM32(sequence)→LSTM16(sequence)→Dense1。保留logcosh/Adam。

- 原main_separate_with_monitoring使用history2/output5，本轮固定history2/output3。
- 训练输入为真实历史VB，目标为后续3个真实VB；只使用训练case的完整2+3有标签窗口。
- 部署输入是统一sensor100 monitor的两条历史预测，**不输入测试真实VB**。单列oracle使用两条实测历史，仅诊断。
- 不输入时间或工况，符合此预测模块原始输入；实际dt只用于报告。这与条件化ODE的信息集有差别，不能把差异全归因积分。
- 2条历史取有标签记录位置；主比较h1/h2/h3为后续有标签记录步数，非固定时长。预测仍按记录步数，因此不宣称任意时间能力。未来不足3条时，评价已有h，不填充标签；训练要求完整3条。
- 迁移到PyTorch，采用Keras式Glorot输入、orthogonal recurrent、有效forget bias+1；随机初始化不声称与旧Keras同seed逐值一致。
- Adam lr=.001、eps=1e-7、batch16、固定100epoch；源代码可用5000epoch，本轮100不代表原文充分收敛配置。
- logcosh在split-specific y_z上计算，而原代码目标尺度不同。这会改变损失形状相对mm误差的强度，属于明确NASA适配。
- 未来损失仅更新LSTM；公共monitor冻结，未实现原文ResNet或均值/中位数校正。

## 主要结果

下表为每case先平均5seed、再case等权MAE(mm)，固定epoch100；使用共同monitor估计历史。

| 范围 | h1 | h2 | h3 |
|---|---:|---:|---:|
| overall | .129963 | .138033 | .161326 |
| fold A | .176113 | .161731 | .189164 |
| fold B | .083814 | .117721 | .137465 |

覆盖每seed h1=110对/14case，h2=96对/13case，h3=83对/13case。前缀历史不足的起点不能评价；不能与包含更多起点的旧结果直接排名。

在完全相同目标上，persistence MAE为.130706/.195159/.272885。该比较仅显示本协议结果，不构成与论文原成绩或所有ODE方法的结论。oracle总体为.114593/.141032/.189847；真实历史不是保证最优的误差下界，部署误差可能抵消动力学/模型偏差。

## 审计

NumPy显式LSTM门方程与PyTorch前向最大误差8.13e-20；输入梯度有限差分误差3.68e-14；未来monitor扰动不改变当前历史输入；大误差logcosh有限。
这是方程级迁移检查，未在旧Keras runtime执行原模型，不冒称原框架逐值复现。

10个run保存epoch0..100全部训练/测试h1..3原始预测，估计历史和oracle分开；合计583780行。无非有限值，dt均正。全部last checkpoint重载误差0，首步梯度非零。与公共reference的origin/target/horizon/dt/真实目标匹配通过。

完整运行19.51秒（run内18.20秒，不含Python启动）；独立审计另计。每方法选择不使用测试结果，无超参搜索，固定最后epoch100。

## 文件与运行

- model.py、run_sun_lstm.py：实现和入口。
- source/：固定来源代码/hash。
- smoke/：1个内层case，3epoch，仅真实历史oracle烟雾，不混入正式表。
- results/runs/<fold>/seed*/：identity.json、epoch0.pt、last.pt、history.csv、epoch_predictions.csv.gz、final_predictions.csv、audit.json。
- results/future_predictions.csv：部署未来结果，含method/fold/seed/case/origin_run/target_run/horizon/dt/target_mm/pred_mm。
- results/final_predictions.csv：含训练/测试与oracle。
- results/summary.csv、case_seed_metrics.csv、coverage.csv：结果与覆盖。
- results/reference_comparison_common_targets.csv：同目标persistence比较。
- results/equivalence_audit.json、independent_audit.json、status.json：验证与耗时。

    D:/anaconda3/envs/CondWear-ODE/python.exe <本目录>/run_sun_lstm.py --smoke
    D:/anaconda3/envs/CondWear-ODE/python.exe <本目录>/run_sun_lstm.py

重跑覆盖本包结果，不改变共享数据。读取主目录data_manifest和references冻结文件；仅使用CPU单线程。

