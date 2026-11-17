> Archived appendix-writing evidence note. Filenames mentioned below refer to the separate appendix source-data package; not all are distributed in this code-only repository. See ../REPRODUCING.md for executable commands.

# Appendix A — 质量规则和无效观测处理

## A.1 实际质量标准

检查 smcAC、smcDC、vib_table、vib_spindle、AE_table、AE_spindle 六个通道。每条记录先检查缺失／空信号、通道长度不一致、非有限数、任一恒定通道、任一绝对幅值 >10^6；任一触发即无效。另在原始加载阶段隔离 >10^6 的记录，Case2/run1 的最大绝对幅值为 1.044564768186717×10^34。

对每个通道计算 a99=Q.99(|x|)，span=Q.95(x)−Q.05(x)，peak=max|x|，spike=log10(max(peak/a99,1))，以及最高 ceil(.001n) 个平方幅值占总平方幅值比例。所有 log 的下界保护为10^-300。

每个训练划分仅用无硬异常的训练记录校准：

- reference length=训练长度的众数；near-zero epsilon=median(a99)×10^-6。
- 对 log10(a99)、log10(span)，下阈值为 median−max(3,6×1.4826×MAD)。
- 对 spike，上阈值为 median+max(3,6×1.4826×MAD)。
- 通道 degraded：a99 和 span 均低于对应阈值，且至少一半样本满足 |x|<epsilon。
- 通道 spiky：spike 大于上阈值，且最高0.1%样本的能量份额 >.95。
- 记录无效：至少3个 degraded 通道；或至少2个 spiky 通道；或有长度异常，同时至少2个 degraded／至少1个 spiky 通道。

这些阈值在验证／测试冻结。`NASA_frozen_thresholds_all_16_splits.csv` 给出16划分×6通道的数值，`recomputed_quality_audit.json` 确认重新判断与训练配置一致。Case2/run1 数值隔离后，总表166条／145标签；主研究14个case为158条／138标签（另两个case不在主划分中）。

### 可直接用的外层统计（数值隔离后）

|fold|partition|记录|有效|无效|有VB标签|无效且有标签|
|---|---|---:|---:|---:|---:|---:|
|A|train|71|71|0|64|0|
|A|test|87|86|1|74|0|
|B|train|87|86|1|74|0|
|B|test|71|71|0|64|0|

外层互换不是两份独立数据；跨fold拼接测试覆盖为158条，不把train+test重复相加。保留的唯一坏sensor是Case12/run1（无标签）。隔离前Case2/run1也有标签，详见同时保留的 raw_before_quarantine 表。

## A.2 顺序与 fallback

先数值隔离→训练内质量规则→有效训练记录上排名／拟合clip与scaler→记录级monitor／递推。clip是特征维度的1/99百分位限制，不是删除记录，不等同质量判断。

固定融合（x为物理mm状态）的实际伪代码：

```text
for each case, sort retained records by actual machining time:
    raw[i] = sensor_monitor(i) if valid[i] else frozen_CT(i)
    x[0] = raw[0]                       # first retained record, even if unlabeled
    for i = 1 .. N-1:
        prior = integrate_rate(x[i-1], condition[i], time[i]-time[i-1])
        posterior = prior + alpha * valid[i] * (raw[i]-prior)
        x[i] = max(x[i-1], posterior)   # when monotonic projection enabled
```

CT只接受工况和时间，使用自己的冻结scaler，输出换算为mm再转入主模型标准化单位；同split／seed，训练预算500轮。禁止真实测试VB重置。首记录无效时采用CT，后续无效时只传播。主A/D alpha为foldA=1、foldB=.25，单调截断开启；不是新拟合参数。

## A.3 训练 mask 与梯度

|实验|当前监督|未来配对起点|无效记录的处理|
|---|---|---|---|
|Warm ABCD|有标签且sensor有效|代码按有标签起点建h1/2/3对，未另按valid删起点|若遇无效起点用冻结CT；C不能通过该起点更新monitor，D仍可更新ODE|
|Cold机制消融|当前／局部均有标签且有效|另要求起点sensor有效；终点只需VB标签|后备冻结，不纳入当前／局部监督|
|部署递推|不是训练损失|从校正状态预测，未来无sensor|坏sensor gain=0；不打断真实时间|

现有14case标签池中的保留坏记录没有VB标签，所以 warm 与 cold 该项代码差异未造成实际有标签未来起点覆盖差异。不能把一般分支描述当成已经有异常有标签起点的实验验证。

## A.4 Stiehl

Stiehl使用自己的三轴力／时间戳完整性检查，所有6418信号通过；51维特征均有限。不要沿用NASA六通道阈值的数量或声称测试了Stiehl无效sensor fallback。参数和逐文件结果见 B_implementation/Stiehl/prepare_features.py、sensor_quality.csv、feature_audit*.json。
