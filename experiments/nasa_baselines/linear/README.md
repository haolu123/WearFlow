# 两点线性磨损趋势：内部简单未来预测基线

这是预设公式的内部对照，不是文献方法复现。无新增训练、无参数搜索、不用测试结果选择版本。两个固定版本全部报告：

- `linear2_sensor100`：原始斜率，不截断斜率或预测值。
- `linear2_nonnegative_sensor100`：仅将斜率限制为非负，不对预测状态另做投影。

使用共同 sensor100 monitor 在同一 case 最近两个有标签记录位置的**估计** VB。位置与 Sun LSTM 的两点历史协议相同；真实 VB 只用于确定既定评价记录及计算误差，不作为历史输入。monitor 已沿用共同协议的无效 sensor 后备。后续 h1/2/3 按有标签记录步数定义，尾部允许不足三个目标。此历史选择依赖既定标签采样位置，不能解释为在全部原始记录上每步更新的在线算法。

令两条历史记录为 `(t_prev, VBhat_prev)`、`(t_i, VBhat_i)`：

`slope = (VBhat_i - VBhat_prev) / (t_i - t_prev)`

`VBhat_j = VBhat_i + slope * (t_j - t_i)`。

非负版本替换 `slope` 为 `max(slope, 0)`。使用数据原始时间单位，不把 h 当时间；不使用未来 sensor、未来真实 VB、未来工况。与不输入时间的 LSTM 相比，本基线明确使用真实采样时间间隔，因此需注明信息差异。

## 结果与覆盖

五 seed 首先在每个 case 内平均 MAE，然后对 case 等权平均；seed 不是独立 case。两个版本均覆盖每 seed h1=110、h2=96、h3=83 个目标（14/13/13 case），与 LSTM 精确同键。

| 固定版本 | h1 MAE mm | h2 MAE mm | h3 MAE mm |
|---|---:|---:|---:|
| 原始斜率 | 0.114392 | 0.143005 | 0.172395 |
| 非负斜率 | 0.114049 | 0.141949 | 0.170286 |

`summary.csv` 提供 overall/fold A/fold B；`case_seed_metrics.csv` 保留逐 case/seed；`future_predictions.csv` 包含目标键、真实 dt、历史估计和斜率。所有结果均为 test/estimated，未混入真实初值 oracle。

## 可复现与审计

```powershell
& 'D:/anaconda3/envs/CondWear-ODE/python.exe' 'experiments/nasa_baselines/linear/run_linear.py'
```

`audit.json` 保存输入文件 SHA256、耗时及验证结果：解析公式样例（包括原始版本允许负预测）、全部 origin 的未来 sensor/未来标签扰动不变性、monitor 与数组键对齐、LSTM 覆盖精确一致、persistence 同键真实 dt/目标一致。无模型 checkpoint，因为没有拟合参数。脚本不修改共享汇总器。
