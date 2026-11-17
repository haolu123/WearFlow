# Zhang PINN 的 NASA milling 迁移

## 来源与范围

公开代码：https://github.com/zhangpengfei247/Tool-wear-monitoring-boring-PINN

固定 commit：`054ec32111b1efc63eabb6016485e1543095b736`。
原始仓库保存在项目根目录 `external/zhang_boring_pinn/`。主要依据 notebook `2. Dataloader construction and PINN model .ipynb`，不是根据摘要重写。
对应论文：Tool wear monitoring and multi-step forecasting in deep hole boring based on physics-informed neural network and ARIMA model，DOI 10.1016/j.ymssp.2026.114078。

本研究是作者公开 PINN **当前 VB 监测实现**的迁移，不声称重现原深孔镗削数据的成绩；不含 ARIMA。
NASA 数据输入和评估必须适配，下面逐项记录。原始 notebook 没有整体执行，避免依赖作者本机路径和缺失的 utils/loader 模块。

## 保留的实现

- solution_u：输入 → Linear(32) → Sin → Linear(32) → Sin → Dropout(0.2) → Linear(32)；随后 Dropout(0.2) → Linear(30) → Sin → Linear(1)。线性层 Xavier normal，监测网络 bias 为 0。
- dynamical_F：4 个线性层、隐层宽度 50、Sin 和 Dropout(0.2)，末层输出标量；按作者 MLP 的初始化方式保留 F 的默认 bias。
- F 输入按作者顺序拼接 `[xt, u, u_x, u_t, u_tt, u_xx, u_xt]`；维度由原始硬编码 80 改为 `3 * input_dim + 2`。
- 重要：`u_xx = grad(u_x.sum(), x)`，是 Hessian 的收缩，并非逐维二阶导数对角线；`u_xt = grad(u_x.sum(), t)` 是一个标量列，不是所有混合偏导组成的向量。保留实际代码运算，未照抄其不准确注释。
- PDE 残差 `u_t - F`；相邻两点的监督 MSE 和残差 MSE 各取 0.5 加权。
- 方向损失 `sum(relu(-(u2-u1)*(y2-y1)))`；alpha=1，beta=0.75。它利用训练标签的变化方向，**不是无条件要求 VB 单调增加**。NASA 标签若下降，该损失也偏好相应下降。
- 保留 train 模式下作者 PINN 的 dropout，包括自动微分计算；不擅自改为另一种 PINN。
- Adam；监测网络初始 lr=0.01，10 轮线性过渡到 0.003，再 cosine 至 0.0001；F 的 lr=0.001。严格沿用代码中 epoch 结束后 scheduler.step 的调用顺序。该设置来自 notebook，不能与论文文字中的其他学习率数值混用。

## NASA 适配及公平对照

1. ct 输入为 4 维工况 + 1 维真实加工时间；sct 再增加现有 54 维传感器统计特征。输入维度分别 5、59；F 输入维度分别 17、179。
2. 原代码使用生成的 cycle index，这里使用 NASA 的真实时间。原代码对每个完整 CSV 归一化，这里沿用每个训练 case 集合拟合的 scalers，验证/测试不参与拟合。
3. 输出为标准化 VB，评价时严格反变换为 mm。原代码 nominal_capacity=1，即原始 VB 单位；本适配改变了损失尺度，alpha/beta 的数值保留不意味着物理单位下约束强度相同。所有 12 个监测配置共用相同输出尺度。
4. 不把真实当前 VB 输入监测网络。真实标签只用于训练目标；冻结 ODE 的未来预测初值始终是预测 VB。
5. 原 notebook 的随机相邻样本对划分、全数据测试 loader 和重复端点评价不迁移；沿用两折 case 隔离、内部 7 留一 case 验证、每折 5 个最终种子。评价每个有标签记录一次。
6. 相邻 pair 为同 case 内按时间排列的相邻有效标签，缺失 VB 不插值；现有数据质量规则统一隔离 case 2/run 1，不改原始文件。pair 必须有正时间间隔。所有监测组共用 pair 监督与 batch size 16，相同 Adam 日程，避免只改变某个组的端点权重或更新次数。
7. 固定 500 epoch，不启用 early stopping；内部验证 MAE 选择 selected.pt，最后训练继续到 500 以记录完整曲线。原代码默认 1000 epoch，本次 500 是统一研究预算，不声称原设置完全复现。
8. 我们的 derivative/forecasting/both 三组仅替换监测网络，F 不参与；统一冻结对应训练划分的 ODE，lambda_derivative=0.1，lambda_forecasting=1。为定义确定的物理状态函数，ODE 辅助损失计算时禁用监测网络 dropout，但不禁用梯度；监督损失仍启用 dropout。
9. ODE 预测训练每个 minibatch 取相邻 pair 起点对应的 h=1/2/3 未来标签，禁止使用未来传感器；最后不足 batch 的损失仍按实际样本计算。逐 epoch 的训练损失按 pair 数加权汇总；监督与 PDE 是均值，方向项保留作者 batch 内求和语义。
10. ODE 预训练另跑 16 次以记录独立来源，100 epoch、全批次、lr=0.001，使用本训练划分的实测 VB 初值；不与估计初值的监测模型成绩混合。

本轮 12 种配置 × (14 内部验证 + 10 最终训练) + 16 预训练 = **304 次训练**，监测 144,000 epoch 加预训练 1,600 epoch，共 **145,600 epoch**；最终保存 60,000 个 checkpoint。
因为优化日程、pair 监督和轮数已改变，旧研究的 MLP 分数只能作背景；本轮重跑的 MLP 才是结构对照。

## 验证与产物

`tests/test_zhang.py` 从原 notebook 仅提取审查过的类与 forward，验证迁移模型在 eval/train（包含相同 dropout 随机种子）下的预测、PDE 残差、监测与 F 参数梯度一致；另有解析多项式检验收缩导数含义，以及冻结 ODE 仍能回传监测网络的检验。

每个 run 保存完整配置、scalers、划分、数据质量、环境、代码来源、history.csv、epoch_predictions.csv.gz 和 checkpoint。
外层测试每轮评估，仅作诊断；不用于挑 epoch 或调参。当前测试池已经在前一轮研究中被查看，本轮是同一基准上的后续研究，不是全新的确认性测试。

## 运行

```powershell
conda activate CondWear-ODE
python -m unittest discover -s wear_monitoring_dynamics/tests -v
python -m wear_monitoring_dynamics.scripts.benchmark_zhang
python -m wear_monitoring_dynamics.scripts.run_study --study wear_monitoring_dynamics/configs/studies/zhang_transfer_20260922.yaml --config wear_monitoring_dynamics/configs/zhang_transfer.yaml --execute
python -m wear_monitoring_dynamics.scripts.analyze_completed --study-id zhang_transfer_20260922
```

正式结果目录：`outputs/zhang_transfer_20260922/`。短运行目录 `outputs/zhang_smoke_20260922/` 不进入正式排名。
