# Zhang PINN：当前 VB 的 top12 NASA 迁移

## 正式批次已完成

`run_suite.py` 完成24次×固定100epoch：14次内层seed0及两fold×5seed外层，不进行调参或测试选epoch。case等权再seed平均 MAE：总体0.10919 mm，fold A 0.12327，fold B 0.09510。训练计时合计57.15秒（不含导入和数据导出）。正式结果、逐轮预测、checkpoint及审计保存在 `formal/`；最终标准表为 `formal/current_predictions.csv`，每seed两fold合计138个原始目标。

正式审计逐run核查100轮、无重复目标键、每轮train/validation/test预测完整及末轮重载。当前所有有标签sensor有效；无效有标签后备尚未验证，代码会明确报错。初始调度误认为inner编号连续，完成5run后发现缺inner_6；仅修目录枚举，旧新源码hash及identity迁移见 `formal/dispatch_correction_audit.json`，5run未重复训练，其checkpoint保留原identity与summary original_identity对应。没有改动训练算法。

以下记录先前smoke阶段与方法映射；正式批次取代“未启动正式实验”的阶段状态。

来源：作者仓库 https://github.com/zhangpengfei247/Tool-wear-monitoring-boring-PINN ，项目已存提交 `054ec32111b1efc63eabb6016485e1543095b736`，notebook `2. Dataloader construction and PINN model .ipynb`。项目核查说明见 `<repository>/wear_monitoring_dynamics/docs/zhang_reproduction.md`。

本目录不改动共享模型，也没有启动 100 epoch 正式实验。`model.py` 只新增输入宽度 wrapper，复用已核查 ZhangMLP、ZhangPredictor、ZhangPhysics。`test_parity.py` 从原 notebook AST 仅提取网络类与 PINN.forward，直接进行参数键严格映射；推理及训练 dropout 两模式下输出、残差、所有参数梯度一致（double 输出容差 1e-12、梯度 1e-10）。不执行原 notebook 数据读取或训练单元。

## 原实现到适配映射

| 部分 | 适配 |
|---|---|
| Solution_u.encoder | 输入改为 17：12 电流特征、4 工况编码、末列时间；3 层、32 隐层及 32 输出保留 |
| Solution_u.predictor | dropout(.2)、32→30、sin、30→1 保留 |
| Solution_u 初始化 | Linear Xavier normal，bias zero 保留 |
| dynamical_F | 输入 `3*17+2=53`，4 层、50 隐层、sin/dropout 保留 |
| 导数 | 保留作者 vector-Jacobian 收缩：uxx 是 Hessian 乘全 1，uxt 为特征时间混合导数的和；不替换为 Hessian 对角 |
| 监督 | 相邻两端 MSE 均值 |
| 残差 | 相邻两端 `(ut-F)^2` 均值，alpha=1 |
| 方向项 | `relu(-(u2-u1)*(y2-y1)).sum()`，beta=.75，batch 求和，不能称无条件单调约束 |
| 优化 | Adam，无新增 weight decay，batch16；monitor .01→.003 的10轮 warmup，再 cosine 至 .0001；F lr=.001，按作者 epoch 后更新调度 |

## 数据与训练边界

使用共享 `data/` 16 split 导出：每划分独立从 50 个电流候选训练内排名选 12，clip_zscore、标准化 VB、时间缩放和工况编码完全沿用 current12_fusion_20261013。训练 pair 仅由训练 case 中有效 sensor 且有标签的记录按实际时间排序，连接相邻可用记录且 dt>0；不会跨 case 配对。质量过滤使其可能跨过无效记录，这不是插值。验证标签不参与配对或 scaler。

这是作者结构/损失的 NASA 迁移，而非原深孔镗削实验逐数值复现：输入及标签单位/缩放、质量过滤、筛选与数据划分改变；计划固定 100 epoch，原作者训练预算不同。相同 alpha/beta 不代表相同物理量纲权重。不含 ARIMA、不包含 Neural ODE、不承担有未来 sensor 的预测。F 是辅助残差网络，不是数值积分传播器。

完整实验应由父 adapter 接入同 split/seed 冻结 CT 后备。此 smoke 遇有标签无效 sensor 会显式报错，不会默默排除目标点或生成错误后备；当前 smoke 的 train51、validation13 个标签均有效。

## 已验证

运行 `D:/anaconda3/envs/CondWear-ODE/python.exe test_parity.py`：通过。运行 `.../python.exe smoke.py`：fold_a/inner_1、seed0、CPU单线程3 epoch，通过；保存逐轮训练/验证原始预测和损失，45 个相邻 pair，3709 monitor 参数、7851 physics 参数。checkpoint 重新载入输出逐元素完全相同。

训练与保存耗时约 0.083 秒（不含进程启动、导入和数据导出），线性估计同划分100轮约2.8秒；仅为微型 smoke 估计，正式跨split运行不能作为已完成耗时。smoke 调度 horizon 始终100，而非3。结果在 `smoke/history.csv`、`smoke/predictions.csv.gz`、`smoke/train_pairs.csv`、`smoke/summary.json`、`smoke/last.pt`。三轮 MAE 仅用于代码验证，不是方法优劣结论。
