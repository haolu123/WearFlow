# 随机初始化四臂机制实验

协议见上级 `internal_protocol.md`。本模块只写自身results目录，复用项目的FeatureMLP、WearDynamics及derivative loss；不修改项目源码。

- `mechanism.py`：共同模型、数据加载、损失、梯度诊断及3epoch smoke。
- `audit_core.py`：解析偏导、梯度和物理单位转换检查；16训练划分mask审计。
- `run_internal.py`：96次100epoch；按运行身份安全复用已完成任务。未完成单次从共同随机起点重跑。
- `report_internal.py`：固定末轮汇总、配对bootstrap、逐case/seed、epoch0与目标键检查。

运行命令：`D:/anaconda3/envs/CondWear-ODE/python.exe -B <本目录>/run_internal.py`，报告同理调用report_internal.py。执行身份含执行代码与关键复用模块、数据、元数据、协议、后备checkpoint哈希。完成后勿修改被哈希文件再以执行入口汇报；报告入口不重新训练。

每run的 `epoch_predictions.npz` 存101个快照（0为初始化，1..100为更新后）；current_z是全部记录原始monitor输出（含共同后备），future_z与oracle_z按pairs映射到来源/目标/horizon。用于评估的标签由train/validation/test_mask规定，不能把166行全部当标签。target_mm为物理磨损，z转mm参数存model.pt的scalers中。train_observations与train_pairs_used是实际损失样本。

history.csv含每轮各训练损失以及三划分当前/未来MAE。空划分指标为空；损失是更新前，预测与指标是更新后。gradients.json存首末轮各损失monitor/rate梯度范数及全参数梯度余弦。model.pt保存共同随机初始态和最终态，不含任何预训练monitor/rate。

CT后备复用既有500轮训练模型且四臂相同，不算随机机制分支的100轮预算；其路径/hash单独记录。当前主协议外层138个有标签点均有效，仍保留后备处理以避免未来数据无效时误用。质量规则与特征选择沿用已观察过测试的探索性协议。

