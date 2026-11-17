# 内部机制消融冻结协议：局部导数约束与积分传播监督

冻结日期：2026-11-03。状态：设计已冻结，尚未实现或训练。本协议替代以 monitor100 / ODE500 为共同起点来检验 gap 的方案；后者的 ODE 已接受未来标签监督，不能作为“没有未来演化监督”的 local-only 对照。

## 研究问题与可检验假设
在相同网络结构、随机初始化和训练预算下，使用未来真实 VB 的积分传播损失，相比只使用当前点局部时间偏导一致性约束，是否改善当前磨损估计与未来状态预测？同时检验两项约束是否互补。

这里的证据针对本协议的局部偏导约束，不代表所有 PINN。预测改善也不能单独证明学得唯一真实物理磨损率。

## 四臂及共同起点
每个 split/seed 先生成一份随机 monitor 与随机 rate 初始参数，保存 initialization checkpoint 和 hash，四臂加载完全相同的参数。不得加载任何曾接受未来标签监督的 ODE，也不得使用预训练 monitor。rate 初始化应非负、有限且能通过数值积分；初始化规则只能在正式实验前用训练数据或合成数值检查确定，不能看测试误差修改。

|臂|固定损失|解释|
|---|---|---|
|current-only|L_current|当前监督、无动力学训练|
|local-only|L_current + L_local|只有局部偏导一致性作为动力学监督|
|integral-only|L_current + L_future|只有真实未来VB积分传播监督|
|both|L_current + L_local + L_future|两项动力学约束同时加入|

“only”仅指辅助监督来源，四臂都使用当前 VB 监督。四臂共同允许 monitor 和 rate 参数更新，current-only 的 rate 因不连接损失而应保持初始值不变；记录这一点。其随机 rate 未来结果仅是无动力学学习诊断，不应当作有意义的未来预测竞争基线。另列 persistence（未来值取当前估计）作为 current-only 的实际未来基准。

## 固定结构、预处理与预算
- monitor：工况4维编码＋归一化时间＋训练内相关排名top12电流特征，MLP[64,32]、SiLU，输出归一化VB；rate沿用共同WearDynamics结构、非负输出、相同RK4步数。
- 特征候选、质量规则、训练内top12与clip_zscore、wear/condition/time scaler严格沿用共享最新current12协议。每个inner划分仅6个训练case拟合，final仅7个训练case拟合。四臂同split使用完全相同特征与scaler；不使用验证/测试标签排名。
- 从随机初始化训练100epoch，full-batch每epoch一次更新，AdamW lr=1e-3、weight_decay=1e-4（核对current12_fusion的config_resolved.yaml及trainer.py非paired分支，与原随机sensor训练一致；该分支无学习率scheduler，warmup_epochs字段不生效）；不延用任何预训练优化器状态。固定末轮评估，不以测试或验证挑epoch。
- current权重1，local与future出现时权重均预设1。本批不扫描lambda。相同lambda不是相同梯度强度，也不是相同物理单位权重；保存首轮和末轮各项梯度范数、方向及损失。
- 若未来拟调lambda，须在开始实验前冻结候选网格，仅用各fold的7个inner验证case等权指标选择，再进行final训练；不得用外层结果补网格。该变更将产生新协议版本，不能事后替换本批预设权重结果。

## 损失、时间与因果性
令 z=(VB-mu_VB)/sigma_VB，tau=t/T，T沿用该训练划分的h3正时间间隔中位数。

L_current：训练集中有标签且sensor_valid的记录上，monitor归一化VB误差平方的点均值。

L_local：与L_current完全相同的训练obs上，`[partial M(s,c,tau)/partial tau - r(M(s,c,tau),c)]^2` 的均值。求导时固定sensor与condition；create_graph=True，monitor状态传入rate时不detach。

这不是沿加工轨迹的全导。真实轨迹还包含sensor变化项 `(partial M/partial s)(ds/dt)`，因此局部约束本身具有建模假设。归一化速率乘sigma_VB/T才是物理速率；报告各split的转换系数。

L_future：使用训练case的预固定h1/h2/h3配对，起点须有标签且sensor_valid，终点须有真实VB标签（不要求终点sensor有效，因为不使用它），真实时间差必须正且区间工况符合共同常工况假设。四臂保存相同pair列表。积分初值是monitor估计，末状态对真实未来VB监督，配对等权MSE。h指后续有标签记录数，不是等长时间；RK4使用真实dt/T。无未来sensor、无真实VB重置。

质量不合格sensor不进入训练排名、scaler或当前/局部损失，也不作为积分训练起点。测试保留原138标签；无效sensor使用同split/seed的共同冻结CT后备。这一后备若采用既有模型，必须明确其独立训练预算且四臂完全一致；单独报告有效sensor子集，以免后备遮盖机制差异。此训练pair定义与旧ABCD若不同，不得直接复用旧成绩。

## 运行数量及复用约束
每臂2fold × (7inner seed0 + 5final seed)=24运行；四臂合计96次100epoch。14inner与10final均执行同协议，即使本批lambda固定，inner也保留作开发诊断与后续可审计记录。不要把五seed视为五个独立case。

只允许复用完全相同初始化hash、split、data keys、特征/质量/scaler、训练mask与pair keys、网络和积分配置、损失与reduction、optimizer与随机数设定、100轮预算的已完成运行。现有monitor100、ODE500及warmstart ABCD不满足随机共同起点，不能作为本协议任何一臂的已完成训练。可复用训练内预处理档案和同协议CT后备，须验hash。

## 结果与审计
主报告当前raw VB与未来h1/h2/h3，给overall、foldA/B、A真实VB<=.74与>.74、逐case/seed。当前所有臂目标键一致；未来各臂目标键与persistence一致。评估起点可包含共同CT后备，但必须单列其覆盖。当前138标签与另行融合协议的128目标不能混比。本批不加入递推融合、EKF、历史窗口或截断。

保存每epoch train/validation/test原始当前及未来预测、各损失、参数变化、epoch0初始化、固定epoch100 checkpoint、耗时、运行identity。epoch0四臂预测必须一致。独立审计局部解析导数与单位转换、无效sensor屏蔽、真实dt、无未来sensor依赖、父参数无未来监督、current-only rate不变、各辅助臂预期梯度、重载一致。

各case先平均seed误差，再做case配对bootstrap。补充oracle真实初值传播诊断，用于观察rate的预测作用；oracle不是实际部署结果。稀疏VB标签、有限工况、monitor与rate相互补偿以及数值积分误差意味着速率函数可能不可辨识；不能把低未来MAE等同于恢复真实瞬时磨损率。局部/积分/两者结果均如实报告，不预设积分获胜。

## warmstart ABCD的单独定位
monitor100＋ODE500出发再训练100轮的已有ABCD，适合回答“已有动力学模型下，冻结ODE适配monitor与联合适配是否改善部署效果”。ODE500已见未来状态监督，因此该实验不能检验“局部约束缺少未来监督”的gap。它应放在适配消融附表，与本随机起点机制实验分开陈述，不混作相同总预算。




单位注：物理速率为mm/原始time单位，需从数据来源确认时间单位后才能标注每秒或每分钟。


