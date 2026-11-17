> Archived appendix-writing evidence note. Filenames mentioned below refer to the separate appendix source-data package; not all are distributed in this code-only repository. See ../REPRODUCING.md for executable commands.

# Appendix C — 评价池、指标与统计

## C.1 各评价池必须独立标识

|协议|当前VB数／seed|未来h1/240s|h2/480s|h3/720s|含义|
|---|---:|---:|---:|---:|---|
|NASA统一外部benchmark|128|100|86|73|当前排除递推初始化；未来取共同有限预测交集|
|NASA原始fusion ABCD|128|114|100|86|相同融合起点协议，未缩减至所有外部方法交集|
|NASA cold-start机制消融|138|124|110|96|直接monitor，含所有有标签起点，不作fusion首点排除|
|NASA E1 oracle（原fusion池）|不适用|114|100|86|实测标签作为初值的传播诊断，不是部署|
|Stiehl|6409|6007|5626|5256|严格previous-run对齐，实际秒间隔|

以上是每个seed的目标数，不乘seed当样本量。NASA benchmark未来case数分别14／13／12。oracle另有不同scope／common池，需以E1的scope字段为准，不把上表数字推广至其全部行。

NASA h表示未来第h个**有标签记录**，不是原run号差，也不是恒定时间；每对dt=target_time−origin_time>0。尾部没有第h个后续标签就无该对。部署fusion初始化记录不进入当前误差，也不作为评分的未来起点；原138标签中10个对应有标签初始化，剩128。当前有些case的首记录无标签，所以不能简单138−14。

`NASA_frozen_main_tables/target_keys.csv` 是逐目标审计依据；coverage.csv保留各method提供、finite、common数量，不能把未覆盖目标算成零误差。外部当前池与未来池独立。original_fusion_predictions.csv 与 original_fusion_reproduction.csv 用于保留原ABCD成绩；统一未来交集另看future_predictions.csv，差别不是重训结果。

Stiehl查询指定240/480/720秒，取满足请求后的目标且容差不超过30秒，实际dt进入积分。current6409=6418−9个无先前run对齐起点；未来尾部／容差继续减少覆盖。逐对时间、工具、来源见Stiehl_targets，不能把h1/h2/h3记录步数概念用于此表。

## C.2 MAE汇总

先在每个 case/tool、seed、horizon、region 内对目标误差求均值；再在每个case/tool内平均seed误差；最后case/tool等权。不要先平均五个seed预测再计算误差，那是ensemble评价，含义不同。fold、A-in/out等区域子集同理；in/out按真实目标VB≤.74/>.74诊断分组，不参与模型选择。

NASA总体为纳入case的宏平均，某horizon缺少case时不强制两个fold各占1/2。不同文件的字段粒度明确：case_seed_metrics是个体case/seed结果；summary才是总体。

## C.3 配对bootstrap

比较同一目标池中方法与reference的case级差值。五seed先在case内平均，然后NASA按fold分层抽取case（有放回，每层保持原case数），合并以case数权重求差值。Stiehl对九个tool成对重采样。取重复统计量2.5%／97.5%分位，不把seed或数千目标点当独立case。

|结果来源|重复数|随机数种子／实现|
|---|---:|---|
|NASA主表配对区间|10000|20261103；NASA_build_tables.py，每个contrast初始化generator|
|NASA E1补充诊断|10000|20261111；NASA_supplement_protocol/analyze.py，循环共享generator|
|Stiehl原formal报告|10000|20261111；B中report_formal.py|
|Stiehl后续逐刀分析|20000|20261111；B中analyze_tools.py|

只报告实际文件对应版本。不同脚本的相同seed不代表随机抽样顺序相同。有限case数、已探索测试、多个诊断比较均限制推断；不写成多个独立确认性检验。

## C.4 误差分解

定义 e0=pred_origin−true_origin；einc=(pred_future−pred_origin)−(true_future−true_origin)；ef=pred_future−true_future，则逐对严格 ef=e0+einc。

对应 MSE_future=MSE_initial+MSE_increment+2E[e0·einc]，各项必须使用同一目标和同一case/seed权重。cross_term文件字段已包含系数2，不要再乘2。MAE不能作加法分解。负cross term／异号错误提示抵消，不证明物理速率更准确；结合oracle实测初值的增量误差判断。E2_pairs、Stiehl_pair_error_decomposition包含复算字段。

## C.5 真实时间分箱

E5A阈值仅来自训练配对：两fold训练dt最小2、最大24，下三分位6，上三分位12；short≤6，medium 6<dt≤12，long>12。outside_train为dt<2或dt>24。训练time_scale分别17和18。单位沿NASA原加工时间字段，不能写秒。

新增的 `D_additional/NASA_actual_dt_bin_coverage_horizon.csv` 用Fusion A、seed0去重目标键后计算case／pair数量及dt范围；它对应原fusion池。原始逐对表另保留in_common，若主图采用外部共同池必须据此再次过滤，不能直接拿原fusion覆盖数配共同池曲线。
