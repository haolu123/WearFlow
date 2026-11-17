# 普通 GPR / SVR：统一输入基线

已完成合成重载验证、一个内层划分烟雾测试以及完整内层选参/外层拟合。不是物理信息 GPR 的文献复现。

## 输入与选择协议

使用主目录 data_manifest.json 的16个划分。每次读取校验 arrays.npz SHA256；输入X共17维，目标y_z；拟合仅train_fit_mask；内层评分仅validation_mask且sensor_valid。数据已按各划分的训练内排名、裁剪与scaler处理，本脚本不重拟合预处理。

- GPR：Constant(1,1e-3..1e3) × RBF或Matérn(ν1.5/2.5)，加White(noise .01,1e-6..1)。初始各向同性length1，范围1e-2..1e3。训练LML拟合，无restart，normalize_y=False，数值jitter alpha=1e-10。
- SVR：RBF，C=.1/1/10/100，epsilon=.01/.1，gamma=scale/.01，共16候选。C/epsilon基于标准化目标。
- 每fold7个内层留一case，按case等权MAE(mm)选择，完全不读取外层评分选参。并列按config_id字典序。
- 确定性算法只运行一次，不伪造5个独立seed；GPR/SVR没有epoch，日志记录每次fit与最终预测。

## 结果（case等权MAE，mm）

| 方法 | overall | fold A | fold B | A <=.74 | A >.74 |
|---|---:|---:|---:|---:|---:|
| GPR | .093732 | .107212 | .080252 | .073541 | .333067 |
| SVR | .104221 | .114765 | .093677 | .073579 | .378585 |

GPR选出：A Matérn1.5，B RBF。
SVR选出：A C10/epsilon.01/gamma.01；B C1/epsilon.01/gamma.01。

执行范围是valid_only，不应用CT后备。本次审计发现所有外层有标签测试点的sensor均有效，因此本次valid_only恰好覆盖全部138点（A74，B64），无排除测试键。**这不表示所有无标签记录sensor都有效，也不表示已实现后备策略。**未做与其他新基线的正式优劣结论。

完整运行266个内层fit＋4个最终fit。脚本内总耗时约1.05秒、模型fit累计约.57秒（不含Python启动及烟雾测试）；小样本CPU单线程。零拟合warning。4个最终pickle在全部有效记录上的重载误差均为0。

## 文件

- results/protocol.json：代码/数据hash、环境、配置和范围。
- results/inner_scores.csv、inner_predictions.csv.gz：逐候选逐内层评分和原始预测。
- results/candidate_summary.csv、selected.json：内层汇总与选出配置。
- results/outer_predictions_valid_only.csv：外层原始目标/预测，2方法共276行。
- results/all_record_predictions_valid_only.csv：各final模型所有有效sensor记录预测，含无标签记录；is_test表示有标签测试池。
- results/case_metrics_valid_only.csv、summary_valid_only.csv：case指标与总表。
- results/models/<fold>/<method>/model.pkl、metadata.json：模型及变换元数据。只加载可信本地pickle。
- results/audit.json、warnings.json、status.json：检查、warnings及耗时。
- smoke/：合成+1split验证，与正式结果分开。

## 运行

从任意目录：

    D:/anaconda3/envs/CondWear-ODE/python.exe <本目录>/run_standard.py --smoke
    D:/anaconda3/envs/CondWear-ODE/python.exe <本目录>/run_standard.py

默认覆盖本脚本results目录中的同名结果，不启动任何神经网络训练，不更改共享数据。

