> Archived appendix-writing evidence note. Filenames mentioned below refer to the separate appendix source-data package; not all are distributed in this code-only repository. See ../REPRODUCING.md for executable commands.

# 写附录前必须统一的表述

段落编号来自 `manuscript_extracted.txt`，不是 Word 页码。该文本索引只抽取普通段落文本，不代替原 DOCX 的数学公式排版。

## 1. Figure 1 附近的初始化说明（段落 84）：实质不一致

正文／建议描述“第一个可靠 monitor 状态；无可靠状态时用训练刀具平均初始 VB”。实际 NASA fusion 从**第一条保留记录**初始化：sensor 有效时用 sensor monitor，sensor 无效时用同 split／seed 冻结的 CT（工况＋时间）网络。没有“训练平均初始 VB”步骤，也没有先等待第一条有效 sensor 的步骤。

建议替换为：

> The recursive estimate was initialized at the first retained record using the sensor-based monitor when the sensor record was valid and a frozen condition–time estimator otherwise. At subsequent invalid sensor records, the observation correction was skipped and the ODE-propagated state was retained. No measured test wear state was used for initialization or resetting in the deployment protocol.

不要为建议中的平均初值额外算一个数字放进附录，那会变成未执行的方法。实测 VB 初始化只属于单独 oracle 诊断。

## 2. 无效观测“不计算 monitor”（段落 145）：应改为“不融合”

实现会批量计算 monitor，并在无效记录以冻结 CT 输出替换；递推阶段 gain=0，所以该值不进入后验校正。精确写法是“the observation was not assimilated”，不是断言其数值从未被计算。

## 3. 质量检查输入与预测器输入不同

NASA 主网络只用电流候选特征，但完整性规则检查六个原始通道。附录必须披露；不能写“仅检查两个电流通道”。Case 2/run 1 在特征表建立前被数值隔离；Case 12/run 1 保留时间位置但被标为无效 sensor，且没有 VB 标签。最终 138 个外层标签没有无效 sensor。

## 4. 三种训练问题不能混为一谈

- Warm A/B/C/D：monitor100＋独立 ODE500；B/C/D 再适配100轮，单次积分损失。部署融合不参与该次训练展开。
- Cold Current/Local/Integral/Both：共同随机起点100轮，检验局部约束与积分监督；不能把预训练 ODE 的 A 当作“未见未来监督”。
- Stiehl：同样随机起点100轮，但任务、特征、时间尺度、损失的刀具等权处理独立定义。

## 5. 统计与单位

NASA 原 fusion 当前128点对应 D=0.08981；benchmark未来共同交集是100/86/73，原 fusion 是114/100/86。不同池得到不同 MAE 是评价对象改变，不能任意替换。Stiehl 原正式报告与后来逐刀分析分别使用10000与20000 bootstrap重复；正文需对应所引用文件注明，不能统一声称所有区间来自相同一次抽样。

NASA 250 Hz 是传感器采样率，不是加工累计时间单位证据；速率应写 mm per dataset machining-time unit，除非另有可核验的单位来源。Stiehl使用秒。

## 6. 文献对照边界

Sun 2024 PI-GPR 是物理均值回归，不是可学习磨损率局部残差网络；PIDGGCN 物理先验拟合训练 VB，不能说没有标签信息；Kim 已有未来阈值时间预测，本任务的未来 VB 是另行明确的三次曲线求值扩展。不要把所有外部方法概括为“无法预测未来”。

## 7. 尚不能由现有材料宣称的内容

没有测试平均初始 VB fallback；没有证明恢复唯一瞬时真实磨损率；Stiehl 全部信号通过质量检查，不能用该实验声称无效观测处理已经在第二数据集验证；NASA 并非预先完全隔离测试的确认性研究。
