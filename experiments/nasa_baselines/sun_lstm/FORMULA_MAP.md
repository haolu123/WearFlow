# Sun2020 源码—迁移对应

| 来源 | 原始实现 | 本包实现 | 差异 |
|---|---|---|---|
| source/seq2seq_tool_wear/model.py build_simple_RNN | CuDNNLSTM64 return_sequences | model.py encoder1 | PyTorch迁移 |
| 同上 | CuDNNLSTM32 last | encoder2最后时刻 | 同一门方程 |
| 同上 | Dense32 | bridge | 线性无激活 |
| 同上 | RepeatVector(output_length) | repeat horizon | 5改3 |
| 同上 | CuDNNLSTM32、16均返回序列 | decoder1/decoder2 | 同一门方程 |
| 同上 | Dense(output_dim) | head | 输出1 |
| 同上 compile | logcosh、Adam | logcosh_loss、torch Adam | y_z损失尺度；明确学习率/epsilon |
| main_separate_with_monitoring.py | 输入2个历史CNN监测磨损，输出5步 | 公共sensor100预测历史2，输出3步 | 非完整CNN系统 |
| 原训练入口 | 真实wear序列训练 | train_mask内完整2+3窗口 | NASA按case隔离 |
| 原训练入口 | test命名数组做validation | 固定100不测试选参 | 统一协议，不复刻可能不适用的划分 |

原模型可表达对固定步数序列的预测；不输入真实时间。time仅用于对齐和报告，不能把它称为连续时间传播方法。

未来扰动测试对象为预测输入构建，不代表整个原文ResNet因果性已核查。本文没有复现原ResNet，不对其因果性作结论。

