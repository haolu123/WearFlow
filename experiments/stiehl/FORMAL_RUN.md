# 正式运行与恢复

冻结协议继续采用 README.md 和 protocol.json；不根据测试指标调参。

- 总入口：`D:/anaconda3/envs/CondWear-ODE/python.exe execute_pipeline.py`。依次训练 216 个神经网络、拟合 18 个 GPR，然后调用 report_formal.py；不启动其他阶段。
- 启动前核实 pipeline_process_id.txt、formal/process_id.txt 对应进程，不能同时启动两个入口。
- 完整 epoch 写入顺序：原始预测 NPZ、分刀具指标 JSON、原子更新 last.pt。恢复以 last.pt 为准；未提交的 epoch 可覆盖重算，不覆盖已确认训练身份。
- last.pt 保存模型、初始模型、优化器、Torch RNG、epoch、累计耗时。输入、划分、模型和执行代码 SHA256 组成 identity；不一致时拒绝复用，不自动重训。
- epoch_000 至 epoch_100 保留 train/validation/test 全部当前预测和未来配对预测。固定目标顺序为 causal_targets.csv 与 forecast_pairs.csv；每个 split 的角色通过 splits/*.json 判定。
- current 的 future_mm 是 persistence，并另存 linear_mm；time 是目标时间查询；local/integral 另存发布标签初值 oracle_mm。未定义的 oracle 用 NaN 显式表示，不能当作有效预测。
- `report_formal.py` 仅审计和汇总，不训练。会核对 216×101 个原始预测档案、共享初始化、运行 identity，以及完成的 GPR。输出 RESULTS.md、reports/summary.csv、逐刀/seed 数据、全部 epoch 测试指标、原始末轮预测和配对 bootstrap。
- formal_stdout.log、formal_stderr.log、formal/active_run.json、formal/status.json 为运行状态。异常查看 pipeline_error.log 或 formal/execution_error.log；优先修报告，不修改训练实现导致身份变化。

check_resume.py 已用于比较恢复的第二轮状态与从相同初值连续运行两轮的状态；结果在 resume_audit.json。测试通过后不再重复调用（正式队列会继续该运行至 100 轮）。

时间估计 3–4 小时包括逐轮评估和保存；实际以完成标记中的 seconds 累计为准。审计未完成前，不把 status 写为全部完成。
