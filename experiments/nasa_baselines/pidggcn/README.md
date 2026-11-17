# PIDGGCN：全文支持的 NASA 因果适配



用户已提供全文，原文核心内容不再阻塞。详细公式/页码、图形状、原文矛盾、NASA适配及未规定细节见 FULLTEXT_DESIGN.md。旧 FORMULA_MAP.md 与 sources.json 记录的是全文获得前的检索阶段，不作为当前实现说明。



当前正式授权：paper_printed、calculus_corrected 两版本各24次（14 inner +10 final）×100epoch，总48次。两版本固定全部报告，inner只诊断。正式结果 formal/，状态formal/status.json，当前进度formal/active_run.json，PID formal/process_id.txt，stdout execution.log，stderr stderr.log，异常execution_error.log。



运行/安全恢复：



```powershell

& 'D:/anaconda3/envs/CondWear-ODE/python.exe' 'experiments/nasa_baselines/pidggcn/run_pidggcn.py' --execute

```



只汇总已完成48次，不训练（不带flags）：



```powershell

& 'D:/anaconda3/envs/CondWear-ODE/python.exe' 'experiments/nasa_baselines/pidggcn/report_pidggcn.py'

```



每epoch原子checkpoint保存model/optimizer/RNG/权重/history/所有记录预测。identity包含model及runner执行源码、输入/metadata/CT父checkpoint哈希和配置；恢复复用，禁止为改报告改执行源码触发重训。完成后导出epoch_predictions.npz，history.csv。报告输出current_predictions.csv标准列及逐case/seed、overall/foldA/B/Ain/out、物理prior拟合/回退、两导数真实单位差异。



正式训练CPU单线程float64。profile显示float32仅快约7%，参数二阶梯度逐元素容差未过，采用float64；逐步独立scalar time的AD与原vector Jacobian对角在预测/导数/参数梯度完全一致，保留全历史参数梯度不detach。profile在derivative_profile.json。完整runner两版本×3epoch smoke共6.79秒，预计48×100约90分钟，预留2小时含评价保存。未按测试调参。



模型是当前VB监测，未来VB不支持，不给它添加任意未来预测头冒充原方法。PDF全文与关键公式渲染位于fulltext/；实际原PDF仍在项目根，status.json记录SHA256。

