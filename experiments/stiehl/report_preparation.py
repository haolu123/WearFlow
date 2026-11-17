"""Preparation report only: does not start training or read test predictions."""
from pathlib import Path
import json
import platform
import pandas as pd
import numpy as np
import scipy
import sklearn
import torch

ROOT=Path(__file__).resolve().parent

def main():
    feature=json.loads((ROOT/'feature_audit.json').read_text())
    data=json.loads((ROOT/'data_audit.json').read_text())
    audit=json.loads((ROOT/'preparation_audit.json').read_text())
    smoke=json.loads((ROOT/'smoke/audit.json').read_text())
    metrics=pd.read_csv(ROOT/'smoke/epoch_metrics.csv')
    quality=pd.read_csv(ROOT/'sensor_quality.csv')
    coverage=pd.read_csv(ROOT/'forecast_coverage.csv').groupby('requested_seconds')['count'].sum()
    assert feature['invalid']==0 and audit['passed']
    assert not audit['outer_test_performance_computed']
    environment={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'training_device':'cpu','torch_threads':1}
    (ROOT/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    timing=metrics.groupby('method').seconds.median()
    lines=['# 第一批：数据协议与开发侧运行验证','',
      '**状态：准备及 smoke 已通过；正式 216 次神经网络训练、18 次 GPR 拟合尚未启动。**',
      '', '## 数据和目标', '',
      f'- 6,418 个信号文件，51 个候选三轴力特征；基础质量检查无失败。特征最终提取耗时 {feature["seconds"]:.1f} 秒。',
      f'- 排除各刀首记录后，当前目标 {data["target_count"]:,} 条。每个目标仅使用上一完成记录的力信号。',
      '- 240/480/720 秒目标使用实际已有记录，传播采用实际 dt；允许匹配滞后最多 30 秒，不插值制造新标签。',
      '', '| 请求跨度（秒） | 配对数（9 刀合计） |','|---|---:|']
    for h,n in coverage.items():lines.append(f'| {h} | {n} |')
    lines += ['',f'时间轴量化修正后，各记录采样率估计范围 {quality.fs.min():.4f}–{quality.fs.max():.4f} Hz，符合标称 25 kHz。初版使用相邻 float32 时间差中位数得到约 24,966 Hz，已在正式训练前修正；旧特征档案仅留作溯源，不用于正式运行。',
      '', '## 已通过检查', '',
      '- 18 个划分：修改所有非训练刀具的标签不会改变该划分排名、前 12 维名单、裁剪阈值及 scaler。',
      '- 未来配对均在同一刀具中严格向前，时间差与原始 C 完全一致，T8 起点保留。',
      '- Time/Current/Local/Integral 四类模型均有限、可反向传播，保存重载预测完全一致。Current-only 的未使用 rate 没有更新，Local/Integral 的 rate 均更新。',
      '- 固定 sensor 的时间偏导与有限差分一致；标准化磨损速率换算为 mm/s 的检查通过。',
      '- ODE 对初值可微、非负速率产生非负增量；RK4 20/40 步误差见 smoke/audit.json。',
      '- 开发侧只用 T1–T7 训练、T8 验证，没有计算 T9 外层测试指标。三轮 smoke 的验证值仅用于运行诊断，不用于比较最终方法优劣。',
      '', '## 预算与实测耗时','',
      f'三轮 smoke（四种网络及一次 GPR 拟合）合计 {smoke["total_seconds"]:.1f} 秒；GPR 使用 896 个训练子集样本，拟合及验证 {smoke["gpr"]["seconds"]:.2f} 秒。',
      '', '| 方法 | 每轮训练＋当前验证中位时间（秒） |','|---|---:|']
    for method,seconds in timing.items():lines.append(f'| {method} | {seconds:.4f} |')
    lines += ['',f'开发侧全量当前＋未来＋发布标签初值传播评估每次约 {np.median(audit["eval_seconds_development_all_current_future_oracle"]):.2f} 秒。因此不能只按反向传播耗时估算正式任务。',
      '', '计划 216 次 × 100 epoch 神经网络运行，加 18 次 GPR；包含逐轮评估、保存和正式审计，当前估计约 **3–4 小时**。这不是已经消耗的训练时间。',
      '', '## 下一阶段与解释边界','',
      '下一阶段落实正式 runner 的逐 epoch 原始预测、断点 identity/优化器/RNG 保存，再按 study_manifest.csv 固定清单运行；不需要根据本次验证值扩充网格。正式冻结末轮 100 epoch，五 seed 先在刀具内平均，按刀具配对统计。',
      '', 'S1 比较当前监测；S2 比较时间查询、persistence、因果线性趋势、Local 与 Integral；S3 复用两种 rate 的发布标签初值积分。这里没有 ODE 预训练、融合或历史网络。',
      '', '发布档案没有可靠原始显微测量位置标记；不从插值曲线推断测量 mask，不把每 40 条当实测点。结果验证公开标签轨迹，不能单独证明真实瞬时机理。固定切削工况、仅 9 刀、机床与批次部分混杂、中央窗口并非精确接触分割，也均须保留。']
    (ROOT/'PREPARATION_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status={'stage':'preparation_and_development_smoke_complete','formal_training_started':False,'formal_nn_runs_planned':216,'formal_gpr_fits_planned':18,'formal_nn_runs_completed':0,'test_performance_computed':False,'next':'implement formal resumable runner then fixed 100-epoch execution','preparation_audit_passed':True}
    (ROOT/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status))

if __name__=='__main__':main()
