"""Package the completed Stiehl study and plot source data, without retraining."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parent
DEST=ROOT.parent/'Stiehl_experiments_results_source_data_20261112'
DEST.mkdir(exist_ok=True)
files=[]
root_names=['README.md','FORMAL_RUN.md','PREPARATION_RESULTS.md','RESULTS.md','protocol.json','protocol_identity.json','status.json','environment.json','study_manifest.csv','gpr_manifest.csv','split_manifest.csv','records.csv','causal_targets.csv','forecast_pairs.csv','forecast_coverage.csv','sensor_quality.csv','features.npz','feature_audit.json','data_audit.json','preparation_audit.json','resume_audit.json']
for name in root_names:files.append(ROOT/name)
files.extend(ROOT.glob('*.py'))
for folder in ['reports','tool_analysis','splits']:
    files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
# Enough run metadata to regenerate analyze_tools.py figures and budget tables.
files.extend((ROOT/'formal').glob('*/*/complete.json'))
files.extend((ROOT/'formal').glob('*/*/identity.json'))
files.extend(p for p in [ROOT/'smoke/audit.json',ROOT/'smoke/epoch_metrics.csv'] if p.exists())
for src in sorted(set(files)):
    dst=DEST/'experiment'/src.relative_to(ROOT)
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dst)

readme='''# Stiehl 实验、图表与原始结果数据交付包

打包日期：2026-11-12。范围是本轮 Stiehl 独立数据集验证；不包含此前 NASA 的全部实验。

## 先看这三个文件

1. [主要发现、逐刀解释与正式主表](experiment/tool_analysis/FINDINGS.md)
2. [完整逐刀与误差分解报告](experiment/tool_analysis/ANALYSIS.md)
3. [基础实验结果](experiment/RESULTS.md)

## 做了什么实验

数据为 Stiehl Milling V3：9 把刀具、3 台机床、6418 个信号文件。三轴力提取51维候选特征，每个训练划分内按相关性选12维，训练内裁剪并标准化。使用上一段已完成的信号预测下一段开始状态，排除各刀首记录，共6409个当前目标。

- S1 当前监测：时间 MLP、Sensor MLP、普通 GPR、Local、Integral。
- S2 未来预测：时间 MLP 在目标时间查询；同一 Sensor MLP 的 persistence 和因果线性外推；Local 与 Integral 的 ODE 传播。
- S3 积分增量：Local 与 Integral 从发布标签初值传播，比较区间累积磨损。其增量误差与标签初值终点误差数学相同。
- 事后诊断：九刀逐刀比较、工具级配对 bootstrap、去掉任一刀的汇总敏感性、初值／传播误差分解、预先指定 T1/T5/T8 的轨迹。

9 折留一刀；每折一个固定内层诊断划分；最终5个seed。216次神经网络训练均100 epoch，另18次确定性GPR拟合。无ODE预训练、无NASA权重迁移。固定末轮，不按测试选超参数或epoch。未来请求240/480/720秒，实际匹配可延后最多30秒，共6007/5626/5256对。五seed先在刀具内平均，刀具作为统计单位。

### 主结果速览（MAE，mm）

| 方法 | 当前 | 240秒 | 480秒 | 720秒 |
|---|---:|---:|---:|---:|
| 时间 MLP | 0.010724 | 0.010603 | 0.010932 | 0.011460 |
| Sensor MLP＋persistence | 0.006549 | 0.008179 | 0.012373 | 0.018299 |
| 同一Sensor MLP＋线性外推 | 同上 | 0.037262 | 0.067564 | 0.093726 |
| GPR | 0.007123 | — | — | — |
| Local＋ODE | 0.006355 | 0.006913 | 0.008942 | 0.012502 |
| Integral＋ODE | 0.006688 | 0.005985 | 0.006025 | 0.006753 |

未来每个跨度均7/9刀改善，退步的为T5/T7；当前仅3/9改善。积分监督主要改善长期传播，不能称为全面提高当前监测。T5存在较强误差抵消反例。标签包含插值、无可靠实测点mask；只验证公开标签轨迹，不声称恢复真实瞬时磨损机理。

## 图表在哪里、对应什么原始数据

| 图表 | 文件（相对于 experiment/） | 对应数据 |
|---|---|---|
| 正式图1：九刀当前／未来／标签初值诊断 | tool_analysis/figure1_tool_comparison.pdf、.svg、.png | tool_analysis/paired_tool_mae.csv；reports/tool_metrics_seed_mean.csv |
| 正式图2：T1/T5/T8轨迹与增量 | tool_analysis/figure2_predeclared_trajectories.pdf、.svg、.png | tool_analysis/figure2_curve_data.csv；tool_analysis/figure_published_labels.csv；pair_error_decomposition.csv.gz |
| 当前／未来主表 | tool_analysis/FINDINGS.md | tool_analysis/table_main.csv；reports/summary.csv |
| 初值与传播误差来源表 | tool_analysis/ANALYSIS.md、FINDINGS.md | tool_analysis/table_focal_error_sources.csv；decomposition_per_tool.csv；decomposition_per_tool_seed.csv；decomposition_macro.csv |
| 收益覆盖、区间及留一刀敏感性 | tool_analysis/ANALYSIS.md | improvement_breadth.csv；paired_tool_mae.csv |
| 覆盖与预算 | tool_analysis/FINDINGS.md | table_coverage.csv；table_budget.csv；study_manifest.csv；gpr_manifest.csv |
| 100轮训练曲线 | reports/training_curves.png | reports/test_metrics_all_epochs.csv |
| 初版探索图（额外保留） | reports/example_trajectories.png；mechanism_comparison.png | reports/test_predictions_epoch100.csv.gz |

上述表中未重复目录前缀的分解CSV均位于 tool_analysis/。正式两图用PDF/SVG排版，PNG便于预览。图中µm，数据表mm。轨迹是seed平均预测，仅作可视化；MAE是每seed计算后平均，不是ensemble平均预测的MAE。

## “原始数据”包含到哪一层

- reports/test_predictions_epoch100.csv.gz：未聚合的末轮测试预测，保留method/seed/tool/任务/起点终点键和标签，可重算结果；包含725704行，current记录以filename对齐，future记录以origin/target对齐。
- reports/test_metrics_all_epochs.csv：epoch 0–100测试集逐刀逐seed指标，重绘训练曲线。
- tool_analysis/pair_error_decomposition.csv.gz：Local/Integral逐预测对、逐seed的初值、预测、标签初值预测和误差分解。
- records.csv、causal_targets.csv、forecast_pairs.csv：记录、严格因果当前目标及真实时间配对；origin/target是causal_targets.csv的零基行号，不是原始run编号。
- features.npz：本次正式使用的6418×51力特征、特征名和filename键。没有包含已废弃的采样率量化初版。
- splits/*.json：18个训练划分的训练刀具、训练内排名、所选特征、裁剪与scaler。
- formal/*/*/identity.json、complete.json：运行身份、参数数、完成状态、耗时和GPR训练子集索引，用于预算和审计追溯。

**本包不包含约9.75GB原始HDF5归档、不包含模型checkpoint、也不包含全部训练／验证／测试逐epoch原始预测NPZ。**这些不属于本次图表重绘所必需的数据；训练原档仍保留在原项目目录。完整训练结果审计计216×101=21816个原始epoch预测档案。本包提供所有上述图表所需的末轮原始预测及逐轮测试指标。

公开数据来源：https://data.mendeley.com/datasets/zpxs87bjt8/3 （DOI 10.17632/zpxs87bjt8.3，CC BY 4.0）。累计接触C按论文解释为秒，原H5中的min属性不用于乘60；VB微米转mm。

## 代码、审计与使用

- 源码保持在 experiment/，避免改变相对路径。analyze_tools.py 可从本包的原始预测、标签和运行元数据重新生成逐刀分析及正式两图；需Python、numpy、pandas、matplotlib。FINDINGS.md是人工核对后的解读，不由脚本覆盖。
- report_formal.py 的完整训练档案审计仍需要未打包的checkpoint及逐epoch预测；不要把它作为本包的独立重绘入口。
- reports/audit.json、preparation_audit.json、resume_audit.json、tool_analysis/audit.json保留核查结果。
- experiment/README.md、PREPARATION_RESULTS.md为当时冻结协议／准备阶段记录，其中“尚未启动”的句子描述历史状态；当前完成状态以status.json及本索引为准。
- MANIFEST.csv列出每个交付文件的相对路径、字节数和SHA256。package_validation.json记录CRC和源数据行数核验。

未新增训练、未修改冻结的原实验结果。
'''
(DEST/'README_先读我.md').write_text(readme,encoding='utf-8')
rows=[]
for path in sorted(DEST.rglob('*')):
    if path.is_file() and path.name not in ['MANIFEST.csv','package_validation.json']:
        rows.append({'path':str(path.relative_to(DEST)).replace('\\','/'),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
pd.DataFrame(rows).to_csv(DEST/'MANIFEST.csv',index=False,encoding='utf-8-sig')
prediction=pd.read_csv(DEST/'experiment/reports/test_predictions_epoch100.csv.gz')
decomp=pd.read_csv(DEST/'experiment/tool_analysis/pair_error_decomposition.csv.gz')
assert len(prediction)==725704
assert len(decomp)==16889*2*5
assert all((DEST/row['path']).exists() for row in rows)
validation={'files_in_manifest':len(rows),'prediction_rows':len(prediction),'decomposition_rows':len(decomp),'formal_neural_runs':216,'gpr_fits':18,'source_files_copied_without_modification':True,'archive_crc':'verified after final archive write'}
(DEST/'package_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
archive=DEST.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(DEST.rglob('*')):
        if path.is_file():z.write(path,path.relative_to(DEST))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,**validation},indent=2))
