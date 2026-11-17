"""Export existing report data and exact plot points; no model evaluation."""
from pathlib import Path
import gzip,shutil,json,hashlib,zipfile
import pandas as pd
P=Path(__file__).resolve().parent
O=P/'report_source_data';O.mkdir(exist_ok=True)
for folder in ['raw','tables','figure_points','reproduction']:(O/folder).mkdir(exist_ok=True)
raw=['oracle_pair_predictions.csv.gz','E2_error_decomposition_pairs.csv.gz','E5A_deployment_pairs.csv.gz','rate_curves.csv.gz','observed_interval_rates.csv']
for name in raw:
    dest=O/'raw'/name.removesuffix('.gz')
    if name.endswith('.gz'):
        with gzip.open(P/name,'rb') as src, dest.open('wb') as target:shutil.copyfileobj(src,target)
    else:shutil.copy2(P/name,dest)
for src in P.glob('*.csv'):
    if src.name not in raw:shutil.copy2(src,O/'tables'/src.name)
for name in ['analyze.py','report.py','verify.py','audit.json','verification.json','RESULTS.md']:
    shutil.copy2(P/name,O/'reproduction'/name)
def write(df,name):df.to_csv(O/'figure_points'/name,index=False,encoding='utf-8-sig')
o=pd.read_csv(O/'raw/oracle_pair_predictions.csv')
q=o[o.original_fusion & o.model.isin(['warm_A','warm_D'])]
write(q.groupby(['model','fold','case','origin_run','target_run','horizon'])[['true_increment','pred_increment']].mean().reset_index(),'E1_panel_a_scatter_seed_mean.csv')
s=pd.read_csv(P/'E1_oracle_summary.csv')
write(s[(s.scope=='original_fusion')&(s.region=='overall')],'E1_panel_b_horizon_lines.csv')
shutil.copy2(P/'E1_time_summary.csv',O/'figure_points/E1_panel_c_time_lines.csv')
e=pd.read_csv(P/'E2_summary.csv')
write(e[(e.scope=='original_fusion')&(e.region=='overall')],'E2_decomposition_bars.csv')
ep=pd.read_csv(O/'raw/E2_error_decomposition_pairs.csv')
write(ep[ep.method.isin(['fusion_A','fusion_D'])].groupby(['method','horizon','fold','case','origin_run','target_run'])[['initial_error','increment_error']].mean().reset_index(),'E2_association_scatter_seed_mean.csv')
r=pd.read_csv(O/'raw/rate_curves.csv')
write(r.groupby(['model','fold','condition','vb']).rate.agg(['mean','std']).reset_index(),'E8_rate_lines_and_seed_sd.csv')
v=pd.read_csv(O/'raw/observed_interval_rates.csv');v=v[v.horizon==1].copy();v['vb_midpoint']=(v.target_mm+v.origin_target_mm)/2
write(v,'E8_observed_interval_average_points.csv')
t=pd.read_csv(P/'E5A_summary.csv')
write(t[(t.region=='overall')&(t.grouping=='dt_bin')],'E5A_time_lines.csv')
index='''# 报告图表原始数据包

对应2026-11-11的E1、E2、E8、E5A补充报告。不是此前论文全部实验的总数据包。所有压缩CSV已解压为普通CSV，可直接用Excel、Origin、Python或R打开。没有重新训练或改变预测。

## 文件对应

| 报告图/表 | 直接复画/复表文件 | 逐条原始数据 |
|---|---|---|
| E1图(a)实测/预测增量散点 | figure_points/E1_panel_a_scatter_seed_mean.csv | raw/oracle_pair_predictions.csv |
| E1图(b)各horizon增量MAE | figure_points/E1_panel_b_horizon_lines.csv | raw/oracle_pair_predictions.csv |
| E1图(c)时间分层误差 | figure_points/E1_panel_c_time_lines.csv | raw/oracle_pair_predictions.csv |
| E1 warm和cold结果表 | tables/E1_oracle_summary.csv；筛选scope=original_fusion、region=overall | tables/E1_case_seed.csv |
| E1 D−A置信区间表 | tables/E1_case_bootstrap.csv；筛选scope=original_fusion、method=warm_D | tables/E1_case_seed.csv |
| E2误差平方分解图及h3变化表 | figure_points/E2_decomposition_bars.csv；h3表取D各项减A | raw/E2_error_decomposition_pairs.csv |
| E2初值/增量误差散点 | figure_points/E2_association_scatter_seed_mean.csv | raw/E2_error_decomposition_pairs.csv |
| E8 warm/cold速率曲线与阴影 | figure_points/E8_rate_lines_and_seed_sd.csv | raw/rate_curves.csv |
| E8黑叉观测点 | figure_points/E8_observed_interval_average_points.csv | raw/observed_interval_rates.csv |
| E8速率变化描述 | tables/E8_rate_change_descriptive.csv | raw/rate_curves.csv |
| E5A图和时间分层表 | figure_points/E5A_time_lines.csv | raw/E5A_deployment_pairs.csv |

tables/另附逐case/seed、覆盖数量、时间边界、按horizon同时分层等完整结果。

## 字段与口径

- fold、case、seed：外层划分、实验case、网络随机种子；seed不是独立case。
- origin_run、target_run、horizon：起点/终点记录编号及后续有标签记录序号。horizon不是固定时间长度。
- dt：实际加工时间间隔，使用数据集时间单位，未擅自标为秒。
- origin_target_mm、target_mm：真实起点/终点VB；origin_pred_mm：实际部署使用的估计起点。
- oracle_future_mm：以真实VB为初值积分所得末状态；不是实际部署成绩。
- true_increment：真实终点VB减真实起点VB；pred_increment：oracle末状态减真实起点VB。
- error/abs_error/squared_error：增量有符号误差/绝对误差/平方误差。
- initial_error、increment_error、future_error：部署初值/增量/未来有符号误差，future=initial+increment。
- cross_term：2×初值误差×增量误差。MSE可加分解，MAE不能相加。
- rate：mm/数据集时间单位；observed_average_rate为区间平均，不是瞬时rate真值。
- E8 condition元组：depth、feed、material代码；mean/std是五seed均值/样本标准差，阴影不是置信区间。
- scope=original_fusion：原完整融合目标池，每seed h1/h2/h3为114/100/86对。
- scope=external_common：外部方法共同目标池，100/86/73对；all_labeled_pairs为全有标签起点诊断池。不能混合不同scope。
- dt_bin：short≤6，medium为6<dt≤12，long>12；边界来自训练配对。范围外计数为0，不证明时间外推。
- warm_A：原预训练rate（A/B/C相同）；warm_D：联合微调rate；cold_*为独立冷启动实验，预算不同。

性能统计先在case/seed内平均配对误差，再在case内平均seed，最后case等权。散点的seed平均只用于显示，不能直接由这些散点重算报告MAE（绝对值与平均不可交换）。文件保留完整精度，报告只做显示四舍五入。

## 完整性

manifest.json包含文件行数、列名与SHA256。reproduction/保存生成与审计脚本以及报告文本，脚本原路径引用用于追溯，不意味着包含checkpoint。该包包含报告数据，不包含模型权重或NASA原始波形。
'''
(O/'README_数据说明.md').write_text(index,encoding='utf-8-sig')
entries=[]
for p in sorted(O.rglob('*')):
    if not p.is_file() or p.name=='manifest.json':continue
    row={'file':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    if p.suffix=='.csv':
        d=pd.read_csv(p);row.update(rows=len(d),columns=list(d.columns))
    entries.append(row)
(O/'manifest.json').write_text(json.dumps({'source_report':str(P/'RESULTS.md'),'files':entries},ensure_ascii=False,indent=2),encoding='utf-8')
z=P/'E1_E2_E8_E5A_report_source_data.zip'
with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in sorted(O.rglob('*')):
        if p.is_file():archive.write(p,Path(O.name)/p.relative_to(O))
with zipfile.ZipFile(z) as archive:
    assert archive.testzip() is None
    for entry in entries:
        data=archive.read(str(Path(O.name)/entry['file']).replace('\\','/'))
        assert hashlib.sha256(data).hexdigest()==entry['sha256']
print(json.dumps({'zip':str(z),'bytes':z.stat().st_size,'files':len(entries)+1,'csv_files':sum('rows' in x for x in entries)},ensure_ascii=False))
