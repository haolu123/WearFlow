"""Read-only diagnosis of completed sensor/CT/gate studies; no test selection."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path, write_json
from ..data.records import load_table
from ..data.feature_archive import attach_features
from ..data.sensor_quality import apply_quality

ROOT = project_path('wear_monitoring_dynamics/outputs/sensor_mechanisms_20261012')
OLD = ROOT.with_name('current_gate_20261012')


def md(df):
    return '\n'.join(['| '+' | '.join(map(str, df.columns))+' |',
                      '|'+'|'.join(['---']*len(df.columns))+'|'] +
                     ['| '+' | '.join(f'{v:.6f}' if isinstance(v, float) else str(v) for v in row)+' |'
                      for row in df.itertuples(index=False, name=None)])


def main():
    out = ROOT / 'step1_existing'
    out.mkdir(parents=True, exist_ok=True)
    manifests = []
    for file, method in [('reused_ct_manifest.csv', 'ct'), ('reused_sensor_manifest.csv', 'sensor')]:
        manifests.append(pd.read_csv(OLD/file).assign(method=method))
    g = pd.read_csv(OLD/'study_manifest.csv')
    manifests.append(g.assign(method=g.name))
    manifest = pd.concat(manifests, ignore_index=True)
    curves = []
    for r in manifest.itertuples():
        h = pd.read_csv(Path(r.run_dir)/'history.csv')
        assert h.epoch.tolist() == list(range(1, 501))
        curves.append(h.assign(method=r.method, fold=r.fold, role=r.role, seed=r.seed, holdout=r.holdout))
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out/'all_learning_curves.csv.gz', index=False)
    validation = curves[curves.role=='inner'].groupby(['method', 'fold', 'epoch']).agg(
        val_mae=('validation_monitoring_case_macro_mae', 'mean'),
        train_mae=('train_monitoring_case_macro_mae', 'mean')).reset_index()
    best = validation.sort_values(['val_mae', 'epoch']).groupby(['method', 'fold']).first().reset_index()
    last = validation[validation.epoch==500].rename(columns={'val_mae':'val_500', 'train_mae':'train_500'})
    best = best.merge(last[['method','fold','val_500','train_500']], on=['method','fold'])
    best['val_deterioration'] = best.val_500-best.val_mae
    best.to_csv(out/'inner_selected_epochs.csv', index=False)
    # These epochs are selected by inner validation only, never by outer curves.
    records=[]
    for r in manifest[(manifest.role=='final') & manifest.method.isin(['ct','sensor'])].itertuples():
        d=Path(r.run_dir)
        e=pd.read_csv(d/'epoch_predictions.csv.gz')
        e=e[e.split=='test_diagnostic']
        assert e.task.eq('monitoring').all()
        selected=int(best[(best.method==r.method)&(best.fold==r.fold)].epoch.iloc[0])
        for label, epoch in [('fixed500',500),('inner_epoch',selected)]:
            q=e[e.epoch==epoch].copy()
            q['method']=r.method;q['fold']=r.fold;q['seed']=r.seed;q['policy']=label
            q['ae']=abs(q.predicted_vb_mm-q.target_vb_mm)
            q['se']=(q.predicted_vb_mm-q.target_vb_mm)**2
            records.append(q)
    records=pd.concat(records,ignore_index=True)
    records.to_csv(out/'paired_test_predictions.csv',index=False)
    cases=records.groupby(['method','policy','fold','seed','case']).agg(mae=('ae','mean'),mse=('se','mean')).reset_index()
    cases.to_csv(out/'case_metrics.csv',index=False)
    summary=cases.groupby(['method','policy']).agg(mae=('mae','mean'),case_macro_mse=('mse','mean')).reset_index()
    summary.to_csv(out/'summary.csv',index=False)
    diff=cases.pivot(index=['policy','fold','seed','case'],columns='method',values=['mae','mse'])
    pd.DataFrame({'delta_mae_sensor_minus_ct':diff['mae']['sensor']-diff['mae']['ct'],
                  'delta_mse_sensor_minus_ct':diff['mse']['sensor']-diff['mse']['ct']}).reset_index().to_csv(out/'paired_case_deltas.csv',index=False)
    clips=[];counts=[]
    for r in manifest[(manifest.method=='sensor') & (manifest.role=='final') & (manifest.seed==0)].itertuples():
        d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config']
        scalers=json.loads((d/'scalers.json').read_text());split=json.loads((d/'split.json').read_text())
        t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))
        t=apply_quality(attach_features(t,cfg['data']['feature_archive']),cfg['data']['sensor_quality'])
        st=scalers['sensor_transform'];idx=st['indices'];raw=t['sensor'][:,idx]
        is_clip=(raw<np.array(st['lower']))|(raw>np.array(st['upper']))
        for case in sorted(set(t['case'])):
            if case not in split['train']+split['outer_test_reserved']:continue
            valid=(t['case']==case)&t['sensor_valid']&np.isfinite(t['VB'])
            if not valid.any():continue
            label='train' if case in split['train'] else 'test'
            counts.append(dict(fold=r.fold,case=int(case),split=label,labeled=int(((t['case']==case)&np.isfinite(t['VB'])).sum()),valid=int(valid.sum())))
            for k,j in enumerate(idx):
                clips.append(dict(fold=r.fold,case=int(case),split=label,feature=cfg['data']['feature_archive']['names'][j],fraction=float(is_clip[valid,k].mean())))
    pd.DataFrame(clips).to_csv(out/'clipping_by_case_feature.csv',index=False)
    pd.DataFrame(counts).to_csv(out/'training_population.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for j,fold in enumerate(['fold_a','fold_b']):
        for method,color in [('ct','tab:blue'),('sensor','tab:orange')]:
            q=validation[(validation.fold==fold)&(validation.method==method)]
            axs[0,j].plot(q.epoch,q.val_mae,label=method,color=color)
            axs[1,j].plot(q.epoch,q.train_mae,label=method,color=color)
        axs[0,j].set(title=fold+' inner validation',ylabel='Case-macro MAE (mm)')
        axs[1,j].set(title=fold+' inner training',ylabel='Case-macro MAE (mm)',xlabel='Epoch')
        for ax in axs[:,j]:ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'learning_curves.png',dpi=160);plt.close(fig)
    body='# 第一步：已有训练结果诊断\n\n只读取原有结果，无新训练。336次运行各500轮。epoch按每fold七个内层留出case的平均MAE选择；报告的内层最优值有选择乐观性，不视为独立性能。最终五seed统一使用所选epoch。\n\n'
    body+='## 无sensor与sensor\n\n'+md(summary)+'\n\n## 内层选轮数及末轮退化\n\n'+md(best[best.method.isin(['ct','sensor'])])
    body+='\n\n## 注意\n\n训练损失是点平均标准化MSE，主指标是case平均MAE，两者权重和损失形状均不同。case_metrics.csv保留case级MSE；不把RMSE简单平均后称为总体MSE。代码中sensor训练跳过无效传感器记录，无sensor保留所有标签；但逐记录核对发现当前14个case的有效标签没有因该规则发生差异，因此不能把它当成本次性能差异的原因。新增机制实验显式统一训练掩码，并共同使用原冻结ct后备处理无效sensor记录。原先138个外层标签仍全部保留。\n\n裁剪统计只包含实际14个研究case，只描述输入，不能单凭裁剪率证明误差原因。已有键配对审计不等同于原始采样窗口与VB测量时刻的物理对齐验证。测试已被查看，本轮均为探索性。\n\n![Learning curves](learning_curves.png)\n'
    (out/'RESULTS.md').write_text(body,encoding='utf-8')
    write_json(out/'audit.json',dict(passed=True,runs=len(manifest),epochs=len(curves),selection='inner_case_macro_mae_only'))
    print(summary.to_string(index=False));print(best[best.method.isin(['ct','sensor'])].to_string(index=False))


if __name__=='__main__':main()
