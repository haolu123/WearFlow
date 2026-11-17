"""Read-only feature distribution audit using actual fitted outer-fold scalers."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..data.sensor_quality import apply_quality
from ..data.feature_archive import attach_features
from ..data.records import load_table
from ..data.preprocessing import fit_scalers
from ..config import project_path,digest
from .run_psd_sweep import ROOT

def main():
    dest=ROOT/'feature_range_diagnostics';dest.mkdir(exist_ok=True)
    manifest=pd.read_csv(ROOT/'study_manifest.csv');selection=pd.read_csv(ROOT/'selected_dimensions.csv')
    rawrows=[];summary=[];detail=[];points=[]
    for fold in ['fold_a','fold_b']:
        for group in ['B','C','D']:
            name=selection[(selection.fold==fold)&(selection.group==group)].name.iloc[0]
            r=manifest[(manifest.fold==fold)&(manifest.name==name)&(manifest.role=='final')&(manifest.seed==0)].iloc[0]
            d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config'];scaler=json.loads((d/'scalers.json').read_text());split=json.loads((d/'split.json').read_text())
            table,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))
            table=apply_quality(attach_features(table,cfg['data']['feature_archive']),cfg['data']['sensor_quality'])
            assert digest(fit_scalers(table,split['train']))==digest(scaler)
            names=cfg['data']['feature_archive']['names'];x=table['sensor'];valid=table['sensor_valid'];train=valid&np.isin(table['case'],split['train'])
            mu,sd=map(np.asarray,scaler['sensor']);z=(x-mu)/sd
            np.testing.assert_allclose(z[train].mean(0),0,atol=1e-6)
            lo=x[train].min(0);hi=x[train].max(0);selected=cfg['monitor']['selected_indices']
            if fold=='fold_a' and group=='B':
                for case in np.unique(table['case']):
                    ix=valid&(table['case']==case)
                    for j,f in enumerate(names):
                        v=x[ix,j];rawrows.append(dict(case=int(case),feature=f,n=len(v),minimum=v.min(),q05=np.quantile(v,.05),median=np.median(v),q95=np.quantile(v,.95),maximum=v.max(),std=v.std()))
            for case in split['test']:
                ix=valid&(table['case']==case);a=z[ix][:,selected];values=x[ix][:,selected];outside=(values<lo[selected])|(values>hi[selected])
                summary.append(dict(fold=fold,case=case,group=group,k=len(selected),n=int(ix.sum()),max_abs_z=abs(a).max(),median_abs_z=np.median(abs(a)),pct_abs_z_gt3=100*np.mean(abs(a)>3),pct_outside_train_range=100*outside.mean()))
                for jj,j in enumerate(selected):
                    detail.append(dict(fold=fold,case=case,group=group,feature=names[j],train_min=lo[j],train_max=hi[j],case_min=values[:,jj].min(),case_max=values[:,jj].max(),train_mean=mu[j],train_std=sd[j],z_min=a[:,jj].min(),z_max=a[:,jj].max(),case_mean_z=a[:,jj].mean(),max_abs_z=abs(a[:,jj]).max()))
                for i in np.flatnonzero(ix):
                    j=max(selected,key=lambda j:abs(z[i,j]));points.append(dict(fold=fold,case=case,run=int(table['run'][i]),time=table['time'][i],VB=table['VB'][i],group=group,feature=names[j],z=z[i,j],raw=x[i,j]))
    raw=pd.DataFrame(rawrows);s=pd.DataFrame(summary);det=pd.DataFrame(detail);pts=pd.DataFrame(points)
    raw.to_csv(dest/'all_case_raw_feature_ranges.csv',index=False);s.to_csv(dest/'selected_feature_case_summary.csv',index=False);det.to_csv(dest/'selected_feature_ranges.csv',index=False);pts.to_csv(dest/'per_record_largest_z.csv',index=False)
    grid=s.pivot(index=['fold','case'],columns='group',values='max_abs_z')
    fig,ax=plt.subplots(figsize=(7,8),layout='constrained');im=ax.imshow(np.log10(1+grid.to_numpy()),aspect='auto',cmap='YlOrRd')
    for i,row in enumerate(grid.to_numpy()):
        for j,value in enumerate(row):ax.text(j,i,f'{value:.1f}',ha='center',va='center',color='black')
    ax.set_xticks(range(3),['B: time-domain','C: PSD','D: combined']);ax.set_yticks(range(len(grid)),[f'{f}: case {c}' for f,c in grid.index]);ax.set_title('Maximum |z| in selected features per test case\nActual training-only mean/std; all valid sensor records')
    fig.colorbar(im,ax=ax,label='log10(1 + max |z|)');fig.savefig(dest/'case_feature_ranges.png',dpi=160);plt.close(fig)
    def md(df):return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])
    lines=['# Sensor特征范围与预处理核查','',
      '已核对最终模型实际保存的scalers.json：sensor每一维用训练case有效sensor记录拟合均值和总体标准差（ddof=0，std下限1e-8），验证/测试复用。不是每case单独标准化。无标签但有效sensor也参与sensor scaler拟合。内层只拟合六训练case，外层七case。',
      '切深/进给训练z-score，材料one-hot，时间除以训练h3间隔中位数，VB按训练有标签均值/标准差缩放。PSD已做Welch分段去均值，但未逐run单位方差归一化。没有log变换、RobustScaler或分位数裁剪。',
      '','## 各测试case选中特征的最大绝对z','',md(grid.reset_index()),
      '','最大|z|只是某条记录某一维的最极端值，不能代表全部输入偏移，更不能当作损坏判据。全部有效sensor（含无VB标签）用于本诊断；训练基准范围为该fold的全部有效训练sensor。pct_outside_train_range是记录×特征单元比例，不是异常记录比例。',
      '','## 最极端的选中特征','',md(det.sort_values('max_abs_z',ascending=False).head(15)),
      '','## 文件','', 'all_case_raw_feature_ranges.csv包含16case×150特征的min/q05/median/q95/max/std；selected_feature_ranges.csv包含模型选中特征相对于对应训练分布的范围；selected_feature_case_summary.csv包含>|3|和超出训练min/max的比例；per_record_largest_z.csv定位具体run。',
      '','## 建议','',
      '先针对非负且重尾的功率/能量做固定形式log变换（尺度或epsilon仅用训练拟合）与现有z-score的对照，再比较训练median/IQR缩放；后者对训练离群值更稳健，但不保证减小测试极端z。裁剪可作单独对照，不能默认去掉真实磨损信号。',
      '不使用完整测试case的均值/方差/min/max做逐case归一化，会借用未来记录且抹去磨损相关幅度。若研究基线相对变化，只能用首次有效sensor或已观测历史，并作为独立协议。',
      '本诊断不训练、不改特征或模型；范围差异可能来自工况、磨损覆盖、传感器响应或数值异常，须结合原始信号判断。']
    (dest/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(grid.round(2).to_string());print(s.round(2).to_string(index=False));print(det.sort_values('max_abs_z',ascending=False).head(8).to_string(index=False));print(pts[(pts.case.isin([5,12]))].sort_values('z',key=abs,ascending=False).head(8).to_string(index=False))

if __name__=='__main__':main()
