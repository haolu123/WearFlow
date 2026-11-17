"""Add measured-VB scatter plots to the PSD inspection gallery; no training."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from .inspect_psd_features import ROOT, SENSORS, BANDS


def main():
    frame=pd.read_csv(ROOT/'psd_features.csv')
    paired=frame[frame.sensor_valid & frame.VB.notna()].copy()
    paired.to_csv(ROOT/'psd_vb_paired_points.csv',index=False)
    fields=[('ac_power','AC spectral power','Stored amplitude squared'),
            ('centroid_hz','Spectral centroid','Hz*'),('bandwidth_hz','Spectral bandwidth','Hz*'),
            ('entropy','Normalized spectral entropy','0–1'),('flatness','Spectral flatness','0–1'),
            ('peak_hz','Dominant non-DC frequency','Hz*')]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for case in sorted(frame.case.unique().astype(int)):
        for channel in SENSORS:
            q=paired[(paired.case==case)&(paired.channel==channel)].sort_values('time')
            assert len(q)>0
            fig,axs=plt.subplots(4,2,figsize=(14,12),layout='constrained');axes=axs.ravel()
            norm=Normalize(vmin=q.time.min(),vmax=max(q.time.max(),q.time.min()+1))
            for ax,(field,title,unit) in zip(axes[:6],fields):
                sc=ax.scatter(q.VB,q[field],c=q.time,cmap='viridis',norm=norm,s=40,edgecolors='white',linewidths=.4)
                ax.set(title=title,ylabel=unit)
                if field=='ac_power':ax.set_yscale('log')
                ax.margins(x=.08,y=.15)
            for lo,hi in BANDS:
                axes[6].scatter(q.VB,q[f'band_{lo}_{hi}_fraction'],s=25,label=f'{lo}–{hi} Hz*')
                axes[7].scatter(q.VB,q[f'band_{lo}_{hi}_power'],s=25,label=f'{lo}–{hi} Hz*')
            axes[6].set(title='Relative band power (color = band)',ylabel='Fraction',ylim=(-.02,1.05))
            axes[7].set(title='Absolute band power (color = band)',ylabel='Stored amplitude squared',yscale='log')
            for ax in axes[6:]:ax.legend(fontsize=8,ncol=2)
            for ax in axes:ax.set_xlabel('Measured VB (mm)');ax.grid(alpha=.18)
            fig.colorbar(sc,ax=list(axes[:6]),label='Dataset time field (color on first 6 panels)',shrink=.65,pad=.02)
            fig.suptitle(f'Case {case} | {channel} | PSD features versus measured VB | n={len(q)}\nOnly valid sensor + measured VB; points are not connected or fitted. *Hz assumes fs=250 Hz.',fontsize=13)
            fig.savefig(ROOT/f'plots/case{case:02d}_{channel}_vs_vb.png',dpi=115);plt.close(fig)
        print(f'Plotted case {case}',flush=True)
    page=ROOT/'index.html';html=page.read_text(encoding='utf-8')
    if 'value="vb"' not in html:
        html=html.replace('<option value="features">','<option value="vb">横轴 VB · 纵轴 PSD 特征</option><option value="features">')
    old="(v.value==='spectra'?'spectra':ch.value)+'.png'"
    new="(v.value==='spectra'?'spectra':ch.value+(v.value==='vb'?'_vs_vb':''))+'.png'"
    html=html.replace(old,new)
    html=html.replace('横轴为原数据 time 字段。','时间视图横轴为原数据 time 字段；VB 视图横轴为实测 VB，前六个面板的点颜色表示加工时间，频带面板颜色表示频带。VB 视图仅保留 sensor 有效且 VB 有标签的记录，不连线、不插值、不拟合。')
    page.write_text(html,encoding='utf-8')
    readme=ROOT/'README.md';body=readme.read_text(encoding='utf-8').split('\n## VB 横轴视图')[0]
    body+='\n## VB 横轴视图\n\n新增96张 `*_vs_vb.png`，图册默认显示此视图。横轴为实测VB，纵轴为各PSD特征；同一case、同一通道内作图。仅使用sensor有效且有VB标签的记录，不填补VB、不连接散点、不拟合曲线。前六面板颜色表示原始time，后两面板颜色表示频带。纵轴按各特征自动取范围，绝对功率采用对数轴。Case6/15仍仅供浏览。\n\n配对点见 `psd_vb_paired_points.csv`；复现：`python -m wear_monitoring_dynamics.scripts.plot_psd_vs_vb`。全体case图供探索，不构成特征选择或泛化性能证据。\n'
    readme.write_text(body,encoding='utf-8')
    count=len(list((ROOT/'plots').glob('*_vs_vb.png')));assert count==96
    assert np.isfinite(paired[['VB']+[x[0] for x in fields]]).all().all()
    (ROOT/'vb_plot_audit.json').write_text(json.dumps(dict(plots=count,paired_rows=len(paired),records=len(paired[['case','run']].drop_duplicates()),training=False),indent=2),encoding='utf-8')


if __name__=='__main__':main()
