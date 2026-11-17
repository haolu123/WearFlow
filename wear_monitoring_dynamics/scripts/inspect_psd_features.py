"""Run-level Welch PSD and VB plots; no fitting, feature selection or training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path, write_json
from ..data.features import SENSORS

ROOT=project_path('wear_monitoring_dynamics/outputs/psd_inspection_20261010')
FS=250.0  # Display convention from existing project documentation, not acquisition verification.
BANDS=[(0,5),(5,15),(15,30),(30,60),(60,125)]

def features(x):
    f,p=welch(np.asarray(x,dtype=float).ravel(),fs=FS,window='hann',nperseg=1024,
              noverlap=512,nfft=1024,detrend='constant',scaling='density',average='mean')
    df=f[1]-f[0];power=float(p.sum()*df);q=p/p.sum()
    centroid=float(np.sum(f*q));bandwidth=float(np.sqrt(np.sum((f-centroid)**2*q)))
    positive=p[1:];weights=positive/positive.sum()
    row=dict(ac_power=power,centroid_hz=centroid,bandwidth_hz=bandwidth,
             entropy=float(-np.sum(weights*np.log(weights+1e-300))/np.log(len(weights))),
             flatness=float(np.exp(np.mean(np.log(positive+1e-300)))/positive.mean()),
             peak_hz=float(f[1:][np.argmax(positive)]))
    for lo,hi in BANDS:
        mask=(f>=lo)&((f<hi) if hi<FS/2 else (f<=hi))
        row[f'band_{lo}_{hi}_fraction']=float(q[mask].sum())
        row[f'band_{lo}_{hi}_power']=float(p[mask].sum()*df)
    assert abs(sum(row[f'band_{a}_{b}_fraction'] for a,b in BANDS)-1)<1e-10
    return f,p,row

def style(ax):
    ax.grid(alpha=.18);ax.set_xlabel('Dataset time field (unit unverified)')

def main():
    ROOT.mkdir(exist_ok=True);(ROOT/'plots').mkdir(exist_ok=True)
    records=sorted(np.load(project_path('mill_numpy/mill.npy'),allow_pickle=True).tolist(),key=lambda r:(r['case'],r['run']))
    quality_root=ROOT.with_name('history_quality_skip_20261008')
    decisions=pd.read_csv(quality_root/'quality_decisions.csv')
    verdicts=decisions.groupby(['case','run']).sensor_valid.agg(['all','any']);assert (verdicts['all']==verdicts['any']).all()
    valid={k:bool(v) for k,v in verdicts['all'].items()}
    rows=[];spectra={};frequencies=None
    for r in records:
        key=(int(r['case']),int(r['run']))
        for ch in SENSORS:
            row={k:r[k] for k in ['case','run','time','VB','DOC','feed','material']}
            row.update(channel=ch,sensor_valid=valid[key])
            if valid[key]:
                f,p,values=features(r[ch]);row.update(values);spectra[(*key,ch)]=p;frequencies=f
            rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'psd_features.csv',index=False)
    np.savez_compressed(ROOT/'psd_arrays.npz',frequency_hz=frequencies,
                        **{f'case{c}_run{r}_{ch}':p for (c,r,ch),p in spectra.items()})
    decisions.drop_duplicates(['case','run'])[['case','run','sensor_valid','reason']].to_csv(ROOT/'quality_records.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # Same heatmap scale across cases within each channel; display only, never a training transform.
    limits={ch:np.quantile(np.concatenate([np.log10(p+1e-300) for (c,r,k),p in spectra.items() if k==ch]),[.02,.98]) for ch in SENSORS}
    for case in sorted(frame.case.unique().astype(int)):
        data=frame[frame.case==case];base=data[data.channel==SENSORS[0]].sort_values('time')
        label=base[base.VB.notna()];bad=base[~base.sensor_valid]
        for ch in SENSORS:
            q=data[data.channel==ch].sort_values('time');fig,axs=plt.subplots(4,2,figsize=(14,12),layout='constrained')
            axes=axs.ravel()
            axes[0].plot(label.time,label.VB,'o-',color='black',ms=4);axes[0].set(title='Measured VB (no label interpolation)',ylabel='VB (mm)')
            if case==2:axes[0].text(.02,.93,'Run 1 VB shown for reference; excluded in prior training',transform=axes[0].transAxes,fontsize=8,va='top')
            fields=[('ac_power','AC spectral power','Stored amplitude squared'),('centroid_hz','Spectral centroid','Hz*'),('bandwidth_hz','Spectral bandwidth','Hz*'),('entropy','Normalized spectral entropy','0–1'),('flatness','Spectral flatness','0–1'),('peak_hz','Dominant non-DC frequency','Hz*')]
            for ax,(field,title,unit) in zip(axes[1:7],fields):
                ax.plot(q.time,q[field],'o-',ms=3,color='#277da8');ax.set(title=title,ylabel=unit)
                if field=='ac_power':ax.set_yscale('log')
                if field in ['entropy','flatness']:ax.set_ylim(0,1)
            for lo,hi in BANDS:axes[7].plot(q.time,q[f'band_{lo}_{hi}_fraction'],'o-',ms=2,label=f'{lo}–{hi} Hz*')
            axes[7].set(title='Relative band power',ylabel='Fraction',ylim=(0,1));axes[7].legend(fontsize=8,ncol=2)
            for ax in axes:
                style(ax)
                for t in bad.time:ax.axvline(t,color='#d1495b',ls=':',alpha=.8)
            fig.suptitle(f'Case {case} | {ch} | VB and PSD features\n*Hz assumes fs=250 Hz; Welch 1024/512, segment mean removed. Red dotted line = invalid sensor.',fontsize=13)
            fig.savefig(ROOT/f'plots/case{case:02d}_{ch}.png',dpi=115);plt.close(fig)
        fig,axes=plt.subplots(7,1,figsize=(13,17),layout='constrained')
        axes[0].plot(label.time,label.VB,'ko-',ms=4);axes[0].set(ylabel='VB (mm)',title='Measured VB; lines only guide the eye')
        times=base.time.to_numpy();edges=np.r_[times[0]-.5,(times[:-1]+times[1:])/2,times[-1]+.5]
        for ax,ch in zip(axes[1:],SENSORS):
            z=np.array([np.log10(spectra[(case,int(r),ch)]+1e-300) if valid[(case,int(r))] else np.full(len(frequencies),np.nan) for r in base.run]).T
            mesh=ax.pcolormesh(edges,np.r_[frequencies-(frequencies[1]/2),frequencies[-1]+frequencies[1]/2],z,shading='flat',vmin=limits[ch][0],vmax=limits[ch][1],cmap='viridis')
            ax.set(ylabel='Hz*',title=ch,ylim=(0,125));fig.colorbar(mesh,ax=ax,label='log10 PSD')
        for ax in axes:
            ax.set_xlabel('Dataset time field (unit unverified)')
            for t in bad.time:ax.axvline(t,color='#d1495b',ls=':',lw=1)
        fig.suptitle(f'Case {case} | Run-level PSD evolution (not within-run STFT)\n*fs=250 Hz assumed. Blank column = invalid sensor. Per-channel color scale fixed across cases.',fontsize=13)
        fig.savefig(ROOT/f'plots/case{case:02d}_spectra.png',dpi=115);plt.close(fig)
        print(f'Plotted case {case}',flush=True)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>PSD 与 VB 对照</title><style>body{font:16px system-ui;margin:24px auto;max-width:1500px;background:#f5f7fa;color:#182737}header{position:sticky;top:0;background:#f5f7faf5;padding:12px;z-index:2;border-bottom:1px solid #ccd}select{font-size:17px;margin:8px;padding:7px}img{width:100%;background:white;margin:16px 0}p{line-height:1.6}a{color:#1469a5}</style><header><b>NASA Milling · PSD 特征与真实 VB</b><br>Case <select id="case"></select>通道 <select id="channel"></select>视图 <select id="view"><option value="features">特征曲线与 VB</option><option value="spectra">六通道频谱随 run 演化</option></select><a href="README.md">方法说明</a> · <a href="psd_features.csv">特征数据</a></header><p>仅可视化，未训练或筛选特征。Hz 采用项目记载的 250 Hz，尚未独立核实采集链；不是原始高频 AE 频谱。横轴为原数据 time 字段。红色虚线标记坏 sensor：case2/run1、case12/run1。VB 只画实测点，连线仅辅助观察。Case 6、15 仅展示，未纳入此前模型实验。</p><p>特征图使用各自物理尺度，不做逐 case 标准化。频谱图每列代表一次 run 的整体 Welch PSD，列宽按相邻 time 中点显示，不代表连续采集；色标按通道跨 case 固定，以全体有效数据的 2–98% 分位数控制显示范围，仅用于绘图。</p><img id="plot"><script>const channels=CHANNELS;const cs=document.getElementById('case'),ch=document.getElementById('channel'),v=document.getElementById('view');for(let i=1;i<=16;i++)cs.add(new Option('Case '+i,i));channels.forEach(s=>ch.add(new Option(s,s)));function show(){ch.disabled=v.value==='spectra';document.getElementById('plot').src='plots/case'+cs.value.padStart(2,'0')+'_'+(v.value==='spectra'?'spectra':ch.value)+'.png'}[cs,ch,v].forEach(e=>e.onchange=show);show();</script></html>'''.replace('CHANNELS',json.dumps(SENSORS))
    (ROOT/'index.html').write_text(html,encoding='utf-8')
    notes=['# PSD 特征与 VB 曲线','', '本次仅提取和绘图，不训练模型，不筛选特征，不修改原始数据。入口：index.html。',
           '','## 处理协议','', '原始167条记录、16个case、6通道。复用已完成质量审计的16个划分一致名单：case2/run1和case12/run1不计算PSD，保留时间位置并留空。Case2/run1的实测VB仅在图上展示，不更改此前训练标签池。Case6/15仅供浏览。',
           '采样率250 Hz来自项目现有记载，本次未独立验证原始采集链。所有Hz刻度均为条件性标注；采样率若不同，频率与PSD密度刻度须重算。AE通道不能解释为未经处理的高频声发射。',
           '每run每通道Welch：Hann窗、1024点、512点重叠、nfft1024、每段减均值、平均周期图、单边density。分辨率约0.24414 Hz。未将每条信号归一化为单位方差；去均值后的功率为AC谱功率，不等于包含DC的原始均方。',
           '频带：[0,5)、[5,15)、[15,30)、[30,60)、[60,125] Hz，预先固定且不根据VB挑选。频带功率=所属频点PSD求和×df；相对功率=频带功率/总功率。边界以离散频点归属，不作边界插值。',
           '重心与带宽按全谱归一化权重计算；谱熵和谱平坦度不包含DC频点；主峰为非DC最大PSD频点。未提取转频或齿通过频率，未假定刀齿数量。',
           '无标签run仍计算PSD；VB只展示实测值，不插值生成标签。所有曲线连接仅辅助观察，不证明时间连续性或因果。',
           '频谱热图是跨run的PSD图，不是单次run内的STFT。固定通道色标取所有有效PSD的log10值2–98%分位数，范围外仅颜色饱和，原始数值保存在npz，不用于训练。',
           '','## 文件','', '`plots/`：96张通道特征图和16张六通道频谱图。`psd_features.csv`：全部记录×通道的特征，坏sensor特征为空。`psd_arrays.npz`：有效记录PSD及频率。`quality_records.csv`：质量标记。',
           '','## 解释限制','', '观察VB与特征同步变化可能受到加工时间、工况和case差异混杂，不能据此证明预测能力。这里不按图挑选最佳case或特征，也不计算全数据筛选名单。',
           '复现：`python -m wear_monitoring_dynamics.scripts.inspect_psd_features`。']
    (ROOT/'README.md').write_text('\n'.join(notes),encoding='utf-8')
    good=frame[frame.sensor_valid];assert np.isfinite(good.select_dtypes('number').drop(columns='VB')).all().all()
    assert len(spectra)==165*6 and len(list((ROOT/'plots').glob('*.png')))==112
    write_json(ROOT/'audit.json',dict(records=167,valid_sensor_records=165,feature_rows=len(frame),plots=112,training=False,fs_assumed=FS))

if __name__=='__main__':main()
