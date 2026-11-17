"""All 167 original records; stitch by run, with no filtering or rescaling."""
import html
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path,write_json,file_hash

ROOT=project_path('wear_monitoring_dynamics/outputs/raw_current_stitched_20261012')
FS=250.
BAD={(2,1),(12,1)}


def main(other_sensors=False):
    global ROOT
    if other_sensors:ROOT=project_path('wear_monitoring_dynamics/outputs/raw_other_sensors_stitched_20261012')
    ROOT.mkdir(parents=True,exist_ok=True)
    source=project_path('mill_numpy/mill.npy')
    records=sorted(np.load(source,allow_pickle=True).tolist(),key=lambda r:(int(r['case']),int(r['run'])))
    channels=['vib_table','vib_spindle','AE_table','AE_spindle'] if other_sensors else ['smcAC','smcDC']
    colors=['#2374ab','#d36b20','#279260','#8859ad'];limits={}
    for ch in channels:
        low=min(np.min(r[ch]) for r in records if (int(r['case']),int(r['run'])) not in BAD)
        high=max(np.max(r[ch]) for r in records if (int(r['case']),int(r['run'])) not in BAD)
        limits[ch]=(float(low-.04*(high-low)),float(high+.04*(high-low)))
    allmeta=[];panels=[];total=0
    plt.rcParams.update({'path.simplify':False,'agg.path.chunksize':20000,'font.size':10})
    for case in range(1,17):
        rr=[r for r in records if int(r['case'])==case];durations=[len(np.asarray(r['smcAC']).ravel())/FS for r in rr]
        edges=np.r_[0,np.cumsum(durations)];meta=[]
        for i,r in enumerate(rr):
            n=len(np.asarray(r[channels[0]]).ravel());assert all(n==len(np.asarray(r[ch]).ravel()) for ch in channels)
            total+=n*len(channels)
            meta.append(dict(case=case,run=int(r['run']),samples=n,start_seconds=float(edges[i]),end_seconds=float(edges[i+1]),
                             dataset_time=float(r['time']),flagged=(case,int(r['run'])) in BAD))
        allmeta.extend(meta)
        modes=['full','detail'] if any(m['flagged'] for m in meta) else ['full']
        for mode in modes:
            fig,axs=plt.subplots(len(channels),1,figsize=(max(18,len(rr)*.88),3.5*len(channels)+1),sharex=True)
            fig.subplots_adjust(left=.075,right=.99,bottom=.06,top=.91,hspace=.08)
            for ax,ch,color in zip(axs,channels,colors):
                for i,r in enumerate(rr):
                    x=np.asarray(r[ch],float).ravel();t=edges[i]+np.arange(len(x))/FS
                    ax.plot(t,x,color=color,lw=.35,rasterized=True)
                    if meta[i]['flagged']:ax.axvspan(edges[i],edges[i+1],color='#e84855',alpha=.12)
                    ax.axvline(edges[i],color='#777',lw=.6,ls='--',alpha=.55)
                ax.set(ylabel=ch+' | stored amplitude',xlim=(0,edges[-1]));ax.grid(axis='y',alpha=.2)
                if mode=='detail':ax.set_ylim(limits[ch])
                elif case not in [2,12]:ax.set_ylim(limits[ch])
                ax.ticklabel_format(axis='y',style='sci',scilimits=(-4,5))
            top=axs[0].secondary_xaxis('top');top.set_xticks((edges[:-1]+edges[1:])/2)
            top.set_xticklabels([str(int(r['run']))+('*' if m['flagged'] else '') for r,m in zip(rr,meta)])
            top.set_xlabel('Run number (* previously flagged record; retained here)')
            axs[-1].set_xlabel('Concatenated sample time (seconds, 250 Hz); gaps between experiments are not represented')
            note='FULL amplitude range, all original samples' if mode=='full' else 'DETAIL: normal display range; out-of-range anomalies remain in data but are off-screen'
            label='Vibration and acoustic emission (stored RMS signals)' if other_sensors else 'AC and DC current'
            first=rr[0];fig.suptitle(f'Case {case} | {label} | {len(rr)} runs | DOC={first["DOC"]}, feed={first["feed"]}, material={first["material"]}\n{note}. No additional filtering, smoothing, normalization or joining across run boundaries.',fontsize=13)
            path=ROOT/f'case{case:02d}_{mode}.png';fig.savefig(path,dpi=145);plt.close(fig)
        table=''.join(f'<tr><td>{m["run"]}</td><td>{m["dataset_time"]:g}</td><td>{m["start_seconds"]:g}–{m["end_seconds"]:g}</td><td>{m["samples"]}</td></tr>' for m in meta)
        detail=f'<p><a href="case{case:02d}_detail.png" target="_blank">打开正常幅值范围细节图</a>（异常超范围部分不可见；完整幅值图仍保留全部样本）</p><img loading="lazy" src="case{case:02d}_detail.png">' if len(modes)>1 else ''
        panels.append(f'<section id="case{case}"><h2>Case {case} · {len(rr)} runs</h2><a href="case{case:02d}_full.png" target="_blank">打开完整分辨率图片</a><img loading="lazy" src="case{case:02d}_full.png">{detail}<details><summary>Run与原数据time字段对应表</summary><table><tr><th>Run</th><th>原time字段</th><th>拼接秒数</th><th>点数/通道</th></tr>{table}</table></details></section>')
        print(f'Plotted case {case}: {len(rr)} runs, {edges[-1]:g}s',flush=True)
    nav=' '.join(f'<a href="#case{i}">{i}</a>' for i in range(1,17))
    document='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>各case原始电流按run拼接</title><style>body{font:16px system-ui;margin:0 auto;padding:24px;max-width:1800px;background:#f5f7fa;color:#182737}nav{position:sticky;top:0;background:#fff;padding:14px;z-index:2;border-bottom:1px solid #ccd}nav a{display:inline-block;margin:3px 9px}a{color:#16699a}section{scroll-margin-top:80px;background:white;margin:24px 0;padding:18px;border-radius:8px}img{width:100%;display:block;margin-top:12px}p{line-height:1.6}table{border-collapse:collapse;margin:10px}td,th{padding:6px 20px;border:1px solid #ddd}</style><h1>原始电流 · 每个case按run顺序拼接</h1><p>全部16个case、167条记录、AC与DC两个通道。保留所有原始采样点，不平滑、不滤波、不归一化。每条run独立画线，虚线为边界；横轴累加各记录采样时长（250Hz），不是连续真实加工时间，记录间未采集的间隔没有补齐。原始time字段保存在下方对应表。</p><p>红色背景及星号标出此前质量检查标记的case2/run1和case12/run1；本次原始图完整保留。两case另附正常幅值范围细节图，避免异常值压扁其他run。其余case按通道共用同一幅值范围，便于比较。幅值使用存储单位，不未经校准标成安培。图片可打开后放大。</p><nav>跳转Case：NAV</nav>PANELS</html>'''.replace('NAV',nav).replace('PANELS',''.join(panels))
    if other_sensors:
        document=document.replace('原始电流','振动与声发射存储信号').replace('AC与DC两个通道','四个通道：vib_table（台面振动）、vib_spindle（主轴振动）、AE_table（台面声发射）、AE_spindle（主轴声发射）')
        document=document.replace('保留所有原始采样点，不平滑、不滤波、不归一化。','原数据采集链已经对这四个通道进行滤波和RMS处理；这里保留全部存储采样点，不额外平滑、滤波或归一化，不能理解为未处理的高频声发射或振动波形。')
        document=document.replace('不未经校准标成安培','不未经校准标成加速度或物理声发射单位')
        document=document.replace('<nav>','<p><a href="../raw_current_stitched_20261012/index.html">切换到AC/DC电流图</a></p><nav>')
    (ROOT/'index.html').write_text(document,encoding='utf-8')
    if other_sensors:
        current=project_path('wear_monitoring_dynamics/outputs/raw_current_stitched_20261012/index.html')
        if current.exists():
            text=current.read_text(encoding='utf-8')
            if '../raw_other_sensors_stitched_20261012/index.html' not in text:
                text=text.replace('<nav>','<p><a href="../raw_other_sensors_stitched_20261012/index.html">查看振动与声发射：全部16个case</a></p><nav>')
                current.write_text(text,encoding='utf-8')
    write_json(ROOT/'manifest.json',dict(source_sha256=file_hash(source),sampling_hz=FS,cases=16,records=len(records),channels=channels,
        plotted_sample_values=total,normal_display_limits=limits,original_samples_preserved=True,raw_time_field_not_used_as_sample_timestamp=True,records_metadata=allmeta))
    assert len(allmeta)==167 and len({(m['case'],m['run']) for m in allmeta})==167
    assert len(list(ROOT.glob('case*_full.png')))==16 and len(list(ROOT.glob('case*_detail.png')))==2


if __name__=='__main__':main()
