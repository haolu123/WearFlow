"""Build audited current processing archive and all-case before/after gallery."""
import json
import numpy as np
import pandas as pd
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..config import project_path,write_json,file_hash
from ..data.current_processing import process_current,spans,DC_RAIL,FS,QUARANTINE

ROOT=project_path('wear_monitoring_dynamics/outputs/current_processing_20261012')


def main():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'arrays').mkdir(exist_ok=True)
    path=project_path('mill_numpy/mill.npy');source_hash=file_hash(path)
    records=sorted(np.load(path,allow_pickle=True).tolist(),key=lambda r:(int(r['case']),int(r['run'])))
    write_json(ROOT/'protocol.json',dict(source=str(path),source_sha256=source_hash,sampling_hz=FS,
        quarantine={f'{c}/{r}':v for (c,r),v in QUARANTINE.items()},dc_rail_stored_code=DC_RAIL,dc_guard_seconds=.2,
        filter_edge_guard_seconds=.5,filter='4th-order Butterworth SOS forward-backward on individual finite spans; effective magnitude is squared',
        dc_trend_cutoffs_hz=[2,5,10],dc_default_display_hz=5,ac='Waveform untouched for accepted records; 51-point centered RMS (204ms) and auxiliary 2Hz lowpass envelope',
        no_interpolation=True,no_cross_run_filtering=True,no_labels_used=True,scope='Exploratory offline preprocessing, no new model or validation performance claim',
        unavailable='NaN and masks preserve original time and length; rail pixels remain available in flag, not treated as exact current',
        online='Uses future samples within each run; intended after run completion, not causal per-sample deployment',
        sources=['https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html','https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html']))
    results={};summary=[];intervals=[];sensitivity=[];total=0
    for r in records:
        case,run=int(r['case']),int(r['run']);ac=np.asarray(r['smcAC'],float).ravel();dc=np.asarray(r['smcDC'],float).ravel()
        out,reason=process_current(ac,dc,case,run);results[case,run]=(r,out,reason);total+=len(ac)
        assert all(len(v)==len(ac) for v in out.values())
        if reason:assert not out['record_valid'].any() and np.isnan(out['dc_observed']).all()
        else:
            np.testing.assert_array_equal(out['ac_waveform'],ac)
            np.testing.assert_array_equal(out['dc_observed'][~out['dc_rail_mask']],dc[~out['dc_rail_mask']])
            assert np.isnan(out['dc_observed'][out['dc_rail_mask']]).all()
        for kind,mask in [('quarantined',~out['record_valid']),('rail',out['dc_rail_mask']),('rail_guard',out['dc_guard_mask'])]:
            for a,b in spans(mask):intervals.append(dict(case=case,run=run,kind=kind,start_sample=int(a),end_sample_exclusive=int(b),start_s=a/FS,end_s=b/FS))
        good=np.isfinite(out['dc_trend_5hz']);middle=np.zeros(len(ac),bool);middle[len(ac)//4:3*len(ac)//4]=True
        row=dict(case=case,run=run,time=float(r['time']),VB=float(r['VB']),n=len(ac),record_valid=not bool(reason),reason=reason,
            rail_samples=int(out['dc_rail_mask'].sum()),rail_fraction=float(out['dc_rail_mask'].mean()),
            dc_observed_available_fraction=float(np.isfinite(out['dc_observed']).mean()),dc_trend_available_fraction=float(good.mean()),
            ac_envelope_available_fraction=float(np.isfinite(out['ac_envelope_2hz']).mean()),
            dc_center50_available_fraction=float(good[middle].mean()),
            dc_residual_rms=float(np.sqrt(np.mean(out['dc_residual_5hz'][good]**2))) if good.any() else np.nan)
        summary.append(row)
        for cutoff in [2,5,10]:
            t=out[f'dc_trend_{cutoff}hz'];m=middle&np.isfinite(t)
            sensitivity.append(dict(case=case,run=run,cutoff_hz=cutoff,n=int(m.sum()),
                mean_change=float(np.mean(t[m]-dc[m])) if m.any() else np.nan,
                rms_change=float(np.sqrt(np.mean((t[m]-dc[m])**2))) if m.any() else np.nan))
    frame=pd.DataFrame(summary);frame.to_csv(ROOT/'quality_summary.csv',index=False)
    pd.DataFrame(intervals).to_csv(ROOT/'masked_intervals.csv',index=False);pd.DataFrame(sensitivity).to_csv(ROOT/'filter_sensitivity.csv',index=False)
    panels=[];plt.rcParams.update({'path.simplify':False,'agg.path.chunksize':20000,'font.size':9})
    for case in range(1,17):
        rr=[v for (c,_),v in results.items() if c==case];edges=np.r_[0,np.cumsum([len(o['ac_waveform'])/FS for _,o,_ in rr])]
        archive={};fig,axs=plt.subplots(4,1,figsize=(max(18,.88*len(rr)),12),sharex=True)
        fig.subplots_adjust(left=.075,right=.99,bottom=.06,top=.9,hspace=.08)
        for i,(r,o,reason) in enumerate(rr):
            run=int(r['run']);x=edges[i]+np.arange(len(o['ac_waveform']))/FS
            for name,value in o.items():archive[f'run{run}_{name}']=value
            if reason:
                for ax in axs:ax.axvspan(edges[i],edges[i+1],color='grey',alpha=.18)
            else:
                axs[0].plot(x,o['ac_waveform'],lw=.3,color='#537e99')
                axs[1].plot(x,o['ac_rms_204ms'],lw=.4,color='#aab3b9');axs[1].plot(x,o['ac_envelope_2hz'],lw=.8,color='#1765a0')
                dc=np.asarray(r['smcDC']).ravel();axs[2].plot(x,dc,color='#b4b8bb',lw=.4)
                axs[2].plot(x,o['dc_observed'],color='#d47c26',lw=.4)
                rail=o['dc_rail_mask'];axs[2].plot(x,np.where(rail,dc,np.nan),color='#c63131',lw=1.5)
                for cutoff,col in [(2,'#158f88'),(10,'#ad65b6'),(5,'#d67c22')]:axs[3].plot(x,o[f'dc_trend_{cutoff}hz'],lw=.65,color=col)
            for ax in axs:ax.axvline(edges[i],color='#888',lw=.5,ls='--',alpha=.5)
        names=['AC waveform (unchanged)','AC RMS 204ms + 2Hz envelope','DC observation | red = rail','DC trends: 2 / 5 / 10 Hz']
        for ax,name in zip(axs,names):ax.set(ylabel=name,xlim=(0,edges[-1]));ax.grid(axis='y',alpha=.2)
        axs[0].set_ylim(-7.6,7.6);axs[1].set_ylim(0,5.2);axs[2].set_ylim(-.4,10.4);axs[3].set_ylim(-.4,10.4)
        top=axs[0].secondary_xaxis('top');top.set_xticks((edges[:-1]+edges[1:])/2);top.set_xticklabels([str(int(r['run']))+('*' if reason else '') for r,_,reason in rr]);top.set_xlabel('Run (* excluded current record; time position retained)')
        axs[-1].set_xlabel('Concatenated sample seconds (250 Hz); no data or interpolation between runs')
        fig.suptitle(f'Case {case} | Current processing preview | stored amplitude units\nGrey blocks: excluded records. Red DC: censored upper code. Trend gaps: saturation guard / filter edge / short segment.\nAC waveform retained; RMS/envelope and DC lowpass are auxiliary views, not proof that residuals are noise.',fontsize=12)
        fig.savefig(ROOT/f'case{case:02d}_processed.png',dpi=140);plt.close(fig)
        np.savez_compressed(ROOT/'arrays'/f'case{case:02d}.npz',**archive)
        # Reload masks and actual values, including intended NaNs.
        with np.load(ROOT/'arrays'/f'case{case:02d}.npz') as restored:
            for name,value in archive.items():np.testing.assert_array_equal(restored[name],value)
        q=frame[frame.case==case]
        table=q[['run','record_valid','rail_fraction','dc_trend_available_fraction','ac_envelope_available_fraction']].to_html(index=False,float_format=lambda v:f'{v:.3f}')
        panels.append(f'<section id="case{case}"><h2>Case {case}</h2><a href="case{case:02d}_processed.png" target="_blank">打开高清图</a> · <a href="../raw_current_stitched_20261012/case{case:02d}_full.png" target="_blank">原始图</a><img loading="lazy" src="case{case:02d}_processed.png"><details><summary>每run可用比例（0–1）</summary>{table}</details></section>')
        print(f'Processed and verified case {case}',flush=True)
    # Representative local windows; raw spectra are not interpreted as pure noise.
    fig,axs=plt.subplots(4,3,figsize=(16,13),layout='constrained')
    for row,key in enumerate([(1,2),(3,2),(12,12),(16,2)]):
        r,o,_=results[key];t=np.arange(len(o['ac_waveform']))/FS;dc=np.asarray(r['smcDC']).ravel()
        for ch,col in [('smcAC','#2574a3'),('smcDC','#c47720')]:
            x=np.asarray(r[ch]).ravel()[4500:6500];f,p=welch(x,fs=FS,nperseg=1024,noverlap=512)
            axs[row,0].semilogy(f,p+1e-15,color=col,label=ch)
        axs[row,0].set(title=f'Case {key[0]} run {key[1]}: 18-26s PSD',xlabel='Hz',ylabel='Stored units squared / Hz');axs[row,0].legend(fontsize=8)
        axs[row,1].plot(t,dc,color='#b7bdc1',lw=.6,label='Recorded DC')
        for cutoff,col in [(2,'#158f88'),(5,'#d67c22'),(10,'#ad65b6')]:axs[row,1].plot(t,o[f'dc_trend_{cutoff}hz'],color=col,label=f'{cutoff}Hz',lw=.8)
        axs[row,1].plot(t,np.where(o['dc_rail_mask'],dc,np.nan),color='red',lw=2,label='Rail, not recovered')
        axs[row,1].set(title='DC close-up',xlim=(18,22),xlabel='Within-run seconds');axs[row,1].legend(fontsize=7)
        axs[row,2].plot(t,o['ac_rms_204ms'],color='#aab3b9',label='204ms RMS',lw=.6);axs[row,2].plot(t,o['ac_envelope_2hz'],color='#1765a0',label='2Hz envelope',lw=1)
        axs[row,2].set(title='AC amplitude view',xlabel='Within-run seconds');axs[row,2].legend(fontsize=8)
        for ax in axs[row]:ax.grid(alpha=.2)
    fig.savefig(ROOT/'local_and_spectral_checks.png',dpi=140);plt.close(fig)
    good=frame[frame.record_valid];sen=pd.DataFrame(sensitivity)
    audit=dict(passed=True,records=len(frame),accepted_records=int(frame.record_valid.sum()),quarantined=frame[~frame.record_valid][['case','run','reason']].to_dict('records'),
        samples_per_channel=total,rail_records=int((good.rail_samples>0).sum()),rail_samples=int(good.rail_samples.sum()),
        accepted_ac_waveform_unchanged=True,unsaturated_dc_observations_unchanged=True,all_archives_reloaded_equal=True,original_sha256_unchanged=file_hash(path)==source_hash,
        dc_trend_available_samples=int(sum(np.isfinite(o['dc_trend_5hz']).sum() for _,o,_ in results.values())),
        dc_trend_available_fraction_median=float(good.dc_trend_available_fraction.median()),
        center50_no_dc_trend_records=int((good.dc_center50_available_fraction==0).sum()),
        filter_sensitivity_median_absolute_center_mean_change={str(hz):float(g.mean_change.abs().median()) for hz,g in sen.groupby('cutoff_hz')},
        synthetic_tests=5,not_causal_within_run=True,no_performance_claim=True)
    assert audit['original_sha256_unchanged'] and len(frame)==167 and len(good)==165
    write_json(ROOT/'audit.json',audit)
    nav=' '.join(f'<a href="#case{i}">{i}</a>' for i in range(1,17))
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>电流处理前后对照</title><style>body{font:16px system-ui;margin:24px auto;max-width:1800px;padding:0 20px;background:#f5f7fa;color:#192e40}p{line-height:1.65}nav{position:sticky;top:0;background:white;padding:14px;z-index:2}nav a{margin:8px}a{color:#166d9b}section{background:white;padding:18px;margin:24px 0;scroll-margin-top:70px}img{width:100%}td,th{padding:6px 12px}table{border-collapse:collapse}</style><h1>电流处理第一版：清除坏记录、标记饱和、分离趋势与波动</h1><p>灰块：case2/run1、case12/run1的电流观测剔除，保留时间和元数据。红线：DC顶端代码，真实幅值没有恢复。AC波形保留；第二行新增204ms RMS及2Hz包络。DC趋势对比2Hz（绿）、5Hz（橙）、10Hz（紫），原始波动没有被丢弃。趋势中的空白包含饱和邻域、滤波边界及过短片段。</p><p>这是当前run完成后的离线预处理，使用run内未来采样；不跨run或缺失区间滤波、不插值。滤掉的分量不一定是杂波。未改变训练管线，未证明预测性能改善。<a href="RESULTS.md">完整处理说明</a> · <a href="../raw_current_stitched_20261012/index.html">原始电流</a> · <a href="quality_summary.csv">质量明细</a></p><details><summary>局部波形与频谱对照</summary><img src="local_and_spectral_checks.png"></details><nav>NAV</nav>PANELS</html>'''.replace('NAV',nav).replace('PANELS',''.join(panels))
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
