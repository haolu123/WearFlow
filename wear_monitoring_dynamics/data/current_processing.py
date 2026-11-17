"""Offline per-run current views, preserving unknown/censored measurements.

No learned calibration or VB inputs. Trends are auxiliary views, not claims that
all removed frequency components are noise. Never concatenate before filtering.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d,binary_dilation
from scipy.signal import butter,sosfiltfilt

FS=250.
DC_RAIL=9.9951171875
QUARANTINE={(2,1):'legacy multichannel extreme numeric corruption',
            (12,1):'legacy multichannel near-zero degradation and isolated spikes'}


def spans(mask):
    edge=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return list(zip(np.flatnonzero(edge==1),np.flatnonzero(edge==-1)))


def lowpass_segments(x,valid,cutoff,fs=FS,guard_seconds=.5):
    """Independent finite segments, forward/backward filtering, edge guards.

    NaNs remain gaps. No smoothing/interpolation across an unavailable interval.
    """
    x=np.asarray(x,float);valid=np.asarray(valid,bool)&np.isfinite(x)
    y=np.full(len(x),np.nan);sos=butter(4,cutoff,fs=fs,output='sos');guard=int(round(guard_seconds*fs))
    for a,b in spans(valid):
        if b-a<max(2*guard+32,32):continue
        v=sosfiltfilt(sos,x[a:b])
        y[a+guard:b-guard]=v[guard:len(v)-guard] if guard else v
    return y


def process_current(ac,dc,case,run):
    ac=np.asarray(ac,float).ravel();dc=np.asarray(dc,float).ravel()
    if len(ac)!=len(dc):raise ValueError('AC/DC length mismatch')
    n=len(ac);reason=QUARANTINE.get((int(case),int(run)),'')
    if not reason and (not np.isfinite(ac).all() or not np.isfinite(dc).all()):reason='nonfinite current values'
    if not reason and max(np.max(abs(ac)),np.max(abs(dc)))>1e6:reason='extreme current exceeds legacy numeric bound'
    valid=np.full(n,not bool(reason));nan=np.full(n,np.nan)
    out=dict(ac_waveform=ac.copy() if not reason else nan.copy(),dc_observed=dc.copy() if not reason else nan.copy(),
        record_valid=valid.copy(),dc_rail_mask=np.zeros(n,bool),dc_guard_mask=np.zeros(n,bool),ac_rms_204ms=nan.copy(),ac_envelope_2hz=nan.copy(),
        dc_trend_2hz=nan.copy(),dc_trend_5hz=nan.copy(),dc_trend_10hz=nan.copy(),dc_residual_5hz=nan.copy())
    if reason:return out,reason
    # Known repeated top code, not a calibrated physical current ceiling.
    rail=dc>=DC_RAIL-1e-9;guard=binary_dilation(rail,iterations=int(.2*FS))
    out['dc_rail_mask']=rail;out['dc_guard_mask']=guard
    out['dc_observed'][rail]=np.nan
    # 51 samples =204ms; no centering/removal of the recorded AC DC-offset.
    rms=np.sqrt(np.maximum(uniform_filter1d(ac*ac,size=51,mode='reflect'),0))
    out['ac_rms_204ms']=rms.copy();out['ac_rms_204ms'][:25]=np.nan;out['ac_rms_204ms'][-25:]=np.nan
    out['ac_envelope_2hz']=lowpass_segments(rms,np.isfinite(out['ac_rms_204ms']),2.)
    for cutoff in [2,5,10]:out[f'dc_trend_{cutoff}hz']=lowpass_segments(dc,~guard,cutoff)
    out['dc_residual_5hz']=out['dc_observed']-out['dc_trend_5hz']
    return out,''
