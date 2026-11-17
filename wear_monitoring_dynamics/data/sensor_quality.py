"""Label-free record integrity checks, calibrated on training cases only."""
import numpy as np
import pandas as pd
from .features import SENSORS

VERSION = 'sensor_integrity_v1'


def raw_quality_metrics(records):
    rows = []
    for record in records:
        signals = [np.asarray(record.get(c, []), dtype=float).ravel() for c in SENSORS]
        lengths = [len(x) for x in signals]
        hard = []
        if min(lengths) == 0: hard.append('missing_or_empty_channel')
        if len(set(lengths)) != 1: hard.append('channel_length_mismatch')
        if any(not np.isfinite(x).all() for x in signals): hard.append('nonfinite_signal')
        if any(len(x) and np.ptp(x) == 0 for x in signals): hard.append('constant_channel')
        if any(len(x) and np.max(abs(x)) > 1e6 for x in signals): hard.append('absolute_limit_1e6')
        for channel, x in zip(SENSORS, signals):
            if not len(x) or not np.isfinite(x).all():
                x = np.zeros(1)
            absolute = abs(x); a99 = float(np.quantile(absolute, .99))
            span = float(np.quantile(x, .95) - np.quantile(x, .05))
            peak = float(absolute.max()); energy = x*x
            top = max(1, int(np.ceil(len(x)*.001)))
            share = float(np.sort(energy)[-top:].sum()/max(energy.sum(), 1e-300))
            rows.append(dict(case=int(record['case']), run=int(record['run']), channel=channel,
                n=len(x), a99=a99, span=span, peak=peak, tail_energy_share=share,
                log_a99=float(np.log10(max(a99,1e-300))), log_span=float(np.log10(max(span,1e-300))),
                log_spike=float(np.log10(max(peak/max(a99,1e-300),1.))), hard_reason=';'.join(hard)))
    return pd.DataFrame(rows)


def fit_rules(metrics, train_cases, multiplier=6.):
    """Median/MAD bounds with a conservative three-decade minimum separation."""
    train = metrics[metrics.case.isin(train_cases) & (metrics.hard_reason=='')]
    if train.empty: raise ValueError('No structurally valid training signals')
    rules = dict(version=VERSION, train_cases=list(map(int,train_cases)), multiplier=float(multiplier),
                 minimum_log_margin=3., near_zero_fraction=.5, tail_energy_share=.95,
                 degraded_channels=3, spiky_channels=2, channels={})
    for channel in SENSORS:
        g = train[train.channel==channel]
        if len(g)<3: raise ValueError('Insufficient training records for sensor calibration')
        q = dict(reference_length=int(g.n.mode().iloc[0]), near_zero_epsilon=float(np.median(g.a99)*1e-6))
        for col, direction in [('log_a99',-1),('log_span',-1),('log_spike',1)]:
            median = float(g[col].median()); mad = float((g[col]-median).abs().median())
            q[col+'_threshold'] = median + direction * max(3., multiplier*1.4826*mad)
        rules['channels'][channel] = q
    return rules


def assess(records, metrics, rules):
    by_key={(int(r['case']),int(r['run'])):r for r in records}
    channel_rows=[]; decisions=[]
    for (case,run), group in metrics.groupby(['case','run'],sort=True):
        degraded=spiky=0; length_flag=False
        for r in group.itertuples():
            threshold=rules['channels'][r.channel]
            x=np.asarray(by_key[(case,run)].get(r.channel,[]),dtype=float).ravel()
            fraction=float(np.mean(abs(x)<threshold['near_zero_epsilon'])) if len(x) else 1.
            low = r.log_a99 < threshold['log_a99_threshold'] and r.log_span < threshold['log_span_threshold'] and fraction>=.5
            spike = r.log_spike > threshold['log_spike_threshold'] and r.tail_energy_share>.95
            unusual_length=r.n!=threshold['reference_length'];length_flag |= unusual_length
            degraded+=int(low);spiky+=int(spike)
            channel_rows.append(dict(case=case,run=run,channel=r.channel,near_zero_fraction=fraction,
                                     degraded=bool(low),spiky=bool(spike),length_flag=bool(unusual_length)))
        reasons=[x for x in str(group.hard_reason.iloc[0]).split(';') if x]
        if degraded>=3:reasons.append('multichannel_degradation')
        if spiky>=2:reasons.append('multichannel_spike_contamination')
        if length_flag and (degraded>=2 or spiky>=1):reasons.append('length_and_numeric_anomaly')
        decisions.append(dict(case=case,run=run,sensor_valid=not reasons,reason=';'.join(reasons),
                              degraded_channels=degraded,spiky_channels=spiky,length_flag=length_flag))
    return pd.DataFrame(decisions), pd.DataFrame(channel_rows)


def apply_quality(table, config):
    if not config:return table
    invalid={tuple(x) for x in config['invalid_records']}
    result=dict(table)
    result['sensor_valid']=np.array([(int(c),int(r)) not in invalid for c,r in zip(table['case'],table['run'])])
    return result
