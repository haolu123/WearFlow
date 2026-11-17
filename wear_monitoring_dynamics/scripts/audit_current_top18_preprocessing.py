"""Audit current-only preprocessing experiment without retraining."""

import json

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from ..config import project_path,digest,write_json

from ..data.records import load_table

from ..data.feature_archive import attach_features

from ..data.sensor_quality import apply_quality

from ..data.preprocessing import fit_scalers,transform

from ..data.splits import make_split

from ..data.monitor_batch import make_monitor_batch

from ..models.registry import build_model

from .run_current_preprocessing import ROOT

from .run_psd_sweep import rank

ROOT=ROOT.with_name("current_top18_preprocessing_20261010")



def md(df):

    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in r)+' |' for r in df.itertuples(index=False,name=None)])



def main():

    torch.set_num_threads(1)

    m=pd.read_csv(ROOT/'study_manifest.csv');assert len(m)==96 and m.id.nunique()==96

    audits=[];extremes=[];clips=[];initial={};feature_lists={}

    for r in m.itertuples():

        d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config'];s=json.loads((d/'scalers.json').read_text());split=make_split(r.fold,r.role,None if pd.isna(r.holdout) else int(r.holdout))

        assert split==json.loads((d/'split.json').read_text())

        t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))

        t=apply_quality(attach_features(t,cfg['data']['feature_archive']),cfg['data']['sensor_quality'])

        assert digest(fit_scalers(t,split['train'],cfg['data']['sensor_preprocessing']))==digest(s)

        assert cfg['data']['sensor_quality']['rules']['train_cases']==split['train']

        ix=cfg['monitor']['selected_indices'];assert len(ix)==18

        names=cfg['data']['feature_archive']['names']

        candidates=[j for j,n in enumerate(names) if n.startswith(('smcAC_','smcDC_'))]

        scores,_=rank(t,split['train'])

        expected=sorted(candidates,key=lambda j:(-scores[j],names[j]))[:18]

        assert ix==expected

        splitkey=tuple(split['train'])

        if splitkey not in feature_lists:feature_lists[splitkey]=ix

        assert ix==feature_lists[splitkey]

        assert cfg['monitor']['selected_features']==[names[j] for j in ix]

        expected_transform=[j for j in ix if names[j].endswith(('_std','_rms','_peak_to_peak','_energy','_power'))] if r.name=='log_zscore' else ix

        assert cfg['data']['sensor_preprocessing']['indices']==expected_transform

        parent=Path(cfg['monitor']['fallback_checkpoint']).parent;pc=json.loads((parent/'provenance.json').read_text())['config']

        assert pc['experiment']['seed']==r.seed and json.loads((parent/'split.json').read_text())==split

        # Mirror RNG-consuming fallback construction before model initialization.

        torch.manual_seed(r.seed);batch=make_monitor_batch(cfg,t,s,'cpu');model=build_model(cfg)

        assert sum(p.numel() for p in model.parameters())==3649

        key=(r.fold,r.role,str(r.holdout),r.seed)

        state={k:v.detach().clone() for k,v in model.state_dict().items()}

        if key in initial:

            for k,v in state.items():torch.testing.assert_close(v,initial[key][k],rtol=0,atol=0)

        else:initial[key]=state

        checkpoint=torch.load(d/'last.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['model']);model.eval()

        with torch.no_grad():

            ids=np.flatnonzero(~t['sensor_valid']);pred=model.estimate(batch,ids)

            torch.testing.assert_close(pred,batch['fallback_vb_z'][ids],rtol=0,atol=0)

        hist=pd.read_csv(d/'history.csv');assert hist.epoch.tolist()==list(range(1,501))

        assert hist.filter(regex='^(train|validation|test_diagnostic)_').notna().all().all()

        met=json.loads((d/'metrics.json').read_text());assert met['epochs']==met['selected_epoch']==500 and met['selected_checkpoint']=='last.pt' and met['reload_verified']

        if r.role=='final':assert len(list((d/'epochs').glob('epoch_*.pt')))==500

        p=pd.read_csv(d/'epoch_predictions.csv.gz');assert np.isfinite(p[['target_vb_mm','predicted_vb_mm']]).all().all()

        for (group,task),q in p.groupby(['split','task']):

            assert task=='monitoring' and q.epoch.nunique()==500 and q.groupby('epoch').size().nunique()==1

            cases=split['outer_test_reserved'] if group=='test_diagnostic' else split[group]

            target={(int(c),int(run)) for c,run,v in zip(t['case'],t['run'],t['VB']) if c in cases and np.isfinite(v)}

            assert set(zip(q.case,q.target_run))==target and (q.origin_run==q.target_run).all()

        group='validation' if r.role=='inner' else 'test_diagnostic';q=p[(p.epoch==500)&(p.split==group)]

        assert abs(abs(q.predicted_vb_mm-q.target_vb_mm).groupby(q.case).mean().mean()-met['monitoring']['case_macro_mae'])<1e-6

        audits.append(dict(id=r.id,prediction_rows=len(p),seconds=met['wall_seconds']))

        if r.role=='final' and r.seed==0:

            z=transform(t,s)['sensor'][:,ix]

            for case in split['test']:

                mask=(t['case']==case)&t['sensor_valid']

                extremes.append(dict(fold=r.fold,case=case,method=r.name,max_abs_scaled=abs(z[mask]).max(),pct_abs_scaled_gt3=100*np.mean(abs(z[mask])>3)))

                if r.name=='clip_zscore':

                    st=s['sensor_transform'];values=t['sensor'][mask][:,st['indices']];hit=(values<st['lower'])|(values>st['upper'])

                    clips.append(dict(fold=r.fold,case=case,records=int(mask.sum()),pct_feature_cells_clipped=100*hit.mean(),pct_records_any_clipped=100*hit.any(1).mean()))

    audit=pd.DataFrame(audits);audit.to_csv(ROOT/'run_audit.csv',index=False)

    pd.DataFrame(extremes).to_csv(ROOT/'scaled_input_extremes.csv',index=False);cl=pd.DataFrame(clips);cl.to_csv(ROOT/'clipping_coverage.csv',index=False)

    p=pd.read_csv(ROOT/'test_predictions.csv');reference=p[p.method=='ct_baseline'][['fold','seed','case','run','vb_mm']]

    for method,q in p[p.method!='ct_baseline'].groupby('method'):

        match=q.merge(reference,on=['fold','seed','case','run'],validate='one_to_one',suffixes=('','_ref'));assert len(match)==len(q)==690

        np.testing.assert_array_equal(match.vb_mm,match.vb_mm_ref)

    scores=pd.read_csv(ROOT/'inner_validation_scores.csv');expected=scores.sort_values(['mae','method']).groupby('fold',as_index=False).first()

    pd.testing.assert_frame_equal(expected,pd.read_csv(ROOT/'selected_preprocessing.csv'),check_exact=False)

    case=pd.read_csv(ROOT/'case_metrics.csv').groupby(['fold','case','method']).mae.mean().unstack('method');case.to_csv(ROOT/'case_mean_mae.csv')

    summary=pd.read_csv(ROOT/'summary.csv').pivot(index='method',columns='subset',values='mae').reset_index()

    old=pd.read_csv(ROOT.with_name('current_preprocessing_20261010')/'summary.csv')

    comparison=pd.read_csv(ROOT/'summary.csv').merge(old,on=['method','subset'],suffixes=('_18','_50'))

    comparison['delta_18_minus_50']=comparison.mae_18-comparison.mae_50

    comparison.to_csv(ROOT/'comparison_with_50.csv',index=False)

    selections=json.loads((ROOT/'split_feature_selections.json').read_text())

    assert len(feature_lists)==len(selections)==16

    composition=[]

    for item in selections:

        assert item['indices']==feature_lists[tuple(item['train_cases'])]

        if item['role']=='final':

            fs=item['features'];freq=sum('_psd_' in f for f in fs)

            composition.append(dict(fold=item['fold'],time_features=18-freq,psd_features=freq,AC=sum(f.startswith('smcAC_') for f in fs),DC=sum(f.startswith('smcDC_') for f in fs)))

    comp=pd.DataFrame(composition);comp.to_csv(ROOT/'feature_composition.csv',index=False)

    lines=['','## 完成审计与结论','',

      '固定18维电流特征下，zscore总体0.09998；log为0.09526（下降4.7%）；robust为0.09946（基本持平）；clip为0.08950（下降10.5%）。裁剪收益主要来自fold B：0.09820降至0.07627，fold A略变差0.10175至0.10272。总体仍未超过无sensor参考0.08349。',

      '内层验证fold A选择clip、fold B选择robust，最终总体0.10153。验证选择没有转化为测试收益；不能把测试最好clip称为验证选出的最优方案。',

      '',md(summary),'','### 每case MAE（五seed均值）','',md(case.reset_index()),

      '','### 与此前50维同方法比较','',md(comparison),

      '所有方法在本次18维下比此前50维改善，但维数减少和训练内筛选同时改变，不能归因为预处理单一因素。此前混合六通道top18也不是当前两通道top18名单。',

      '','### 外层训练选中特征组成','',md(comp),

      '完整名单见split_feature_selections.json；频域与时域合计18维。',

      '','### 裁剪覆盖','',md(cl),

      '特征单元比例为有效记录乘18维的裁剪占比，含无VB记录。稳健缩放极值单位为IQR，与z-score标准差单位不能直接等同。详见scaled_input_extremes.csv。',

      f'96次运行各500epoch通过审计，共{int(audit.prediction_rows.sum())}行逐轮预测及20000个final逐轮checkpoint。16个划分重新计算训练内排名均与保存名单一致；同split四方法同名单、3649参数、训练内预处理拟合、同seed初始化重建、后备预测、138点配对和内层选择均通过。初始化为复现调用顺序重建，不是epoch0快照。累计训练{audit.seconds.sum()/60:.2f}分钟。',

      '五seed不作为独立case；结果属观察过测试后的探索性对照。裁剪改变训练分布，不能仅凭误差改善断言异常是唯一原因。未启动下一批。复现审计：python -m wear_monitoring_dynamics.scripts.audit_current_top18_preprocessing。']

    path=ROOT/'RESULTS.md';body=path.read_text(encoding='utf-8').split('\n## 完成审计与结论')[0];path.write_text(body+'\n'.join(lines)+'\n',encoding='utf-8')

    result=dict(passed=True,runs=96,epochs=48000,prediction_rows=int(audit.prediction_rows.sum()),training_seconds=float(audit.seconds.sum()))

    write_json(ROOT/'completion_audit.json',result);print(result);print(case.round(5).to_string());print(cl.round(2).to_string(index=False));print(pd.DataFrame(extremes).pivot(index=['fold','case'],columns='method',values='max_abs_scaled').round(2).to_string())



if __name__=='__main__':main()

