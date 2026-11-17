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

def md(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in r)+' |' for r in df.itertuples(index=False,name=None)])

def main():
    torch.set_num_threads(1)
    m=pd.read_csv(ROOT/'study_manifest.csv');assert len(m)==96 and m.id.nunique()==96
    audits=[];extremes=[];clips=[];initial={};feature_list=None
    for r in m.itertuples():
        d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config'];s=json.loads((d/'scalers.json').read_text());split=make_split(r.fold,r.role,None if pd.isna(r.holdout) else int(r.holdout))
        assert split==json.loads((d/'split.json').read_text())
        t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']))
        t=apply_quality(attach_features(t,cfg['data']['feature_archive']),cfg['data']['sensor_quality'])
        assert digest(fit_scalers(t,split['train'],cfg['data']['sensor_preprocessing']))==digest(s)
        assert cfg['data']['sensor_quality']['rules']['train_cases']==split['train']
        ix=cfg['monitor']['selected_indices'];assert len(ix)==50
        if feature_list is None:feature_list=ix
        assert ix==feature_list
        parent=Path(cfg['monitor']['fallback_checkpoint']).parent;pc=json.loads((parent/'provenance.json').read_text())['config']
        assert pc['experiment']['seed']==r.seed and json.loads((parent/'split.json').read_text())==split
        # Mirror RNG-consuming fallback construction before model initialization.
        torch.manual_seed(r.seed);batch=make_monitor_batch(cfg,t,s,'cpu');model=build_model(cfg)
        assert sum(p.numel() for p in model.parameters())==5697
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
    lines=['','## 完成审计与结论','',
      '固定四方案中clip_zscore总体最好：0.15832→0.10786，相对zscore降低约31.9%；主要来自fold B。log略改善总体，robust变差。所有固定电流方案仍差于无sensor参考0.08349。',
      '内层验证fold A选clip、fold B选log，选择流程总体0.14795。不能因为看到测试结果就改称clip是验证选出的最优方法。',
      '',md(summary),'','### 每case MAE（五seed均值）','',md(case.reset_index()),
      '','### 裁剪覆盖','',md(cl),
      '特征单元比例是记录×50特征的裁剪占比；任一维被裁剪的记录比例通常更高。所有有效sensor记录含无VB标签，裁剪不删除记录。超过3的稳健缩放数值单位是IQR，不能直接与z-score的标准差单位等同。',
      f'审计96×500=48000epoch，{int(audit.prediction_rows.sum())}行逐轮预测、20000个final逐轮checkpoint、训练内预处理参数、同名单与5697参数、同seed初始化重建、后备预测、138点目标配对及内层选择均通过。初始化为复现训练调用顺序重建检查，不是保存的epoch0快照。累计新训练{audit.seconds.sum()/60:.2f}分钟。',
      '该结果只适用于固定50维电流输入，不能和此前18维混合sensor结果直接归因比较。尚不能证明所有改善来自异常抑制，裁剪也改变了正常训练分布。',
      '本次未启动下一批。复现：python -m wear_monitoring_dynamics.scripts.audit_current_preprocessing。']
    path=ROOT/'RESULTS.md';body=path.read_text(encoding='utf-8').split('\n## 完成审计与结论')[0];path.write_text(body+'\n'.join(lines)+'\n',encoding='utf-8')
    result=dict(passed=True,runs=96,epochs=48000,prediction_rows=int(audit.prediction_rows.sum()),training_seconds=float(audit.seconds.sum()))
    write_json(ROOT/'completion_audit.json',result);print(result);print(case.round(5).to_string());print(cl.round(2).to_string(index=False));print(pd.DataFrame(extremes).pivot(index=['fold','case'],columns='method',values='max_abs_scaled').round(2).to_string())

if __name__=='__main__':main()

