"""Re-evaluate saved epoch-500 predictions, without training or selecting epochs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import project_path,write_json
from ..evaluation.bootstrap import paired_case_bootstrap
from ..evaluation.monotonicity import per_case_monotonicity
from .run_fixed_conditioning import MODELS


def main():
    output=project_path('wear_monitoring_dynamics/outputs')
    root=output/'top18_fixed500_20260928';root.mkdir(exist_ok=True)
    all_predictions=[];cases=[];runs=[];monos=[];sources=[]
    for mode,study in [('pure','top18_sensor_features_20260927'),('ode','top18_sensor_ode_20260927')]:
        studyroot=output/study
        manifest=pd.read_csv(studyroot/'study_manifest.csv')
        expected=pd.read_csv(studyroot/'summary/final_epoch_curves_5seed.csv').query('epoch==500')
        for run in manifest[(manifest.type=='monitor')&(manifest.role=='final')].itertuples():
            directory=Path(run.run_dir)
            assert json.loads((directory/'status.json').read_text())['status']=='completed'
            assert (directory/'epochs/epoch_0500.pt').exists()
            frame=pd.read_csv(directory/'epoch_predictions.csv.gz',usecols=['epoch','split','task','case','origin_run','target_vb_mm','predicted_vb_mm'])
            frame=frame[(frame.epoch==500)&(frame.split=='test_diagnostic')&(frame.task=='monitoring')].copy()
            assert len(frame)>0 and frame.case.nunique()==7
            assert np.isfinite(frame[['target_vb_mm','predicted_vb_mm']].to_numpy()).all()
            assert not frame.duplicated(['case','origin_run']).any()
            name=run.experiment.split('_top18')[0]
            meta=dict(model=name,mode=mode,fold=run.fold,seed=run.seed,epoch=500)
            frame=frame.assign(**meta);all_predictions.append(frame)
            frame['absolute_error']=(frame.predicted_vb_mm-frame.target_vb_mm).abs()
            frame['squared_error']=(frame.predicted_vb_mm-frame.target_vb_mm)**2
            case=frame.groupby('case').agg(mae=('absolute_error','mean'),mse=('squared_error','mean'),n=('case','size')).reset_index().assign(**meta)
            cases.append(case)
            history=pd.read_csv(directory/'history.csv');h=history[history.epoch==500].iloc[0]
            assert abs(case.mae.mean()-h.test_diagnostic_monitoring_case_macro_mae)<1e-7
            runs.append(dict(**meta,mae=case.mae.mean(),rmse=np.sqrt(frame.squared_error.mean())))
            d=per_case_monotonicity(frame.rename(columns={'origin_run':'run'})).assign(**meta);monos.append(d)
            sources.append(dict(**meta,run_dir=str(directory),prediction_file=str(directory/'epoch_predictions.csv.gz')))
        actual=pd.DataFrame(runs).query('mode==@mode').groupby(['model','fold']).mae.mean()
        for row in expected.itertuples():
            assert abs(actual.loc[(row.experiment.split('_top18')[0],row.fold)]-row.mean_test_case_macro_mae)<1e-7
    case=pd.concat(cases,ignore_index=True);diag=pd.concat(monos,ignore_index=True);rf=pd.DataFrame(runs)
    rows=[]
    for name,label,_,_ in MODELS[1:]:
        q=case[case.model==name]
        a=q[q['mode']=='ode'].groupby(['fold','case']).mae.mean()
        b=q[q['mode']=='pure'].groupby(['fold','case']).mae.mean()
        row=dict(model=name,label=label,**paired_case_bootstrap(a,b))
        for mode in ('pure','ode'):
            c=q[q['mode']==mode];d=diag[(diag.model==name)&(diag['mode']==mode)]
            row[mode+'_mae']=c.mae.mean()
            row[mode+'_mean_run_rmse']=rf[(rf.model==name)&(rf['mode']==mode)].rmse.mean()
            row[mode+'_violation_rate']=d.decreases.sum()/d.pairs.sum()
            row[mode+'_decreases']=int(d.decreases.sum());row[mode+'_pairs']=int(d.pairs.sum())
            row[mode+'_mean_drop_mm']=(d.mean_drop_mm*d.pairs).sum()/d.pairs.sum()
            row[mode+'_max_drop_mm']=d.max_drop_mm.max()
            for fold in ('fold_a','fold_b'):row[mode+'_'+fold+'_mae']=c[c.fold==fold].mae.mean()
        row['ode_relative_change_pct']=100*(row['ode_mae']/row['pure_mae']-1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
    case.to_csv(root/'per_case_metrics.csv',index=False);diag.to_csv(root/'monotonicity_per_case.csv',index=False)
    rf.to_csv(root/'run_metrics.csv',index=False);pd.DataFrame(sources).to_csv(root/'source_runs.csv',index=False)
    pred=pd.concat(all_predictions,ignore_index=True);pred.to_csv(root/'epoch500_test_predictions.csv',index=False)
    assert len(rf)==80 and len(case)==560
    write_json(root/'audit.json',dict(epoch=500,final_runs=80,case_seed_evaluations=len(case),prediction_rows=len(pred),
        retrained=False,history_and_saved_summary_match=True,all_checks_passed=True,
        interpretation='Fixed-budget post-hoc diagnostic on existing test cases; globally selected 18 features. Not a new independent test.'))
    print(pd.DataFrame(rows).drop(columns='label').to_string(index=False))


if __name__=='__main__':main()
