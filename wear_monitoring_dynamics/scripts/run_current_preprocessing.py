"""Fixed current-only features, four train-fitted preprocessing protocols."""
import argparse
import copy
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import PACKAGE,project_path,write_json
from ..models.registry import build_model
from ..training.trainer import run_training

ROOT=project_path('wear_monitoring_dynamics/outputs/current_preprocessing_20261010')
PREVIOUS=ROOT.with_name('psd_dimension_sweep_20261010')
METHODS=['zscore','log_zscore','robust','clip_zscore']

def plan(root,top_k=None):
    root.mkdir(parents=True,exist_ok=True)
    old=pd.read_csv(PREVIOUS/'study_manifest.csv');parents=old[old.group=='A'];assert len(parents)==24
    tasks=[];features=None;log_features=None;ranking_cache={};rankings=[];selections=[]
    for r in parents.itertuples():
        base=json.loads((Path(r.run_dir)/'provenance.json').read_text())['config']
        names=base['data']['feature_archive']['names'];idx=[j for j,f in enumerate(names) if f.startswith(('smcAC_','smcDC_'))]
        logidx=[j for j in idx if names[j].endswith(('_std','_rms','_peak_to_peak','_energy','_power'))]
        assert len(idx)==50 and len(logidx)==20
        if top_k is not None:
            from ..data.records import load_table
            from ..data.feature_archive import attach_features
            from ..data.sensor_quality import apply_quality
            from ..data.splits import make_split
            from .run_psd_sweep import rank
            split=make_split(r.fold,r.role,base['experiment']['holdout']);key=tuple(split['train'])
            if key not in ranking_cache:
                table,_=load_table(project_path(base['data']['path']),project_path(base['data']['cache']))
                table=apply_quality(attach_features(table,base['data']['feature_archive']),base['data']['sensor_quality'])
                scores,_=rank(table,split['train'])
                ordered=sorted(idx,key=lambda j:(-scores[j],names[j]));ranking_cache[key]=ordered[:top_k]
                for position,j in enumerate(ordered):rankings.append(dict(fold=r.fold,role=r.role,holdout=base['experiment']['holdout'],rank=position+1,feature=names[j],index=j,score=scores[j]))
                selections.append(dict(fold=r.fold,role=r.role,holdout=base['experiment']['holdout'],train_cases=split['train'],features=[names[j] for j in ordered[:top_k]],indices=ordered[:top_k]))
            idx=ranking_cache[key];logidx=[j for j in idx if j in logidx]
            assert len(idx)==top_k
        features=[names[j] for j in idx];log_features=[names[j] for j in logidx]
        for method in METHODS:
            cfg=copy.deepcopy(base);cfg['study_id']=root.name;cfg['experiment']['name']=method
            cfg['monitor'].update(inputs=['sensor','condition','time'],selected_indices=idx,selected_features=features,skip_invalid_sensor=True,fallback_checkpoint=str(Path(r.run_dir)/'last.pt'))
            cfg['data']['sensor_preprocessing']=dict(method=method,indices=logidx if method=='log_zscore' else idx)
            if method=='clip_zscore':cfg['data']['sensor_preprocessing']['quantiles']=[.01,.99]
            tasks.append(dict(id=f'{r.fold}/{r.role}/{cfg["experiment"]["holdout"]}/{method}/{r.seed}',name=method,fold=r.fold,role=r.role,holdout=cfg['experiment']['holdout'],seed=r.seed,config=cfg))
    assert len(tasks)==96
    parents.to_csv(root/'reused_baseline_manifest.csv',index=False);write_json(root/'planned_tasks.json',tasks)
    write_json(root/'feature_protocol.json',dict(features=features,log_features=log_features,hidden_dims=[64,32],parameters=sum(p.numel() for p in build_model(tasks[0]['config']).parameters())))
    if top_k is not None:
        pd.DataFrame(rankings).to_csv(root/'feature_rankings.csv',index=False)
        write_json(root/'split_feature_selections.json',selections)
        write_json(root/'feature_protocol.json',dict(top_k=top_k,candidate_features=50,selection_file='split_feature_selections.json',rank_method='mean within-training-case absolute Spearman(feature,VB)',hidden_dims=[64,32],parameters=sum(p.numel() for p in build_model(tasks[0]['config']).parameters())))
    return tasks

def report(root,epochs=500):
    manifest=pd.read_csv(root/'study_manifest.csv');rows=[];inner=[];curves=[];seconds=0
    if epochs==500:assert len(manifest)==96
    for r in manifest.itertuples():
        d=Path(r.run_dir);m=json.loads((d/'metrics.json').read_text());h=pd.read_csv(d/'history.csv')
        assert m['epochs']==epochs and m['selected_epoch']==epochs and m['reload_verified'] and h.epoch.tolist()==list(range(1,epochs+1))
        seconds+=m['wall_seconds'];curves.append(h.assign(method=r.name,fold=r.fold,role=r.role,holdout=r.holdout,seed=r.seed))
        if r.role=='inner':inner.append(dict(fold=r.fold,method=r.name,holdout=r.holdout,mae=m['monitoring']['case_macro_mae']))
        else:rows.append(pd.read_csv(d/'monitoring_predictions.csv').assign(method=r.name,fold=r.fold,seed=r.seed))
    baseline=pd.read_csv(root/'reused_baseline_manifest.csv');baseline=baseline[baseline.role=='final']
    if epochs!=500:baseline=baseline[(baseline.fold=='fold_a')&(baseline.seed==0)]
    for r in baseline.itertuples():rows.append(pd.read_csv(Path(r.run_dir)/'monitoring_predictions.csv').assign(method='ct_baseline',fold=r.fold,seed=r.seed))
    pred=pd.concat(rows);pred['absolute_error']=abs(pred.predicted_vb_mm-pred.vb_mm);pred.to_csv(root/'test_predictions.csv',index=False)
    pd.concat(curves).to_csv(root/'epoch_metrics.csv.gz',index=False)
    cases=pred.groupby(['method','fold','seed','case']).absolute_error.mean().rename('mae').reset_index();cases.to_csv(root/'case_metrics.csv',index=False)
    scores=pd.DataFrame(inner).groupby(['fold','method']).mae.mean().reset_index();scores.to_csv(root/'inner_validation_scores.csv',index=False)
    selected=scores.sort_values(['mae','method']).groupby('fold',as_index=False).first();selected.to_csv(root/'selected_preprocessing.csv',index=False)
    chosen=pd.concat([pred[(pred.fold==r.fold)&(pred.method==r.method)].assign(method='validation_selected') for r in selected.itertuples()])
    summaries=[]
    for method,q in pd.concat([pred,chosen]).groupby('method'):
        masks=dict(overall=np.ones(len(q),bool),fold_a=q.fold=='fold_a',fold_b=q.fold=='fold_b',fold_a_in_range=(q.fold=='fold_a')&(q.vb_mm<=.74),fold_a_out_of_range=(q.fold=='fold_a')&(q.vb_mm>.74))
        for subset,mask in masks.items():
            s=q[mask]
            if len(s):summaries.append(dict(method=method,subset=subset,mae=s.groupby(['fold','seed','case']).absolute_error.mean().mean()))
    summary=pd.DataFrame(summaries);summary.to_csv(root/'summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for ax,fold in zip(axes,['fold_a','fold_b']):
        q=cases[cases.fold==fold].groupby(['case','method']).mae.mean().unstack('method')
        if q.empty:
            ax.set_visible(False);continue
        q.plot.bar(ax=ax);ax.set(title=fold,ylabel='Case-macro MAE (mm)');ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'case_mae.png',dpi=150);plt.close(fig)
    def md(df):return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])
    first_cfg=json.loads((Path(manifest.iloc[0].run_dir)/'provenance.json').read_text())['config']
    count=len(first_cfg['monitor']['selected_indices'])
    lines=['# 电流特征预处理对照','',f'{len(manifest)}次新训练，每次{epochs}epoch，累计{seconds/60:.2f}分钟；ct基线复用此前固定500epoch模型。','',
       f'每个划分四组固定同一{count}维AC/DC电流特征、同一MLP64/32及同seed；预处理之间不重新排名。18维实验仅用该划分训练case原始特征排名，取前18；不同划分名单允许不同。原始time/condition/VB处理不变。',
       '','## 主结果','',md(summary),'','## 内层验证选择','',md(selected),
       '','ct基线只作无sensor参考；四种预处理之间是同参数量比较。测试结果不能用来重新选择阈值或特征。',
       'log仅用于选中特征中属于预设非负幅度/功率类别的维度：log1p(x/s)，s为训练正值中位数；随后训练z-score。robust为训练median/IQR；clip为训练1%/99%分位裁剪，再用裁剪后的训练数据拟合z-score，测试同阈值裁剪。',
       '质量规则沿用同split；case2/run1继续隔离，case12/run1冻结ct后备，不改变138个外层标签。频谱250Hz已在原始mill/Readme.pdf第3页核实，AE/振动预处理链不影响本次电流限定。',
       '全部逐epoch指标和预测保留，五seed不当作独立case。此研究已观察过测试异常，为探索性预处理对照；不能把更小预测误差反过来当作坏数据判据。',
       '','![逐case误差](case_mae.png)']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8');write_json(root/'status.json',dict(status='completed' if epochs==500 else 'smoke_completed',runs=len(manifest),epochs=epochs,training_seconds=seconds))

def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--report-only',action='store_true');parser.add_argument('--top18',action='store_true');args=parser.parse_args()
    if args.top18:ROOT=ROOT.with_name('current_top18_preprocessing_20261010')
    root=ROOT.with_name(ROOT.name+'_smoke') if args.smoke else ROOT
    if args.report_only:report(root);return
    tasks=plan(root,18 if args.top18 else None)
    if not(args.execute or args.smoke):return
    if args.smoke:tasks=[t for t in tasks if t['fold']=='fold_a' and t['seed']==0 and (t['role']=='final' or t['holdout']==1)]
    path=root/'study_manifest.csv';old={r['id']:r for r in pd.read_csv(path).to_dict('records')} if path.exists() else {};rows=[]
    for t in tasks:
        cfg=copy.deepcopy(t['config'])
        if args.smoke:cfg['training'].update(epochs=3,selection_epoch=3)
        if t['id'] in old:
            directory=Path(old[t['id']]['run_dir']);identity=json.loads((directory/'provenance.json').read_text());assert identity['config']==cfg
            with zipfile.ZipFile(project_path(cfg['output_root'])/'sources'/(identity['source_hash']+'.zip')) as z:
                for p in PACKAGE.rglob('*.py'):
                    if p.relative_to(PACKAGE).parts[0] not in ('scripts','tests','outputs'):assert z.read(p.relative_to(PACKAGE.parent).as_posix())==p.read_bytes(),'Execution source changed'
            assert json.loads((directory/'status.json').read_text())['status']=='completed'
        else:directory=run_training(cfg)
        rows.append({**{k:v for k,v in t.items() if k!='config'},'run_dir':str(directory)});pd.DataFrame(rows).to_csv(path,index=False)
        write_json(root/'status.json',dict(status='running',completed=len(rows),planned=len(tasks)))
    report(root,3 if args.smoke else 500)

if __name__=='__main__':
    try:main()
    except Exception as e:
        import sys
        root=ROOT.with_name(ROOT.name+'_smoke') if '--smoke' in sys.argv else ROOT
        write_json(root/'failure.json',dict(error=str(e)));raise
