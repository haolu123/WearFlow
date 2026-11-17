"""A/B/C/D current-wear feature-dimension study with nested case validation."""
import argparse
import copy
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from ..config import PACKAGE, project_path, load_config, write_json, file_hash
from ..data.records import load_table
from ..data.features import SENSORS, STATISTICS
from ..data.splits import POOLS, make_split
from ..data.feature_archive import attach_features
from ..data.sensor_quality import apply_quality
from ..models.registry import build_model
from ..training.trainer import run_training

ROOT=project_path('wear_monitoring_dynamics/outputs/psd_dimension_sweep_20261010')
QC=ROOT.with_name('history_quality_skip_20261008')
PSD=ROOT.with_name('psd_inspection_20261010')
FREQ=['ac_power','centroid_hz','bandwidth_hz','entropy','flatness','peak_hz']+[
    f'band_{a}_{b}_{kind}' for a,b in [(0,5),(5,15),(15,30),(30,60),(60,125)] for kind in ['fraction','power']]

def archive(root,table):
    df=pd.read_csv(PSD/'psd_features.csv');names=[f'{c}_{s}' for c in SENSORS for s in STATISTICS]+[f'{c}_psd_{f}' for c in SENSORS for f in FREQ]
    lookup=df.set_index(['case','run','channel']);values=[]
    for i,(case,run) in enumerate(zip(table['case'],table['run'])):
        psd=np.concatenate([lookup.loc[(case,run,c),FREQ].to_numpy(dtype=float) for c in SENSORS])
        vector=np.r_[table['sensor'][i],psd]
        if not bool(lookup.loc[(case,run,SENSORS[0]),'sensor_valid']):vector[:]=0
        assert np.isfinite(vector).all();values.append(vector)
    path=root/'feature_archive.npz'
    # Keep byte identity stable across resumed plans.
    if path.exists():
        with np.load(path,allow_pickle=False) as a:
            np.testing.assert_array_equal(a['features'],values);assert a['names'].tolist()==names
    else:np.savez_compressed(path,features=np.array(values),case=table['case'],run=table['run'],names=np.array(names))
    return dict(path=str(path),sha256=file_hash(path),names=names,source_csv_sha256=file_hash(PSD/'psd_features.csv'))

def rank(table,train_cases):
    scores=[];details=[]
    for j in range(table['sensor'].shape[1]):
        rs=[]
        for case in train_cases:
            idx=(table['case']==case)&table['sensor_valid']&np.isfinite(table['VB'])
            x=table['sensor'][idx,j];y=table['VB'][idx]
            rho=float(spearmanr(x,y).statistic) if len(y)>=3 and np.ptp(x)>0 and np.ptp(y)>0 else 0.
            rs.append(abs(rho));details.append(dict(feature_index=j,case=case,rho=rho))
        scores.append(np.mean(rs))
    return np.asarray(scores),details

def plan(root):
    root.mkdir(parents=True,exist_ok=True);base=load_config(PACKAGE/'configs/time_conditioning.yaml')
    base['study_id']=root.name;base['training'].update(fixed_final_epoch=True,selection_epoch=500)
    table,_=load_table(project_path(base['data']['path']),project_path(base['data']['cache']))
    ar=archive(root,table);base['data']['feature_archive']=ar;table=attach_features(table,ar)
    pools={'A':[],'B':list(range(54)),'C':list(range(54,150)),'D':list(range(150))}
    defs=[('A',0)]+[(group,k) for group in ['B','C','D'] for k in sorted(set([6,12,18,36,54,len(pools[group])]))]
    tasks=[];rankings=[];params=[]
    for fold in POOLS:
        for role,holdout in [('inner',h) for h in POOLS[fold]]+[('final',None)]:
            split=make_split(fold,role,holdout)
            quality=json.loads((QC/'quality_rules'/f'{fold}_{role}_{holdout}.json').read_text())
            assert quality['rules']['train_cases']==split['train']
            qt=apply_quality(table,quality);scores,details=rank(qt,split['train'])
            for group,k in defs:
                ordered=sorted(pools[group],key=lambda j:(-scores[j],ar['names'][j]))
                selected=ordered[:k];name=f'{group}_k{k:03d}'
                if k==len(pools[group]) and k:
                    for pos,j in enumerate(ordered):rankings.append(dict(fold=fold,role=role,holdout=holdout,group=group,rank=pos+1,feature=ar['names'][j],index=j,score=scores[j]))
                for seed in ([0] if role=='inner' else range(5)):
                    cfg=copy.deepcopy(base);cfg['data']['sensor_quality']=quality
                    cfg['experiment'].update(name=name,fold=fold,role=role,holdout=holdout,seed=seed)
                    cfg['monitor']=dict(architecture='feature_mlp',inputs=['condition','time'] if group=='A' else ['sensor','condition','time'],hidden_dims=[64,32],selected_indices=selected,selected_features=[ar['names'][j] for j in selected],skip_invalid_sensor=group!='A')
                    tasks.append(dict(id=f'{fold}/{role}/{holdout}/{name}/{seed}',name=name,group=group,k=k,fold=fold,role=role,holdout=holdout,seed=seed,config=cfg))
                    if fold=='fold_a' and role=='final' and seed==0:params.append(dict(name=name,k=k,parameters=sum(p.numel() for p in build_model(cfg).parameters())))
    assert len(tasks)==432
    write_json(root/'planned_tasks.json',tasks);pd.DataFrame(rankings).to_csv(root/'feature_rankings.csv',index=False);pd.DataFrame(params).to_csv(root/'parameter_counts.csv',index=False)
    return tasks

def report(root,epochs=500):
    m=pd.read_csv(root/'study_manifest.csv');inner=[];final=[];seconds=0;epochrows=[]
    for r in m.itertuples():
        d=Path(r.run_dir);metrics=json.loads((d/'metrics.json').read_text());h=pd.read_csv(d/'history.csv')
        assert h.epoch.tolist()==list(range(1,epochs+1)) and metrics['selected_epoch']==epochs and metrics['reload_verified']
        seconds+=metrics['wall_seconds'];epochrows.append(h.assign(name=r.name,fold=r.fold,role=r.role,holdout=r.holdout,seed=r.seed))
        if r.role=='inner':inner.append(dict(fold=r.fold,group=r.group,name=r.name,k=r.k,holdout=r.holdout,mae=metrics['monitoring']['case_macro_mae']))
        else:final.append(pd.read_csv(d/'monitoring_predictions.csv').assign(name=r.name,group=r.group,k=r.k,fold=r.fold,seed=r.seed))
    inner=pd.DataFrame(inner);scores=inner.groupby(['fold','group','name','k']).mae.mean().reset_index()
    chosen=scores.sort_values(['mae','k','name']).groupby(['fold','group'],as_index=False).first()
    if epochs==500:
        assert len(m)==432 and len(final)==180 and len(inner)==252
        for (fold,name),q in inner.groupby(['fold','name']):assert set(q.holdout)==set(POOLS[fold])
    inner.to_csv(root/'inner_case_scores.csv',index=False);scores.to_csv(root/'inner_validation_scores.csv',index=False);chosen.to_csv(root/'selected_dimensions.csv',index=False)
    p=pd.concat(final);p['absolute_error']=abs(p.predicted_vb_mm-p.vb_mm);p.to_csv(root/'test_predictions.csv',index=False)
    p.groupby(['name','fold','seed','case']).absolute_error.mean().rename('mae').to_csv(root/'case_metrics.csv')
    pd.concat(epochrows).to_csv(root/'epoch_metrics.csv.gz',index=False)
    selected=pd.concat([p[(p.fold==r.fold)&(p.name==r.name)].assign(name='selected_'+r.group) for r in chosen.itertuples()])
    summaries=[]
    for name,q in pd.concat([p,selected]).groupby('name'):
        masks=dict(overall=np.ones(len(q),bool),fold_a=q.fold=='fold_a',fold_b=q.fold=='fold_b',fold_a_in_range=(q.fold=='fold_a')&(q.vb_mm<=.74),fold_a_out_of_range=(q.fold=='fold_a')&(q.vb_mm>.74))
        for subset,mask in masks.items():
            s=q[mask]
            if len(s):summaries.append(dict(name=name,subset=subset,mae=s.groupby(['fold','seed','case']).absolute_error.mean().mean()))
    summary=pd.DataFrame(summaries);summary.to_csv(root/'summary.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for ax,subset in zip(axes,['overall','fold_a','fold_b']):
        for group in ['B','C','D']:
            s=summary[(summary.subset==subset)&summary.name.str.startswith(group+'_')].copy();s['k']=s.name.str.split('k').str[-1].astype(int);s=s.sort_values('k');ax.plot(s.k,s.mae,'o-',label=group)
        baseline=summary[(summary.subset==subset)&(summary.name=='A_k000')]
        if len(baseline):ax.axhline(baseline.mae.iloc[0],ls='--',color='black',label='A: condition+time')
        ax.set(title=subset,xlabel='Selected sensor feature count',ylabel='Case-macro MAE (mm)');ax.grid(alpha=.2);ax.legend()
    fig.savefig(root/'dimension_sweep.png',dpi=150);plt.close(fig)
    def md(df):
        return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])
    text=['# PSD与时域特征维数扫描','',f'累计训练 {seconds/60:.2f} 分钟；{len(m)} 次运行，每次 {epochs} epoch。固定末轮。','',
          'A=工况+时间；B=时域；C=频域；D=时域+频域。B/C/D均额外保留工况+时间。','',
          '## 内层验证选择的维数','',md(chosen),'','## 外层测试完整结果','',md(summary),
          '', '![维数扫描](dimension_sweep.png)','',
          '固定维数曲线为描述性扫描，不能从外层测试最低点挑选部署维数。selected_A/B/C/D仅在各fold内层验证选择；不同fold可选不同k。全部逐case、seed和逐epoch指标与预测保留。',
          '固定MLP隐藏宽度64/32，因此维数增加同时增加第一层参数；参数数见parameter_counts.csv，不能把收益完全归于频域信息。全部特征保留原始尺度后使用训练内z-score；未调节对数变换或网络宽度。',
          '排名按训练case内特征与实测VB的绝对Spearman相关系数等权平均，不要求随时间单调，但仍偏向单调的特征-VB关系，可能遗漏非单调交互信息。常量/不足3点的case贡献0。',
          '频谱按先前可视化协议（假定250Hz）预计算，不含VB拟合。质量规则沿用此前每split训练校准结果；已知测试异常启发规则设计，本轮属探索性验证。没有运行历史窗口或ODE。']
    (root/'RESULTS.md').write_text('\n'.join(text),encoding='utf-8');write_json(root/'status.json',dict(status='completed' if epochs==500 else 'smoke_completed',runs=len(m),epochs=epochs,training_seconds=seconds))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--report-only',action='store_true');args=parser.parse_args()
    root=ROOT.with_name(ROOT.name+'_smoke') if args.smoke else ROOT
    if args.report_only:report(root);return
    tasks=plan(root)
    if not(args.execute or args.smoke):return
    if args.smoke:tasks=[t for t in tasks if t['fold']=='fold_a' and t['seed']==0 and (t['role']=='final' or t['holdout']==1)]
    manifest=root/'study_manifest.csv';existing={r['id']:r for r in pd.read_csv(manifest).to_dict('records')} if manifest.exists() else {};rows=[]
    write_json(root/'status.json',dict(status='running',completed=0,planned=len(tasks)))
    for task in tasks:
        cfg=copy.deepcopy(task['config'])
        if args.smoke:cfg['training'].update(epochs=3,selection_epoch=3)
        if task['group']!='A':
            parent=[r for r in rows if r['group']=='A' and all(r[k]==task[k] for k in ['fold','role','holdout','seed'])]
            assert len(parent)==1;cfg['monitor']['fallback_checkpoint']=str(Path(parent[0]['run_dir'])/'last.pt')
        old=existing.get(task['id'])
        if old:
            d=Path(old['run_dir']);provenance=json.loads((d/'provenance.json').read_text());assert provenance['config']==cfg
            with zipfile.ZipFile(project_path(cfg['output_root'])/'sources'/(provenance['source_hash']+'.zip')) as z:
                for path in PACKAGE.rglob('*.py'):
                    if path.relative_to(PACKAGE).parts[0] not in ('scripts','tests','outputs'):assert z.read(path.relative_to(PACKAGE.parent).as_posix())==path.read_bytes(),'Execution code changed'
            assert json.loads((d/'status.json').read_text())['status']=='completed'
        else:d=run_training(cfg)
        rows.append({**{k:v for k,v in task.items() if k!='config'},'run_dir':str(d)})
        pd.DataFrame(rows).to_csv(manifest,index=False);write_json(root/'status.json',dict(status='running',completed=len(rows),planned=len(tasks)))
    report(root,3 if args.smoke else 500)

if __name__=='__main__':
    try:main()
    except Exception as e:
        import sys
        root=ROOT.with_name(ROOT.name+'_smoke') if '--smoke' in sys.argv else ROOT
        write_json(root/'failure.json',dict(error=str(e)));raise
