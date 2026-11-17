"""Verify completed feature sweep and append descriptive, matched comparisons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import project_path,digest,write_json
from ..data.records import load_table
from ..data.feature_archive import attach_features
from ..data.sensor_quality import apply_quality
from ..data.preprocessing import fit_scalers
from ..data.splits import make_split,POOLS
from .run_psd_sweep import ROOT,rank

def md(df):
    return '\n'.join(['| '+' | '.join(df.columns)+' |','|'+'|'.join(['---']*len(df.columns))+'|']+['| '+' | '.join(f'{x:.5f}' if isinstance(x,float) else str(x) for x in r)+' |' for r in df.itertuples(index=False,name=None)])

def main():
    m=pd.read_csv(ROOT/'study_manifest.csv');assert len(m)==432 and m.id.nunique()==432
    first=json.loads((Path(m.iloc[0].run_dir)/'provenance.json').read_text())['config']
    raw,_=load_table(project_path(first['data']['path']),project_path(first['data']['cache']))
    table=attach_features(raw,first['data']['feature_archive']);cache={};audits=[];composition=[]
    for r in m.itertuples():
        d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config'];split=make_split(r.fold,r.role,None if pd.isna(r.holdout) else int(r.holdout))
        assert cfg['data']['feature_archive']==first['data']['feature_archive']
        assert split==json.loads((d/'split.json').read_text());quality=cfg['data']['sensor_quality']
        assert quality['rules']['train_cases']==split['train'];qt=apply_quality(table,quality)
        key=tuple(split['train'])
        if key not in cache:cache[key]=(fit_scalers(qt,split['train']),rank(qt,split['train'])[0])
        scaler,scores=cache[key];assert digest(scaler)==digest(json.loads((d/'scalers.json').read_text()))
        pool={'A':[],'B':list(range(54)),'C':list(range(54,150)),'D':list(range(150))}[r.group]
        names=cfg['data']['feature_archive']['names'];selected=sorted(pool,key=lambda j:(-scores[j],names[j]))[:r.k]
        assert selected==cfg['monitor']['selected_indices']
        if r.group!='A':
            parent=Path(cfg['monitor']['fallback_checkpoint']).parent
            pc=json.loads((parent/'provenance.json').read_text())['config']
            assert pc['experiment']['seed']==r.seed and pc['monitor']['selected_indices']==[]
            assert json.loads((parent/'split.json').read_text())==split
        if r.group=='D' and r.role=='final' and r.seed==0:composition.append(dict(fold=r.fold,k=r.k,time_features=sum(j<54 for j in selected),psd_features=sum(j>=54 for j in selected)))
        h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,501))
        assert h.filter(regex='^(train|validation|test_diagnostic)_').notna().all().all()
        met=json.loads((d/'metrics.json').read_text());assert met['selected_epoch']==500 and met['epochs']==500 and met['reload_verified'] and met['selected_checkpoint']=='last.pt'
        if r.role=='final':assert len(list((d/'epochs').glob('epoch_*.pt')))==500
        p=pd.read_csv(d/'epoch_predictions.csv.gz');assert np.isfinite(p[['target_vb_mm','predicted_vb_mm']]).all().all()
        for (s,t),q in p.groupby(['split','task']):
            assert t=='monitoring' and q.epoch.nunique()==500 and q.groupby('epoch').size().nunique()==1
            cases=split['outer_test_reserved'] if s=='test_diagnostic' else split[s]
            expected={(int(c),int(run)) for c,run,vb in zip(qt['case'],qt['run'],qt['VB']) if c in cases and np.isfinite(vb)}
            assert (q.origin_run==q.target_run).all() and (q.horizon==0).all()
            assert set(zip(q.case.astype(int),q.target_run.astype(int)))==expected
        s='validation' if r.role=='inner' else 'test_diagnostic';q=p[(p.epoch==500)&(p.split==s)]
        actual=abs(q.predicted_vb_mm-q.target_vb_mm).groupby(q.case).mean().mean();assert abs(actual-met['monitoring']['case_macro_mae'])<1e-6
        audits.append(dict(id=r.id,rows=len(p),seconds=met['wall_seconds']))
    audit=pd.DataFrame(audits);audit.to_csv(ROOT/'run_audit.csv',index=False)
    inner=pd.read_csv(ROOT/'inner_case_scores.csv');scores=inner.groupby(['fold','group','name','k']).mae.mean().reset_index()
    for (fold,name),q in inner.groupby(['fold','name']):assert set(q.holdout)==set(POOLS[fold])
    expected=scores.sort_values(['mae','k','name']).groupby(['fold','group'],as_index=False).first()
    chosen=pd.read_csv(ROOT/'selected_dimensions.csv');pd.testing.assert_frame_equal(expected,chosen,check_exact=False,atol=1e-12,rtol=1e-12)
    pred=pd.read_csv(ROOT/'test_predictions.csv');old=pd.read_csv(ROOT.with_name('history_quality_skip_20261008')/'phase1_test_predictions.csv')
    targets=old[['fold','case','run','vb_mm']].drop_duplicates()
    assert len(targets)==138
    for _,q in pred.groupby(['name','seed']):
        paired=q.merge(targets,on=['fold','case','run'],validate='one_to_one',suffixes=('','_reference'))
        assert len(paired)==len(q)==138;np.testing.assert_array_equal(paired.vb_mm,paired.vb_mm_reference)
    composition=pd.DataFrame(composition);composition.to_csv(ROOT/'joint_feature_composition.csv',index=False)
    summary=pd.read_csv(ROOT/'summary.csv');fixed=summary[~summary.name.str.startswith('selected_')].pivot(index='name',columns='subset',values='mae').reset_index()
    selected_summary=summary[summary.name.str.startswith('selected_')].pivot(index='name',columns='subset',values='mae').reset_index()
    fixed.to_csv(ROOT/'fixed_dimension_comparison.csv',index=False)
    same=[]
    for k in [6,12,18,36,54]:
        for subset in summary.subset.unique():
            s=summary[summary.subset==subset].set_index('name').mae
            same.append(dict(k=k,subset=subset,B=s[f'B_k{k:03d}'],C=s[f'C_k{k:03d}'],D=s[f'D_k{k:03d}'],C_minus_B=s[f'C_k{k:03d}']-s[f'B_k{k:03d}'],D_minus_B=s[f'D_k{k:03d}']-s[f'B_k{k:03d}']))
    same=pd.DataFrame(same);same.to_csv(ROOT/'same_dimension_comparison.csv',index=False)
    lines=['','## 完成审计与主要结论','',
        '432次运行均500epoch。内层验证在两个fold都为B选择18维、C选择12维、D选择18维。频域C和联合D总体MAE比B小约0.00138/0.00143mm，但均未超过A；不是稳定且显著的性能提升结论。',
        'C在fold B优于B，但fold A较差；D同样在fold B改善、fold A变差。范围外磨损时C/D均不如B。',
        '',md(selected_summary),'','### 固定维数完整结果','',md(fixed),
        '', '固定维数结果中最低测试分数不能重新定义最佳维数。应以预先约定的内层选择结果为主。','',
        '### 同维数比较（总体）','',md(same[same.subset=='overall']),
        '','### 联合组特征组成','',md(composition),
        '','### 审计范围','',
        f'检查{len(m)}次运行、216000个epoch、{int(audit.rows.sum())}行逐epoch预测；训练/验证/测试case-run目标、固定末轮、checkpoint重载标志、90000个最终模型逐epoch checkpoint、输入档案hash和键对齐、训练内排名/scaler、后备split/seed及内层维数选择均通过。所有配置各seed保留相同138个外层标签。',
        f'已完成运行累计训练{audit.seconds.sum()/60:.2f}分钟；此前进程中断及等待不包含在此累计时间内。',
        '已有两条异常sensor的排除及后备策略延续，无新增测试点删除。频域候选包含冗余绝对/相对能量，单变量排名不能排除冗余或捕捉所有非线性交互。',
        '本轮固定隐藏宽度使相同k的B/C/D参数量相同，更大k参数量增加；各组“全部特征”维数不同。seed重复不是独立case，不据微小平均差异宣称统计显著。',
        '复现完成审计：`python -m wear_monitoring_dynamics.scripts.audit_psd_sweep`。']
    path=ROOT/'RESULTS.md';body=path.read_text(encoding='utf-8').split('\n## 完成审计与主要结论')[0];path.write_text(body+'\n'.join(lines)+'\n',encoding='utf-8')
    result=dict(passed=True,runs=432,epochs=216000,prediction_rows=int(audit.rows.sum()),training_seconds=float(audit.seconds.sum()),test_labels=138)
    write_json(ROOT/'completion_audit.json',result);print(result);print(composition.to_string(index=False))

if __name__=='__main__':main()
