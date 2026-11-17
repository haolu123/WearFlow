"""Audit and report ABCD fixed-protocol adaptations; never trains."""

import json,time

from pathlib import Path

import numpy as np

import pandas as pd

import torch

from ..config import file_hash,write_json

from ..training.adaptation_from100 import prepare

from ..training.recursive_adaptation import layout,record

from ..data.preprocessing import fit_scalers

from .run_psd_sweep import rank

from .report_current12_fusion import metrics,md

from .audit_current12_recursive_joint import Collector,COLS



LABELS={'A':'A 冻结基线','B':'B 继续当前监督','C':'C 冻结ODE适配monitor','D':'D 联合微调'}



def report(root):

    start=time.perf_counter();torch.set_num_threads(1);m=pd.read_csv(root/'study_manifest.csv');assert len(m)==30

    records=[];audit=[];initials={};curves=[];oracles=[];seen=set()

    for r in m.itertuples():

        d=Path(r.run_dir);identity=json.loads((d/'identity.json').read_text());task=identity['task']

        assert identity['monitor100_hash']==file_hash(task['monitor100']) and identity['ode_hash']==file_hash(Path(task['ode_dir'])/'last.pt')

        assert identity['execution_hash']==file_hash('wear_monitoring_dynamics/training/adaptation_from100.py')

        met=json.loads((d/'metrics.json').read_text());grad=json.loads((d/'gradient_audit.json').read_text())

        assert met['epochs']==100 and met['reload_verified'] and met['changes']['monitor']>0

        assert met['changes']['dynamics']>0 if r.group=='D' else met['changes']['dynamics']==0

        assert grad['future_to_monitor']>0 and grad['no_future_sensor']

        assert grad['future_to_ode']>0 if r.group=='D' else grad['future_to_ode']==0

        h=pd.read_csv(d/'history.csv');assert h.epoch.tolist()==list(range(1,101))

        rowcount=0

        for filename,protocol in [('epoch_predictions.csv.gz','single_shot'),('fusion_epoch_predictions.csv.gz','fusion')]:

            q=pd.read_csv(d/filename);assert np.isfinite(q.predicted_vb_mm).all()

            count=q.groupby('epoch').size();assert count.index.tolist()==list(range(101)) and count.nunique()==1

            keycols=[k for k in q if k not in ['epoch','predicted_vb_mm']]

            hashes=pd.util.hash_pandas_object(q[keycols],index=False).to_numpy().reshape(101,-1)

            assert len(np.unique(hashes[0]))==len(hashes[0]) and np.all(hashes==hashes[0]);rowcount+=len(q)

            zero=q[q.epoch==0].reset_index(drop=True);key=(r.fold,r.seed,protocol)

            if key not in initials:

                initials[key]=zero;records.append(zero.assign(group='A',fold=r.fold,seed=r.seed,protocol=protocol))

            else:pd.testing.assert_frame_equal(initials[key],zero,check_exact=True)

            records.append(q[q.epoch==100].assign(group=r.group,fold=r.fold,seed=r.seed,protocol=protocol))

            error=q.assign(ae=abs(q.predicted_vb_mm-q.target_vb_mm))

            cm=error.groupby(['epoch','split','task','case']).ae.mean().groupby(['epoch','split','task']).mean().reset_index()

            for (split,typ),g in cm[cm.epoch>0].groupby(['split','task']):

                name=f'{split}_{typ}_case_macro_mae' if protocol=='single_shot' else f'fusion_{split}_a{r.alpha:g}_m1_{typ}_mae'

                np.testing.assert_allclose(g.sort_values('epoch').ae,h[name],atol=1e-8,rtol=1e-6)

            curves.append(cm.assign(group=r.group,fold=r.fold,seed=r.seed,protocol=protocol))

        model,b,t,parent=prepare(task);sc=parent['scalers'];sp=parent['split'];cfg=parent['config']

        assert not set(sp['train'])&set(sp['test'])

        if r.fold not in seen:

            assert json.loads(json.dumps(fit_scalers(t,sp['train'],cfg['data']['sensor_preprocessing'])))==json.loads(json.dumps(sc))

            scores,_=rank(t,sp['train']);names=cfg['data']['feature_archive']['names'];order=sorted([j for j,n in enumerate(names) if n.startswith(('smcAC_','smcDC_'))],key=lambda j:(-scores[j],names[j]))

            assert cfg['monitor']['selected_indices']==order[:12];seen.add(r.fold)

        fallback=torch.load(cfg['monitor']['fallback_checkpoint'],map_location='cpu',weights_only=True)

        assert fallback['split']==sp and fallback['config']['experiment']['seed']==r.seed

        state=torch.load(d/'last.pt',map_location='cpu',weights_only=True);assert state['split']==sp and state['scalers']==sc

        for prefix in ['monitor','dynamics']:

            actual=max(float(abs(v-state['model'][k]).max()) for k,v in model.state_dict().items() if k.startswith(prefix+'.'))

            assert abs(actual-met['changes'][prefix])<1e-8

        structures={name:layout(t,cases,'cpu') for name,cases in [('train',sp['train']),('test_diagnostic',sp['test'])]}

        for group,epoch in ([('A',0),(r.group,100)] if r.group=='B' else [(r.group,100)]):

            if epoch:model.load_state_dict(state['model'])

            model.eval();collector=Collector();record(model,b,t,sc,structures,[(r.alpha,True)],epoch,collector)

            expected=pd.DataFrame(collector.rows,columns=COLS)

            logged=pd.read_csv(d/'fusion_epoch_predictions.csv.gz');logged=logged[logged.epoch==epoch]

            np.testing.assert_allclose(expected.predicted_vb_mm,logged.predicted_vb_mm,atol=3e-6,rtol=1e-5)

            st=structures['test_diagnostic'];a,z=st['pairs'][:,0],st['pairs'][:,1]

            with torch.no_grad():y=model.dynamics.propagate(b['target'][a],b['condition'][a],b['time'][z]-b['time'][a]).numpy().ravel()*sc['wear'][1]+sc['wear'][0]

            oracles.append(pd.DataFrame(dict(case=t['case'][a],horizon=st['pairs'][:,2],target_vb_mm=t['VB'][z],predicted_vb_mm=y)).assign(group=group,fold=r.fold,seed=r.seed))

        audit.append(dict(id=r.id,epochs=100,rows=rowcount,seconds=met['seconds'],monitor_change=met['changes']['monitor'],ode_change=met['changes']['dynamics']))

    p=pd.concat(records,ignore_index=True);p.to_csv(root/'predictions.csv.gz',index=False);external=p[p.split=='test_diagnostic']

    for (protocol,typ,seed),g in external.groupby(['protocol','task','seed']):

        sets=[set(map(tuple,x[['case','origin_run','target_run','horizon']].to_numpy())) for _,x in g.groupby('group')];assert len(sets)==4 and all(x==sets[0] for x in sets)

    for _,g in external[(external.protocol=='fusion')&(external.task=='current')].groupby(['group','seed']):assert len(g)==128

    # Frozen A must reproduce the previously evaluated monitor100 + ODE500 baseline.

    old=pd.read_csv(root.parent/'model_learning_curves_20261018/epoch_case_metrics.csv.gz');old=old[(old.method=='sensor_fusion')&(old.epoch==100)&(old.partition=='test')]

    a=external[(external.group=='A')&(external.protocol=='fusion')&(external.task=='current')].assign(ae=lambda q:abs(q.predicted_vb_mm-q.target_vb_mm)).groupby(['fold','seed','case']).ae.mean().reset_index()

    cmp=a.merge(old,on=['fold','seed','case'],validate='one_to_one');assert len(cmp)==70;np.testing.assert_allclose(cmp.ae,cmp.mae,atol=3e-6,rtol=1e-5)

    summaries=[];percases=[]

    for (protocol,typ),g in external.groupby(['protocol','task']):

        keys=['group']+(['horizon'] if typ in ['future','forecasting'] else [])

        per,s=metrics(g,keys);summaries.append(s.assign(protocol=protocol,task=typ));percases.append(per.assign(protocol=protocol,task=typ))

    sm=pd.concat(summaries);pc=pd.concat(percases);sm.to_csv(root/'summary.csv',index=False);pc.to_csv(root/'case_seed_metrics.csv',index=False)

    oo=pd.concat(oracles);_,os=metrics(oo,['group','horizon']);os.to_csv(root/'oracle_summary.csv',index=False)

    boot=[];rng=np.random.default_rng(20261019)

    for (protocol,typ),g in pc.groupby(['protocol','task']):

        for horizon,x in (g.groupby('horizon') if typ in ['future','forecasting'] else [(0,g)]):

            pivot=x.groupby(['fold','case','group']).mae.mean().unstack('group')

            for method,reference in [('B','A'),('C','A'),('D','A'),('C','B'),('D','B'),('D','C')]:

                delta=pivot[method]-pivot[reference]

                draws=sum(rng.choice(v.to_numpy(),(5000,len(v)),replace=True).sum(1) for _,v in delta.groupby(level=0))/len(delta)

                lo,hi=np.quantile(draws,[.025,.975]);boot.append(dict(protocol=protocol,task=typ,horizon=horizon,group=method,reference=reference,delta_mae=delta.mean(),ci_low=lo,ci_high=hi,cases=len(delta)))

    bs=pd.DataFrame(boot);bs.to_csv(root/'case_bootstrap.csv',index=False)

    cc=pd.concat(curves);cc.to_csv(root/'epoch_curves.csv.gz',index=False);plot(root,cc)

    pd.DataFrame(audit).to_csv(root/'run_audit.csv',index=False)

    now=sm[(sm.protocol=='fusion')&(sm.task=='current')];ft=sm[(sm.protocol=='fusion')&(sm.task=='future')]

    text='# monitor100＋ODE500共同起点的ABCD实验\n\n'+md(now[['group','subset','mae','bias']])+'\n\n## 融合状态出发的未来VB\n\n'+md(ft[['group','horizon','subset','mae','bias']])+'\n\n## Case配对区间\n\n'+md(bs[bs.protocol=='fusion'])+'\n\n## 实测初值oracle传播\n\n'+md(os[os.subset=='overall'])+'''



A=冻结基线；B=仅当前VB监督继续训练monitor；C=当前＋未来VB监督、冻结ODE适配monitor；D=当前＋未来VB监督、同时更新monitor和ODE。B/C/D均从同一monitor100+独立ODE500出发，再训练100轮，AdamW lr1e-4、weight_decay1e-4，C/D未来损失权重1。当前和未来训练MSE分别按记录/配对等权，沿用上一批单次适配协议；报告MAE则按case等权。没有把新实验混入最新递推训练的三项case等权损失。



训练是单次传播，没有把融合递推放入训练图；融合是固定部署评估。alpha沿用原内层选择：A折1、B折.25，均单调截断；没有根据本批测试重选参数。原alpha是500轮模型下选择的，因此不是100轮协议重新校准的最优权重。当前融合排除初始化点，每seed128标签；未来h1/2/3排除初始化起点，使用实际dt，不读取未来sensor。raw monitor138标签和单次未来另列在summary.csv，不能与融合128标签混比。



本批30次新训练；A由共同epoch0复用，无新训练。固定100轮且不搜索损失权重，所以不新增内层训练/验证选参；沿用原case划分、训练内12维排名、clip/scaler、质量规则与同split/seed冻结CT后备。当前无效sensor跳过校正。5seed先平均，再按fold内case成对bootstrap5000次；区间未作多重比较校正，不把seed当独立case。所有结果属已查看过测试的探索性研究。



完整每轮训练/测试当前与未来原始记录：各run/epoch_predictions.csv.gz（单次），fusion_epoch_predictions.csv.gz（融合）；history.csv含实际训练loss及评估MAE；epoch0和末轮重载核查，B/C的ODE参数零变化，D非零变化，未来梯度、因果检查见gradient_audit.json。各run保存每轮checkpoint，不重写父模型。



![训练曲线](epoch_curves.png)

'''

    if (root/'CONCLUSIONS.md').exists():

        title,body=text.split('\n\n',1)

        text=title+'\n\n'+(root/'CONCLUSIONS.md').read_text(encoding='utf-8')+'\n'+body

    text+=f'\n训练合计{sum(x["seconds"] for x in audit)/60:.2f}分钟；审计{sum(x["rows"] for x in audit):,}行逐轮预测。详见audit.json与run_audit.csv。\n'

    (root/'RESULTS.md').write_text(text,encoding='utf-8')

    write_json(root/'audit.json',dict(status='passed',runs=30,epochs=3000,prediction_rows=sum(x['rows'] for x in audit),common_epoch0=True,baseline_reproduced=True))

    write_json(root/'status.json',dict(status='completed',runs=30,training_seconds=sum(x['seconds'] for x in audit),report_seconds=time.perf_counter()-start))

    print(now[now.subset=='overall'].to_string(index=False),flush=True)



def plot(root,cc):

    import matplotlib

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False

    fig,axs=plt.subplots(2,3,figsize=(15,8));colors={'B':'#2474b5','C':'#bd7a15','D':'#a73158'}

    for col,subset in enumerate(['overall','fold_a','fold_b']):

        for row,typ in enumerate(['current','future']):

            ax=axs[row,col];q=cc[(cc.protocol=='fusion')&(cc.task==typ)]

            if subset!='overall':q=q[q.fold==subset]

            for (group,split),g in q.groupby(['group','split']):

                y=g.groupby('epoch').ae.mean();ax.plot(y.index,y,label=LABELS[group]+('/训练' if split=='train' else '/测试'),color=colors[group],ls='--' if split=='train' else '-')

            base=q[(q.group=='B')&(q.epoch==0)&(q.split=='test_diagnostic')].ae.mean();ax.axhline(base,color='black',ls=':',label='A冻结基线/测试')

            ax.set(title=f'{subset} / {typ}',xlabel='额外训练epoch',ylabel='Case MAE (mm)');ax.grid(alpha=.2);ax.legend(fontsize=7)

    fig.suptitle('共同起点monitor100＋ODE500；融合权重及截断固定；未来曲线为case内所有horizon配对平均')

    fig.tight_layout();fig.savefig(root/'epoch_curves.png',dpi=150);plt.close(fig)

