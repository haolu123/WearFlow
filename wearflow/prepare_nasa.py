"""Prepare train-only NASA inputs and (explicitly requested) parent training.

Reuses the exact feature, quality, scaling and training implementations. No test
selection; historical run directories are generated through new manifests.
"""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat
from wearflow_paths import ROOT
from wear_monitoring_dynamics.config import file_hash,write_json
from wear_monitoring_dynamics.data.features import SENSORS,STATISTICS
from wear_monitoring_dynamics.data.records import load_table
from wear_monitoring_dynamics.data.sensor_quality import raw_quality_metrics,fit_rules,assess,apply_quality
from wear_monitoring_dynamics.data.preprocessing import fit_scalers,transform,forecasting_pairs
from wear_monitoring_dynamics.data.feature_archive import attach_features
from wear_monitoring_dynamics.scripts.inspect_psd_features import features
from wear_monitoring_dynamics.scripts.run_psd_sweep import FREQ,rank

SUITE=ROOT/'experiments/nasa_baselines'
PARENTS=ROOT/'wear_monitoring_dynamics/outputs/current12_fusion_20261013'

def ingest(path):
    target=ROOT/'mill_numpy/mill.npy';target.parent.mkdir(exist_ok=True)
    if path.suffix.lower()=='.mat':
        mat_target=ROOT/'mill/mill.mat';mat_target.parent.mkdir(exist_ok=True)
        if path.resolve()!=mat_target.resolve():shutil.copy2(path,mat_target)
        raw=loadmat(path,squeeze_me=True,struct_as_record=False)['mill']
        keys=['case','run','VB','time','DOC','feed','material']+list(SENSORS)
        records=[{k:getattr(r,k) for k in keys} for r in raw]
        np.save(target,np.asarray(records,dtype=object),allow_pickle=True)
    else:
        if path.resolve()!=target.resolve():shutil.copy2(path,target)
    # NPY objects must come from the trusted NASA conversion, never untrusted pickles.
    return target,np.load(target,allow_pickle=True).tolist()

def prepare(source,device):
    path,raw=ingest(source)
    cache=ROOT/'wear_monitoring_dynamics/outputs/cache'
    table,signature=load_table(path,cache)
    metrics=raw_quality_metrics(raw)
    bykey={(int(r['case']),int(r['run'])):r for r in raw}
    names=[f'{s}_{k}' for s in SENSORS for k in STATISTICS]+[f'{s}_psd_{k}' for s in SENSORS for k in FREQ]
    # Features themselves are label independent. Invalidity is calibrated separately
    # within each split below; invalid rows never enter feature ranking or scaling.
    values=[]
    for i,(c,r) in enumerate(zip(table['case'],table['run'])):
        record=bykey[int(c),int(r)]
        channel_features=[features(record[s])[2] for s in SENSORS]
        psd=np.concatenate([np.array([f[k] for k in FREQ]) for f in channel_features])
        values.append(np.r_[table['sensor'][i],psd])
    values=np.asarray(values);assert values.shape==(166,150) and np.isfinite(values).all()
    arpath=ROOT/'wear_monitoring_dynamics/outputs/wearflow_prepared/feature_archive.npz'
    arpath.parent.mkdir(parents=True,exist_ok=True)
    if arpath.exists():
        old=np.load(arpath);np.testing.assert_array_equal(old['features'],values)
    else:np.savez_compressed(arpath,features=values,case=table['case'],run=table['run'],names=np.asarray(names))
    ar=dict(path=str(arpath),sha256=file_hash(arpath),names=names)
    table=attach_features(table,ar);tasks=[];entries=[]
    for template in sorted((ROOT/'configs/nasa').glob('fold_*.json')):
        ref=json.loads(template.read_text());cfg=ref['monitor'];split=ref['split'];exp=cfg['experiment']
        rules=fit_rules(metrics,split['train']);decision,_=assess(raw,metrics,rules)
        invalid=decision.loc[~decision.sensor_valid,['case','run']].astype(int).values.tolist()
        quality=dict(version='sensor_integrity_v1',rules=rules,invalid_records=invalid,fallback='frozen_condition_time',retain_previous_hard_exclusions=True)
        t=apply_quality(table,quality);scores,_=rank(t,split['train'])
        candidates=[i for i,n in enumerate(names) if n.startswith(('smcAC_','smcDC_'))]
        selected=sorted(candidates,key=lambda i:(-scores[i],names[i]))[:12]
        assert selected==cfg['monitor']['selected_indices'],'Input data does not reproduce the frozen feature ranking'
        cfg['data'].update(path=str(path),cache=str(cache),feature_archive=ar,sensor_quality=quality)
        cfg['data']['sensor_preprocessing']['indices']=selected
        cfg['monitor'].pop('fallback_checkpoint',None)
        cfg['training'].update(device=device,epochs=100,selection_epoch=100,fixed_final_epoch=True,record_all_epoch_splits=True,save_final_epoch_checkpoints=True)
        cfg.update(output_root=str(ROOT/'wear_monitoring_dynamics/outputs'),study_id='wearflow_parents')
        sc=fit_scalers(t,split['train'],cfg['data']['sensor_preprocessing']);z=transform(t,sc)
        # Preserve byte-equivalent valid inputs; invalid feature rows are ignored.
        for k in ['wear','condition','time_scale']:
            np.testing.assert_allclose(sc[k],ref['scalers'][k],atol=1e-10,rtol=1e-10)
        sensor_z=z['sensor'][:,selected].copy();sensor_z[~t['sensor_valid']]=0
        X=np.column_stack([sensor_z,z['condition'],z['time']])
        arrays=dict(X=X,X_ct=np.column_stack([z['condition'],z['time']]),sensor_raw=t['sensor'][:,selected],sensor_z=sensor_z,case=t['case'].astype(int),run=t['run'].astype(int),time=t['time'],condition_raw=np.column_stack([t[k] for k in ['DOC','feed','material']]),y_mm=t['VB'],y_z=z['target'].ravel(),sensor_valid=t['sensor_valid'])
        for role in ['train','validation','test']:
            arrays[role+'_mask']=np.isin(t['case'],split[role])&np.isfinite(t['VB'])
            arrays[role+'_fit_mask']=arrays[role+'_mask']&t['sensor_valid']
            arrays[role+'_pairs']=forecasting_pairs(t,split[role],[1,2,3])
        key=exp['fold']+('/final' if exp['role']=='final' else '/inner_'+str(exp['holdout']))
        d=SUITE/'data'/key;d.mkdir(parents=True,exist_ok=True)
        # Do not rewrite an identical NPZ: its byte hash is part of run identity.
        if (d/'arrays.npz').exists():
            old=np.load(d/'arrays.npz')
            for k,v in arrays.items():np.testing.assert_array_equal(old[k],v)
        else:np.savez_compressed(d/'arrays.npz',**arrays)
        info=dict(fold=exp['fold'],role=exp['role'],split=split,scalers=sc,selected_features=[names[i] for i in selected],selected_indices=selected,feature_archive=ar,data_signature=signature,quality=quality,counts={k:int(arrays[k+'_mask'].sum()) for k in ['train','validation','test']},audit=dict(training_only_ranking_recomputed=True,scalers_recomputed=True,quality_thresholds_refit=True))
        write_json(d/'metadata.json',info)
        entries.append(dict(key=key,directory=str(d),arrays_sha256=file_hash(d/'arrays.npz'),**info['counts']))
        for seed in ([0] if exp['role']=='inner' else range(5)):
            cc=copy.deepcopy(cfg);cc['experiment']['seed']=seed
            ct=copy.deepcopy(ref['ct']);ct['data'].update(path=str(path),cache=str(cache),feature_archive=ar,sensor_quality=quality)
            ct['training'].update(device=device,epochs=500,selection_epoch=500,fixed_final_epoch=True)
            ct['experiment']['seed']=seed;ct.update(output_root=cc['output_root'],study_id='wearflow_parents')
            tasks.append(dict(fold=exp['fold'],role=exp['role'],holdout=exp.get('holdout'),seed=seed,config=cc,ct_config=ct))
    write_json(SUITE/'data_manifest.json',dict(splits=entries,unique_outer_labels=138,candidate_count=50,selected_count=12))
    PARENTS.mkdir(parents=True,exist_ok=True);write_json(PARENTS/'portable_parent_plan.json',tasks)
    return tasks

def parents(tasks,mode):
    from wear_monitoring_dynamics.training.trainer import run_training
    ctrows=[];monrows=[];oderows=[]
    for task in tasks:
        base={k:task[k] for k in ['fold','role','holdout','seed']}
        ct=run_training(task['ct_config']);ctrows.append(dict(**base,run_dir=str(ct)))
        pd.DataFrame(ctrows).to_csv(PARENTS/'reused_ct_manifest.csv',index=False)
        if mode=='all':
            cfg=copy.deepcopy(task['config']);cfg['monitor']['fallback_checkpoint']=str(ct/'last.pt')
            monitor=run_training(cfg);monrows.append(dict(**base,type='monitor',run_dir=str(monitor)))
            pd.DataFrame(monrows).to_csv(PARENTS/'study_manifest.csv',index=False)
            if task['role']=='final':
                oc=copy.deepcopy(cfg);oc['dynamics'].update(pretrain_epochs=500,pretrain_seed=task['seed']);oc['training'].update(selection_epoch=500)
                ode=run_training(oc,pretrain=True);oderows.append(dict(**base,run_dir=str(ode)))
                pd.DataFrame(oderows).to_csv(PARENTS/'reused_ode_manifest.csv',index=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--train-parents',choices=['ct','all']);a=p.parse_args()
    import torch
    torch.set_num_threads(1)
    tasks=prepare(a.data.resolve(),a.device)
    print('Prepared 16 train-only splits. Parent runs planned:',len(tasks))
    if a.train_parents:parents(tasks,a.train_parents)
if __name__=='__main__':main()
