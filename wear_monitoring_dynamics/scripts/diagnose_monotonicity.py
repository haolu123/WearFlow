"""Read-only checkpoint gradients and a fixed 32-run lambda diagnostic.

Training implementation is unchanged. Only training data enter gradients;
validation, never outer test, enters the diagnostic epoch selection.
"""
import argparse
import copy
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..config import PACKAGE, PROJECT, project_path, write_json, file_hash, digest
from ..data.records import load_table
from ..data.datasets import tensors, observations, pair_tensors
from ..models.registry import build_model
from ..losses.supervised import monitoring_loss
from ..losses.forecasting import forecasting_loss
from ..losses.monotonicity import monotonicity_loss
from ..evaluation.monotonicity import epoch_monotonicity
from ..training.reproducibility import setup
from ..training.trainer import run_training
from .run_monotonicity import verify_compatible_source

STUDY = 'monotonicity_strength_20260927'
OUTPUT = project_path('wear_monitoring_dynamics/outputs')
BASE = OUTPUT/'trajectory_monotonicity_20260923'
MODELS = ['time_sct_encoded', 'time_sct_attention']
# Chosen from conditions alone, before examining this diagnostic's scores.
# Both folds cover (DOC 1.5, feed .5, material 1) and (.75, .25, material 2).
HOLDOUTS = {'fold_a': [1, 7], 'fold_b': [9, 13]}
EPOCHS = [1, 10, 50, 100, 200, 500]


def gradient_vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for p, g in zip(parameters, gradients)]).detach().double()


def geometry(a, b):
    """a relative to b; undefined geometry remains NaN rather than fake zero."""
    na, nb = float(a.norm()), float(b.norm())
    return {'numerator_norm': na, 'denominator_norm': nb,
            'ratio': na/nb if nb else np.nan,
            'cosine': float(torch.dot(a, b)/(na*nb)) if na and nb else np.nan}


def checkpoint_gradients(path, table, signature, weight):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert state['data_signature'] == signature
    cfg, split = state['config'], state['split']
    device = setup(cfg['experiment']['seed'], 'cuda', True)
    model = build_model(cfg).to(device)
    model.load_state_dict(state['model'])
    model.train()
    batch = tensors(table, state['scalers'], device)
    obs = observations(table, split['train'])
    adjacent = pair_tensors(table, split['train'], [1], device)
    params = list(model.monitor.parameters())
    sup = monitoring_loss(model, batch, obs)
    gs = gradient_vector(sup, params)
    mono = monotonicity_loss(model, batch, adjacent)
    gm = gradient_vector(mono, params)
    go = torch.zeros_like(gs)
    forecast_value = 0.
    if model.dynamics is not None:
        forecast = forecasting_loss(model, batch, pair_tensors(table, split['train'], cfg['data']['horizons'], device))
        forecast_value = float(forecast.detach())
        go = gradient_vector(forecast, params)
    gb = cfg['loss']['monitoring_weight']*gs + cfg['loss']['forecasting_weight']*go
    with torch.no_grad():
        delta = (model.estimate(batch, adjacent[:, 0])-model.estimate(batch, adjacent[:, 1])).flatten()
        drops = delta.clamp(min=0)*state['scalers']['wear'][1]
    row = dict(epoch=state['epoch'], weight=weight, checkpoint=str(path),
               loss_supervised=float(sup.detach()), loss_mono=float(mono.detach()), loss_forecast=forecast_value,
               norm_supervised=float(gs.norm()), norm_mono=float(gm.norm()), norm_ode=float(go.norm()),
               train_pairs=len(adjacent), train_violation_rate=float((drops>1e-8).double().mean()),
               train_mean_drop_mm=float(drops.mean()), train_max_drop_mm=float(drops.max()))
    for name, left, right in [('mono_vs_sup', weight*gm, gs), ('mono_vs_base', weight*gm, gb),
                              ('mono_vs_ode', weight*gm, go), ('sup_vs_ode', gs, go)]:
        row.update({name+'_'+k: v for k, v in geometry(left, right).items()})
    # autograd.grad never populates .grad, and diagnostics never step an optimizer.
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v.cpu(), state['model'][k]) for k, v in model.state_dict().items())
    return row


def audit_sources(directories, signature):
    checked = set()
    for directory in directories:
        directory = Path(directory)
        p = json.loads((directory/'provenance.json').read_text())
        status = json.loads((directory/'status.json').read_text())
        assert status['status']=='completed' and status['run_id']==digest(p)[:12]
        assert p['data_signature']==signature
        assert (p['torch'],p['numpy'],p['python'],p['device_name']) == (str(torch.__version__),str(np.__version__),platform.python_version(),torch.cuda.get_device_name(0))
        if p['parent_hash']:
            assert file_hash(p['config']['dynamics']['checkpoint'])==p['parent_hash']
        source = p['source_hash']
        if source in checked:
            continue
        archive = OUTPUT/'sources'/f'{source}.zip'
        if p['config'].get('monotonicity',{}).get('weight',0):
            with zipfile.ZipFile(archive) as z:
                paths = [f for f in PACKAGE.rglob('*.py') if f.relative_to(PACKAGE).parts[0] not in ('scripts','tests','outputs')]
                paths.append(PROJECT/'release/CondWear-ODE/models/vb_sensor_condition_ode.py')
                for f in paths:
                    assert f.read_bytes()==z.read(f.relative_to(PROJECT).as_posix()), str(f)
        else:
            verify_compatible_source(archive)
        checked.add(source)
    return sorted(checked)


def prepare():
    manifest = pd.read_csv(BASE/'study_manifest.csv')
    rows, configs = [], {}
    for r in manifest[manifest.type=='monitor'].itertuples():
        cfg = yaml.safe_load((Path(r.run_dir)/'config_resolved.yaml').read_text(encoding='utf-8'))
        for name in MODELS:
            for ode in (False, True):
                prefix = name+('_forecast' if ode else '')
                if r.experiment not in (prefix, prefix+'_mono'):
                    continue
                weight = cfg.get('monotonicity',{}).get('weight',0.)
                if r.role=='inner' and cfg['experiment']['holdout'] in HOLDOUTS[r.fold]:
                    rows.append(dict(model=name, ode=ode, weight=weight, fold=r.fold,
                                     holdout=cfg['experiment']['holdout'], origin='reused',run_dir=r.run_dir))
                    if weight==.1:
                        for w in (1.,10.):
                            fresh = copy.deepcopy(cfg)
                            fresh['study_id']=STUDY
                            fresh['experiment']['name']=prefix+f'_mono_w{int(w)}'
                            fresh['monotonicity']['weight']=w
                            key=(name,ode,w,r.fold,cfg['experiment']['holdout'])
                            configs[key]=fresh
    assert len(rows)==32 and len(configs)==32
    return manifest, rows, configs


def report(root, rows):
    frames=[]; count_predictions=0
    for row in rows:
        directory=Path(row['run_dir'])
        h=pd.read_csv(directory/'history.csv')
        assert h.epoch.tolist()==list(range(1,501))
        assert np.isfinite(h.select_dtypes('number').to_numpy()).all()
        p=pd.read_csv(directory/'epoch_predictions.csv.gz',usecols=['epoch','target_vb_mm','predicted_vb_mm'])
        assert np.isfinite(p.to_numpy()).all() and p.epoch.nunique()==500
        assert p.groupby('epoch').size().nunique()==1
        count_predictions+=len(p)
        mono=epoch_monotonicity(directory/'epoch_predictions.csv.gz')
        for split in ('train','validation'):
            values=h[['epoch',split+'_monitoring_case_macro_mae']].rename(columns={split+'_monitoring_case_macro_mae':'mae'})
            values=values.merge(mono[mono.split==split],on='epoch',validate='one_to_one')
            assert len(values)==500
            for k,v in row.items(): values[k]=v
            frames.append(values)
    curves=pd.concat(frames,ignore_index=True)
    curves.to_csv(root/'epoch_diagnostics.csv.gz',index=False)
    selected=[]
    for key,g in curves.groupby(['model','ode','weight']):
        val=g[g.split=='validation'].groupby('epoch').mae.mean()
        epoch=int(val.idxmin())
        for which,e in [('validation_selected',epoch),('fixed_500',500)]:
            for split in ('train','validation'):
                q=g[(g.epoch==e)&(g.split==split)]
                assert len(q)==4
                selected.append(dict(model=key[0],ode=key[1],weight=key[2],selection=which,epoch=e,split=split,
                    mae=q.mae.mean(),violation_rate=q.violation_rate.mean(),mean_drop_mm=q.mean_drop_mm.mean(),
                    maximum_drop_mm=q.max_drop_mm.max()))
    pd.DataFrame(selected).to_csv(root/'comparison.csv',index=False)
    completion=dict(new_runs=32,reused_runs=32,recorded_epochs=32000,prediction_rows=count_predictions,
        all_checks_passed=True,new_training_seconds=sum(json.loads((Path(r['run_dir'])/'metrics.json').read_text())['wall_seconds'] for r in rows if r['origin']=='new'))
    write_json(root/'completion.json',completion)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (name,ode),g in curves.groupby(['model','ode']):
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for i,split in enumerate(('train','validation')):
            for j,metric in enumerate(('mae','violation_rate','mean_drop_mm')):
                for weight,q in g[g.split==split].groupby('weight'):
                    y=q.groupby('epoch')[metric].mean()
                    axes[i,j].plot(y.index,y.values,label=f'lambda={weight:g}')
                axes[i,j].set_title(split+' / '+metric); axes[i,j].set_xlabel('Epoch');axes[i,j].legend()
        fig.suptitle(name+(' + ODE' if ode else '')+' (four inner splits, seed 0)')
        fig.tight_layout(); fig.savefig(root/f'{name}_ode{int(ode)}.png',dpi=140);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    root=OUTPUT/STUDY;root.mkdir(exist_ok=True)
    manifest,rows,configs=prepare()
    write_json(root/'plan.json',dict(holdouts=HOLDOUTS,weights=[0,.1,1,10],seed=0,epochs=500,new_runs=32,
        rationale='Condition-only choice: DOC/feed/material (1.5,.5,1) and (.75,.25,2) in each fold; no outcome-based selection.',
        gradient_epochs=EPOCHS,configs=list(configs.values())))
    if not args.execute:
        print('Plan saved: 32 new + 32 reused inner runs.');return
    write_json(root/'status.json',{'status':'running'})
    example=next(iter(configs.values()))
    table,signature=load_table(project_path(example['data']['path']),project_path(example['data']['cache']),example['data']['raw_sensor_abs_limit'])
    setup(0,'cuda',True)
    finals=manifest[(manifest.role=='final')&(manifest.seed==0)&manifest.experiment.isin([n+s for n in MODELS for s in ('_mono','_forecast_mono')])]
    assert len(finals)==8
    sources=audit_sources([r['run_dir'] for r in rows]+finals.run_dir.tolist(),signature)
    write_json(root/'reuse_audit.json',dict(source_hashes=sources,data_signature=signature,execution_modules_compatible=True))
    gradients=[]
    for r in finals.itertuples():
        directory=Path(r.run_dir);m=json.loads((directory/'metrics.json').read_text())
        for epoch in sorted(set(EPOCHS+[m['selected_epoch']])):
            p=directory/'epochs'/f'epoch_{epoch:04d}.pt'
            gradients.append(dict(experiment=r.experiment,fold=r.fold,phase='existing_final',**checkpoint_gradients(p,table,signature,.1)))
    pd.DataFrame(gradients).to_csv(root/'existing_checkpoint_gradients.csv',index=False)
    print('Existing checkpoint gradient diagnostics complete.',flush=True)
    pd.DataFrame(rows).to_csv(root/'study_manifest.csv',index=False)
    for key,cfg in configs.items():
        directory=run_training(cfg)
        rows.append(dict(model=key[0],ode=key[1],weight=key[2],fold=key[3],holdout=key[4],origin='new',run_dir=str(directory)))
        pd.DataFrame(rows).to_csv(root/'study_manifest.csv',index=False)
    # Inner runs retain best and last checkpoints. No change to training/checkpoint policy.
    inner_gradients=[]
    for r in rows:
        for filename in ('best.pt','last.pt'):
            inner_gradients.append(dict(**r,checkpoint_kind=filename,**{k:v for k,v in checkpoint_gradients(Path(r['run_dir'])/filename,table,signature,r['weight']).items() if k!='weight'}))
    pd.DataFrame(inner_gradients).to_csv(root/'inner_checkpoint_gradients.csv',index=False)
    report(root,rows)
    write_json(root/'status.json',{'status':'completed','audit_passed':True})
    print('COMPLETE: 32 new trainings, 32 reused, gradients and curves audited.',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write_json(OUTPUT/STUDY/'status.json',{'status':'failed','error':str(error)})
        raise
