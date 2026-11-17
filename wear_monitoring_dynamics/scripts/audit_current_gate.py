"""Final gate values and leave-one-feature-out sensitivity; no retraining."""
import json
from pathlib import Path
import numpy as np,pandas as pd,torch
from ..config import project_path,digest,write_json
from ..data.records import load_table
from ..data.feature_archive import attach_features
from ..data.sensor_quality import apply_quality
from ..data.preprocessing import fit_scalers
from ..data.monitor_batch import make_monitor_batch
from ..models.registry import build_model
from .run_current_gate import ROOT

def main():
    torch.set_num_threads(1);m=pd.read_csv(ROOT/'study_manifest.csv');gates=[];ablations=[];weights=[];identities={};audit=[]
    parents=pd.read_csv(ROOT/'reused_sensor_manifest.csv')
    for r in m.itertuples():
        d=Path(r.run_dir);cfg=json.loads((d/'provenance.json').read_text())['config'];s=json.loads((d/'scalers.json').read_text());split=json.loads((d/'split.json').read_text())
        match=parents[(parents.fold==r.fold)&(parents.role==r.role)&(parents.seed==r.seed)]
        if r.role=='inner':match=match[match.holdout==r.holdout]
        assert len(match)==1;pcfg=json.loads((Path(match.iloc[0].run_dir)/'provenance.json').read_text())['config']
        assert cfg['data']==pcfg['data'] and cfg['training']==pcfg['training']
        for key in ['selected_indices','selected_features','fallback_checkpoint']:assert cfg['monitor'][key]==pcfg['monitor'][key]
        t,_=load_table(project_path(cfg['data']['path']),project_path(cfg['data']['cache']));t=apply_quality(attach_features(t,cfg['data']['feature_archive']),cfg['data']['sensor_quality'])
        assert digest(fit_scalers(t,split['train'],cfg['data']['sensor_preprocessing']))==digest(s)
        torch.manual_seed(r.seed);batch=make_monitor_batch(cfg,t,s,'cpu');model=build_model(cfg);k=(r.fold,r.role,str(r.holdout),r.seed)
        state={key:v.clone() for key,v in model.net.state_dict().items()}
        if k in identities:
            for key,v in state.items():torch.testing.assert_close(v,identities[k][key],rtol=0,atol=0)
        else:identities[k]=state
        model.load_state_dict(torch.load(d/'last.pt',map_location='cpu',weights_only=True)['model']);model.eval()
        total=sum(x.numel() for x in model.parameters());expected=3649+{'static':18,'condition':386,'time_condition':402}[r.kind];assert total==expected
        audit.append(dict(id=r.id,parameters=total))
        if r.role!='final':continue
        assert len(list((d/'epochs').glob('epoch_*.pt')))==500
        ids=np.arange(len(t['case']));names=cfg['monitor']['selected_features']
        with torch.no_grad():g=model.gates(batch,ids).numpy();pred=model.estimate(batch,ids).numpy().ravel()*s['wear'][1]+s['wear'][0]
        test=np.isin(t['case'],split['test'])&np.isfinite(t['VB']);original=pd.read_csv(d/'monitoring_predictions.csv');lookup={(int(c),int(run)):v for c,run,v in zip(t['case'],t['run'],pred)}
        np.testing.assert_allclose([lookup[(int(q.case),int(q.run))] for q in original.itertuples()],original.predicted_vb_mm,atol=1e-6,rtol=1e-5)
        w=model.net[0].weight.detach().numpy()[:,:18]
        for j,feature in enumerate(names):
            weights.append(dict(name=r.name,fold=r.fold,seed=r.seed,feature=feature,input_weight_norm=np.linalg.norm(w[:,j])))
            for i in ids:
                gates.append(dict(name=r.name,fold=r.fold,seed=r.seed,case=int(t['case'][i]),run=int(t['run'][i]),time=t['time'][i],DOC=t['DOC'][i],feed=t['feed'][i],material=t['material'][i],split='train' if t['case'][i] in split['train'] else 'test' if t['case'][i] in split['test'] else 'outside_study',sensor_valid=bool(t['sensor_valid'][i]),feature=feature,gate=g[i,j]))
            modified=dict(batch);modified['sensor']=batch['sensor'].clone();modified['sensor'][:,cfg['monitor']['selected_indices'][j]]=0
            with torch.no_grad():out=model.estimate(modified,ids).numpy().ravel()*s['wear'][1]+s['wear'][0]
            for case in split['test']:
                mask=test&(t['case']==case);base=abs(pred[mask]-t['VB'][mask]).mean();changed=abs(out[mask]-t['VB'][mask]).mean()
                ablations.append(dict(name=r.name,fold=r.fold,seed=r.seed,case=case,feature=feature,original_mae=base,zeroed_mae=changed,delta_mae=changed-base))
    pd.DataFrame(gates).to_csv(ROOT/'final_gate_values.csv.gz',index=False);pd.DataFrame(weights).to_csv(ROOT/'input_weight_norms.csv',index=False);pd.DataFrame(ablations).to_csv(ROOT/'feature_zeroing_sensitivity.csv',index=False);pd.DataFrame(audit).to_csv(ROOT/'gate_config_audit.csv',index=False)
    write_json(ROOT/'gate_audit.json',dict(passed=True,runs=len(m),final_runs=int((m.role=='final').sum()),note='Zeroing standardized sensor approximates closing feature gate; diagnostic only, no test-based selection.'))
    print('Gate diagnostics and config audit complete')

if __name__=='__main__':main()

