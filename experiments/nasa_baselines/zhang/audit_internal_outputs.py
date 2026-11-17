"""Independent read-only audit of internal study; writes only this agent's audit."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parent
STUDY=ROOT.parent/'internal/results'

def main():
    groups={};checked=[]
    for marker in sorted(STUDY.glob('*/*/seed_*/*/completed.json')):
        folder=marker.parent;summary=json.loads(marker.read_text())
        cp=torch.load(folder/'model.pt',weights_only=True)
        p=dict(np.load(folder/'epoch_predictions.npz'));h=pd.read_csv(folder/'history.csv')
        assert cp['epoch']==100 and h.epoch.tolist()==list(range(101))
        assert all(p[k].shape[0]==101 and np.isfinite(p[k]).all() for k in ['current_z','future_z','oracle_z'])
        assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in cp['identity']['sources'].items())
        assert all(torch.equal(cp['initial'][k],cp['final'][k]) for k in cp['initial'] if k.startswith('dynamics'))==(folder.name=='current_only')
        mask=p['train_mask'];obs=p['train_observations'];pairs=p['train_pairs_used']
        assert np.all(mask[obs]&p['sensor_valid'][obs]);assert np.all(mask[pairs[:,0]]&mask[pairs[:,1]]&p['sensor_valid'][pairs[:,0]])
        np.testing.assert_array_equal(p['case'][pairs[:,0]],p['case'][pairs[:,1]])
        assert np.all(p['time'][pairs[:,1]]>p['time'][pairs[:,0]])
        mu,sd=cp['scalers']['wear']
        for split in ['train','validation','test']:
            idx=p[split+'_mask'];fidx=idx[p['pairs'][:,0]]
            if idx.any():
                values=np.abs(p['current_z'][:,idx]*sd+mu-p['target_mm'][None,idx]).mean(axis=1)
                np.testing.assert_allclose(values,h[split+'_current_mae_mm'],atol=1e-7,rtol=1e-6)
            if fidx.any():
                values=np.abs(p['future_z'][:,fidx]*sd+mu-p['target_mm'][p['pairs'][fidx,1]][None,:]).mean(axis=1)
                np.testing.assert_allclose(values,h[split+'_future_mae_mm'],atol=1e-7,rtol=1e-6)
        key=str(folder.parent)
        if key in groups:
            base=groups[key]
            assert summary['initial_hash']==base['hash']
            for k in ['current_z','future_z','oracle_z']:np.testing.assert_array_equal(p[k][0],base[k])
        else:groups[key]={'hash':summary['initial_hash'],**{k:p[k][0] for k in ['current_z','future_z','oracle_z']}}
        checked.append(str(folder.relative_to(STUDY)))
    report={'status':'complete' if len(checked)==96 else 'partial','checked':len(checked),'runs':checked,'epoch0_groups':len(groups),'checks':['101 epoch snapshots','100 training epochs','training masks','pair causal dt','rate unchanged iff current-only','same epoch0 within groups','raw-derived metrics','source hashes']}
    (ROOT/'internal_outputs_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='runs'}))
if __name__=='__main__':main()
