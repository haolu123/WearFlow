"""Key-aligned, content-verified run feature archives."""
import numpy as np
from ..config import project_path, file_hash

def attach_features(table, config):
    path=project_path(config['path'])
    if file_hash(path)!=config['sha256']:raise ValueError('Feature archive changed')
    with np.load(path,allow_pickle=False) as a:
        keys=list(zip(a['case'].astype(int),a['run'].astype(int)))
        if len(set(keys))!=len(keys):raise ValueError('Duplicate feature records')
        lookup={key:i for i,key in enumerate(keys)}
        order=[lookup[(int(c),int(r))] for c,r in zip(table['case'],table['run'])]
        values=a['features'][order]
        if not np.isfinite(values).all():raise ValueError('Nonfinite feature archive')
        if a['names'].tolist()!=config['names']:raise ValueError('Feature schema changed')
    result=dict(table);result['sensor']=values
    return result
