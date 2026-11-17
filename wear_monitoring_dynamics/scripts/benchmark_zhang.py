"""Short real-data smoke runs for all transferred objectives; not formal results."""
import json
import time
import pandas as pd
import yaml
from ..config import PACKAGE, load_config, project_path
from ..study import variants
from ..training.trainer import run_training


def main():
    base = load_config(PACKAGE/'configs/zhang_transfer.yaml')
    base['training']['epochs'] = 20
    base['dynamics']['pretrain_epochs'] = 5
    base['study_id'] = 'zhang_smoke_20260922'
    base['experiment'].update(fold='fold_a', role='inner', holdout=8)
    pretrain = run_training(base, pretrain=True)
    study = yaml.safe_load((PACKAGE/'configs/studies/zhang_transfer_20260922.yaml').read_text())
    study['study_id'] = base['study_id']
    rows = []
    for cfg in variants(base, study):
        if cfg['dynamics']['mode'] == 'frozen':
            cfg['dynamics']['checkpoint'] = str(pretrain/'last.pt')
        directory = run_training(cfg)
        m = json.loads((directory/'metrics.json').read_text())
        rows.append({'experiment':cfg['experiment']['name'], 'run_dir':str(directory),
                     'seconds': m['wall_seconds'], 'steady_epoch_seconds':m['steady_epoch_seconds'],
                     'peak_cuda_memory_mb':m['peak_cuda_memory_mb']})
        pd.DataFrame(rows).to_csv(project_path(cfg['output_root'])/cfg['study_id']/'timing.csv', index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
