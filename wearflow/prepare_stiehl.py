"""Extract Stiehl features and generate the frozen 9-tool experiment manifests."""
import argparse,sys,importlib.util
from pathlib import Path
import pandas as pd
from wearflow_paths import ROOT

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data',type=Path,required=True,help='Dataset directory with filelist.csv and data/**/*.h5');a=ap.parse_args()
    data=a.data.resolve()
    if not (data/'filelist.csv').exists() or not (data/'data').is_dir():raise SystemExit('Expected filelist.csv and data/ under --data')
    folder=ROOT/'experiments/stiehl';sys.path.insert(0,str(folder))
    import prepare_features,data_protocol
    prepare_features.DATA=data
    prepare_features.main();data_protocol.freeze_splits()
    tasks=[]
    for test in range(1,10):
        for role in ['inner','final']:
            for seed in ([0] if role=='inner' else range(5)):
                for method in ['time','current','local','integral']:
                    tasks.append(dict(split=f'test{test}_{role}',method=method,seed=seed))
    pd.DataFrame(tasks).to_csv(folder/'study_manifest.csv',index=False)
    pd.DataFrame([dict(split=f'test{test}_{role}') for test in range(1,10) for role in ['inner','final']]).to_csv(folder/'gpr_manifest.csv',index=False)
    print('Prepared 18 tool-disjoint splits, 216 neural runs and 18 GPR fits. No training started.')
if __name__=='__main__':main()
