import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from wear_monitoring_dynamics.config import PACKAGE, PROJECT, load_config
from wear_monitoring_dynamics.losses.monotonicity import monotonicity_loss
from wear_monitoring_dynamics.evaluation.monotonicity import per_case_monotonicity, epoch_monotonicity
from wear_monitoring_dynamics.scripts.run_monotonicity import specification, verify_compatible_source
from wear_monitoring_dynamics.scripts.run_fixed_conditioning import MODELS
from wear_monitoring_dynamics.study import build_tasks
from wear_monitoring_dynamics.training.trainer import run_training


class ToyMonitor(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values=torch.nn.Parameter(torch.tensor(values,dtype=torch.double).reshape(-1,1))

    def estimate(self,batch,indices):
        return self.values[indices]


class MonotonicityTests(unittest.TestCase):
    def test_loss_direction_gradient_and_label_independence(self):
        model=ToyMonitor([1.,0.,2.])
        pairs=torch.tensor([[0,1,1],[1,2,1]])
        loss=monotonicity_loss(model,{'target':torch.randn(3,1)},pairs)
        self.assertEqual(loss.item(),.5)
        loss.backward()
        torch.testing.assert_close(model.values.grad,torch.tensor([[1.],[-1.],[0.]],dtype=torch.double))
        self.assertEqual(monotonicity_loss(model,{'target':torch.full((3,1),99.)},pairs).item(),.5)
        self.assertEqual(monotonicity_loss(model,{},pairs,1.).item(),0.)
        self.assertEqual(monotonicity_loss(ToyMonitor([0.,0.,1.]),{},pairs).item(),0.)

    def test_diagnostics_do_not_cross_cases_or_epochs(self):
        frame=pd.DataFrame({'case':[1,1,1,2,2],'run':[1,2,3,1,2],'predicted_vb_mm':[.2,.1,.3,0.,.2]})
        result=per_case_monotonicity(frame)
        self.assertEqual(result.pairs.sum(),3)
        self.assertEqual(result.decreases.sum(),1)
        self.assertAlmostEqual(result.mean_drop_mm.iloc[0],.05)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'predictions.csv.gz'
            first=frame.rename(columns={'run':'origin_run'}).assign(epoch=1,split='train',task='monitoring')
            second=first.assign(epoch=2,predicted_vb_mm=[.1,.2,.3,0.,.2])
            pd.concat([first,second]).to_csv(path,index=False)
            stats=epoch_monotonicity(path)
            self.assertEqual(stats.pairs.tolist(),[3,3])
            self.assertEqual(stats.decreases.tolist(),[1,0])

    def test_full_matrix_and_training_only_pairs(self):
        from wear_monitoring_dynamics.data.preprocessing import forecasting_pairs
        tasks=build_tasks(load_config(PACKAGE/'configs/time_conditioning.yaml'),specification())
        self.assertEqual(len(tasks),496)
        self.assertEqual(sum(bool(t['config'].get('monotonicity',{}).get('weight')) for t in tasks),240)
        table=dict(case=np.array([1,1,1,2,2]),time=np.array([0,1,2,0,1]),VB=np.array([.2,np.nan,.1,0.,.1]),DOC=np.ones(5),feed=np.ones(5),material=np.ones(5))
        self.assertEqual(forecasting_pairs(table,[1],[1]).tolist(),[[0,2,1]])

    def test_zero_weight_matches_archived_training_exactly(self):
        from wear_monitoring_dynamics.data.features import SENSORS
        source_hash='cf1fde46f1614cc4b91baa3fd58eb709ae8633fe93d4dc946049bf55e8232211'
        archive=PACKAGE/'outputs/sources'/f'{source_hash}.zip'
        verify_compatible_source(archive)
        with zipfile.ZipFile(archive) as z:
            code=z.read('wear_monitoring_dynamics/training/trainer.py').decode('utf-8')
        namespace={'__name__':'wear_monitoring_dynamics.training.reference_trainer','__package__':'wear_monitoring_dynamics.training'}
        # Reviewed local archived trainer, isolated namespace; never execute external notebooks.
        exec(compile(code,str(archive), 'exec'),namespace)
        records=[]
        for case in [1,2,3,4,5,7,8,9,10,11,12,13,14,16]:
            for run in range(1,6):
                item=dict(case=case,run=run,time=float(run*2),VB=.02*run,DOC=.75*(1+case%2),feed=.25,material=1+case%2)
                item.update({name:np.arange(12,dtype=float)*.01+run*.001 for name in SENSORS})
                records.append(item)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            np.save(root/'data.npy',np.array(records,dtype=object))
            for name,_,architecture,inputs in MODELS:
                for forecast in (False,True):
                    cfg=load_config()
                    cfg['data'].update(path=str(root/'data.npy'),cache=str(root/'cache'))
                    cfg['output_root']=str(root/'output')
                    cfg['monitor'].update(architecture=architecture,inputs=inputs)
                    cfg['training'].update(device='cpu',epochs=2,warmup_epochs=1,record_all_epoch_splits=True)
                    cfg['experiment'].update(name=name,role='final',holdout=None)
                    cfg['dynamics']['mode']='joint_scratch' if forecast else 'none'
                    cfg['loss']['forecasting_weight']=float(forecast)
                    cfg['study_id']=f'reference_{name}_{forecast}'
                    old=namespace['run_training'](cfg)
                    cfg['study_id']=f'current_{name}_{forecast}'
                    cfg['monotonicity']={'weight':0.,'tolerance_z':0.}
                    new=run_training(cfg)
                    a=torch.load(old/'last.pt',weights_only=True)['model']
                    b=torch.load(new/'last.pt',weights_only=True)['model']
                    self.assertEqual(a.keys(),b.keys())
                    for key in a:
                        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                    pd.testing.assert_frame_equal(pd.read_csv(old/'epoch_predictions.csv.gz'),pd.read_csv(new/'epoch_predictions.csv.gz'),check_exact=True)
