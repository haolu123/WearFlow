import unittest
import numpy as np
import torch
from ..models.feature_mlp import FeatureMLP
from ..scripts.run_psd_sweep import rank

class FeatureSweepTests(unittest.TestCase):
    def test_ranking_ignores_heldout_and_invalid(self):
        t=dict(case=np.array([1]*4+[2]*4),VB=np.arange(8,dtype=float),sensor=np.column_stack([np.arange(8),np.arange(8)[::-1]]).astype(float),sensor_valid=np.array([True]*3+[False]+[True]*4))
        first,_=rank(t,[1]);t['VB'][3:]=999;t['sensor'][3:]=1e8
        second,_=rank(t,[1]);np.testing.assert_array_equal(first,second)

    def test_only_selected_current_features_used(self):
        model=FeatureMLP(dict(selected_indices=[1,4],hidden_dims=[64,32]))
        batch=dict(sensor=torch.randn(3,6),condition=torch.randn(3,4),time=torch.randn(3,1))
        before=model.estimate(batch,[0,1,2]);batch['sensor'][:,[0,2,3,5]]=999
        torch.testing.assert_close(before,model.estimate(batch,[0,1,2]),rtol=0,atol=0)
        batch['sensor_valid']=torch.tensor([True,False,True]);batch['fallback_vb_z']=torch.ones(3,1)*.3
        self.assertAlmostEqual(float(model.estimate(batch,[1])),.3,places=6)

if __name__=='__main__':unittest.main()
