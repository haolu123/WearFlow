"""Fast checks of physical propagation, gradients, causality and target rules."""
import importlib.util,sys,unittest
import numpy as np
import pandas as pd
import torch
from wearflow_paths import ROOT
from wear_monitoring_dynamics.models.dynamics import WearDynamics
from wear_monitoring_dynamics.models.feature_mlp import FeatureMLP
from wear_monitoring_dynamics.data.preprocessing import forecasting_pairs

class CoreTests(unittest.TestCase):
    def test_rate_integration_and_gradient(self):
        torch.manual_seed(0);ode=WearDynamics(32,20,True).double()
        z=torch.tensor([[.2],[.3]],dtype=torch.float64,requires_grad=True)
        c=torch.zeros(2,4,dtype=torch.float64);dt=torch.tensor([[.1],[.4]],dtype=torch.float64)
        end=ode.propagate(z,c,dt)
        self.assertTrue(torch.all(end>=z));self.assertTrue(torch.equal(ode.propagate(z,c,dt*0),z))
        end.sum().backward();self.assertGreater(float(z.grad.abs().sum()),0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in ode.parameters()))

    def test_future_sensor_not_used(self):
        torch.manual_seed(1);m=FeatureMLP(dict(selected_indices=list(range(12)),hidden_dims=[64,32]));ode=WearDynamics()
        b=dict(sensor=torch.randn(3,12),condition=torch.zeros(3,4),time=torch.arange(3.).view(-1,1))
        i=torch.tensor([0]);before=ode.propagate(m.estimate(b,i),b['condition'][i],torch.ones(1,1))
        b['sensor'][1:]+=1000
        after=ode.propagate(m.estimate(b,i),b['condition'][i],torch.ones(1,1))
        torch.testing.assert_close(before,after,atol=0,rtol=0)

    def test_pairs_use_labeled_order_and_actual_time(self):
        t=dict(case=np.array([1,1,1,1]),VB=np.array([.1,np.nan,.2,.3]),time=np.array([0,1,7,19]),DOC=np.ones(4),feed=np.ones(4),material=np.ones(4))
        p=forecasting_pairs(t,[1],[1,2,3]);np.testing.assert_array_equal(p,[[0,2,1],[0,3,2],[2,3,1]])

    def test_stiehl_tolerance(self):
        path=ROOT/'experiments/stiehl/data_protocol.py';spec=importlib.util.spec_from_file_location('stiehl_data',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        frame=pd.DataFrame(dict(tool=[1,1,1],cumulated_tool_contact_time=[0,241,600]))
        p=mod.pairs(frame);self.assertTrue(((p.actual_seconds>=p.requested_seconds)&(p.actual_seconds<=p.requested_seconds+30)).all())
        self.assertEqual(len(p),1)

if __name__=='__main__':unittest.main()
