"""Mechanism controls: parameter matching, function-preserving initialization and gradients."""
import unittest
import numpy as np
import torch
from ..models.feature_mlp import FeatureMLP
from ..scripts.run_sensor_mechanisms import make_net, warm_start, fit


class MechanismTests(unittest.TestCase):
    def test_legacy_forward_and_update(self):
        torch.manual_seed(7)
        old=FeatureMLP(dict(selected_indices=list(range(18)),hidden_dims=[64,32]))
        new=make_net(True,[64,32],7)
        batch={'sensor':torch.randn(9,18),'condition':torch.randn(9,4),'time':torch.randn(9,1)}
        x=torch.cat([batch['sensor'],batch['condition'],batch['time']],dim=1)
        y=torch.randn(9,1)
        self.assertTrue(torch.equal(old.estimate(batch,np.arange(9)),new(x)))
        opt=torch.optim.AdamW(old.parameters(),lr=.001,weight_decay=.0001)
        opt.zero_grad();(old.estimate(batch,np.arange(9))-y).square().mean().backward();opt.step()
        fit(new,x,y,np.arange(9),1)
        for a,b in zip(old.parameters(),new.parameters()):self.assertTrue(torch.equal(a,b))

    def test_warm_identity_and_sensor_gradient(self):
        base=make_net(False,[64,32],0);model=warm_start(base,True,0)
        x=torch.randn(13,23)
        torch.testing.assert_close(model(x),base(x[:,-5:]),atol=1e-7,rtol=1e-6)
        model(x).square().mean().backward()
        self.assertGreater(model.net[0].weight.grad[:,:18].abs().sum().item(),0)
        self.assertIsNone(base.net[0].weight.grad)

    def test_capacity_match(self):
        self.assertEqual(sum(p.numel() for p in make_net(False,[64,32],0).parameters()),2497)
        self.assertEqual(sum(p.numel() for p in make_net(True,[50,25],0).parameters()),2501)

    def test_excluded_targets_cannot_change_fit(self):
        x=torch.randn(10,5);y=torch.randn(10,1);changed=y.clone();changed[6:]+=1000
        a=make_net(False,[32,16],3);b=make_net(False,[32,16],3)
        fit(a,x,y,np.arange(6),3);fit(b,x,changed,np.arange(6),3)
        for p,q in zip(a.parameters(),b.parameters()):self.assertTrue(torch.equal(p,q))


if __name__=='__main__':unittest.main()
