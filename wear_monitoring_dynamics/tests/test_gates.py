import unittest,torch
from wear_monitoring_dynamics.models.gated_features import GatedFeatureMLP
from wear_monitoring_dynamics.models.feature_mlp import FeatureMLP

class GateTests(unittest.TestCase):
    def make(self,kind):
        cfg=dict(selected_indices=[0,2,4],hidden_dims=[8,4],gate=dict(kind=kind))
        torch.manual_seed(12);m=GatedFeatureMLP(cfg)
        b=dict(sensor=torch.randn(5,6),condition=torch.randn(5,4),time=torch.arange(5.).reshape(-1,1))
        return cfg,m,b
    def test_initialization(self):
        for k in ['static','condition','time_condition']:
            cfg,m,b=self.make(k);torch.manual_seed(12);base=FeatureMLP(cfg)
            for a,c in zip(m.net.parameters(),base.net.parameters()):torch.testing.assert_close(a,c)
            torch.testing.assert_close(m.gates(b,[0,1]),torch.full((2,3),.5))
    def test_gate_inputs(self):
        for k in ['static','condition','time_condition']:
            _,m,b=self.make(k)
            with torch.no_grad():
                if k!='static':m.gate_net[-1].weight.fill_(.2)
            other=dict(b,sensor=b['sensor']*100,target=torch.randn(5,1))
            torch.testing.assert_close(m.gates(b,[0,1]),m.gates(other,[0,1]))
            later=dict(b,time=b['time']+10)
            if k!='time_condition':torch.testing.assert_close(m.gates(b,[0,1]),m.gates(later,[0,1]))
            else:self.assertFalse(torch.allclose(m.gates(b,[0,1]),m.gates(later,[0,1])))
    def test_l1_reaches_gate(self):
        _,m,b=self.make('static');m.gates(b,[0,1]).mean().backward()
        self.assertTrue((m.gate_logits.grad>0).all());self.assertTrue(all(p.grad is None for p in m.net.parameters()))
    def test_fallback(self):
        _,m,b=self.make('time_condition');b.update(sensor_valid=torch.tensor([True,False,True,True,True]),fallback_vb_z=torch.full((5,1),3.))
        self.assertEqual(m.estimate(b,[1]).item(),3.)
if __name__=='__main__':unittest.main()

