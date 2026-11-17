import unittest
import torch
from wear_monitoring_dynamics.config import load_config
from wear_monitoring_dynamics.models.registry import build_model,build_dynamics

class DirectTransitionTests(unittest.TestCase):
    def test_identity_and_positive_increment(self):
        for kind in ['direct','direct_positive']:
            cfg=load_config();cfg['dynamics']['transition']=kind
            net=build_dynamics(cfg['dynamics'])
            wear=torch.randn(4,1);cond=torch.randn(4,4)
            torch.testing.assert_close(net.propagate(wear,cond,torch.zeros(4,1)),wear,rtol=0,atol=0)
            if kind=='direct_positive':
                self.assertTrue(bool((net.propagate(wear,cond,torch.ones(4,1))>=wear).all()))

    def test_no_future_sensor_or_label_input_and_monitor_gradients(self):
        for kind in ['direct','direct_positive','ode']:
            torch.manual_seed(5)
            cfg=load_config();cfg['dynamics'].update(mode='joint_pretrained',transition=kind)
            model=build_model(cfg)
            batch=dict(sensor=torch.randn(4,54),condition=torch.randn(4,4),time=torch.arange(4.).reshape(4,1),target=torch.randn(4,1))
            pairs=torch.tensor([[0,2,1],[1,3,1]])
            pred=model.forecast(batch,pairs)
            changed={k:v.clone() for k,v in batch.items()}
            changed['sensor'][2:]=9999;changed['condition'][2:]=-9999;changed['target'][:]=9999
            torch.testing.assert_close(model.forecast(changed,pairs),pred,rtol=0,atol=0)
            pred.square().mean().backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.monitor.parameters() if p.grad is not None),0)

if __name__=='__main__':unittest.main()

