import unittest
import torch
from wear_monitoring_dynamics.config import load_config
from wear_monitoring_dynamics.study import build_tasks,variants
from wear_monitoring_dynamics.models.registry import build_model
from wear_monitoring_dynamics.scripts.run_top18_ode import specification


class Top18ODETests(unittest.TestCase):
    def test_full_matrix_and_split_dependencies(self):
        tasks=build_tasks(load_config(),specification())
        self.assertEqual(len(tasks),112)
        self.assertEqual(sum(t['type']=='pretrain' for t in tasks),16)
        for t in tasks:
            if t['type']=='monitor':
                cfg=t['config'];self.assertEqual(len(cfg['monitor']['selected_features']),18)
                self.assertEqual(cfg['loss']['forecasting_weight'],1.)
                self.assertEqual(cfg['loss']['derivative_weight'],0.)
                e=cfg['experiment'];self.assertIn(f'ode/{e["fold"]}/{e["holdout"]}',t['dependencies'])

    def test_forecast_uses_estimated_state_and_trains_both_branches(self):
        for cfg in variants(load_config(),specification()):
            torch.manual_seed(0);m=build_model(cfg)
            batch={'sensor':torch.randn(4,54),'condition':torch.randn(4,4),'time':torch.arange(4.).reshape(-1,1),'target':torch.randn(4,1)}
            pairs=torch.tensor([[0,1,1],[1,2,1]])
            prediction=m.forecast(batch,pairs)
            alternate=dict(batch,target=batch['target']+100.)
            torch.testing.assert_close(prediction,m.forecast(alternate,pairs),rtol=0,atol=0)
            prediction.sum().backward()
            for module in [m.monitor,m.dynamics]:
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None),0.)
            start=m.estimate(batch,pairs[:,0]).detach()
            self.assertTrue(bool((prediction.detach()>=start).all()))


if __name__=='__main__':unittest.main()
