import copy
import unittest
import zipfile
import torch
from wear_monitoring_dynamics.config import load_config, PACKAGE, project_path
from wear_monitoring_dynamics.study import variants, build_tasks
from wear_monitoring_dynamics.models.registry import build_model
from wear_monitoring_dynamics.models.feature_subset import feature_indices
from wear_monitoring_dynamics.scripts.run_top18_features import specification


class FeatureSubsetTests(unittest.TestCase):
    def test_unselected_features_cannot_affect_outputs_or_gradients(self):
        cfg=load_config(PACKAGE/'configs/time_conditioning.yaml')
        for c in variants(cfg,specification()):
            torch.manual_seed(3);model=build_model(c)
            x=torch.randn(5,54,requires_grad=True);condition=torch.randn(5,4);time=torch.randn(5,1)
            idx=feature_indices(c['monitor']['selected_features']);other=[i for i in range(54) if i not in idx]
            y=model.monitor(x,condition,time);changed=x.detach().clone();changed[:,other]=1e8
            torch.testing.assert_close(y,model.monitor(changed,condition,time),rtol=0,atol=0)
            g=torch.autograd.grad(y.sum(),x)[0]
            self.assertEqual(int(torch.count_nonzero(g[:,other])),0)
            self.assertGreater(int(torch.count_nonzero(g[:,idx])),0)
            self.assertEqual(tuple(y.shape),(5,1))
            restored=build_model(c);restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(y,restored.monitor(x,condition,time),rtol=0,atol=0)
            if c['monitor']['architecture']=='sensor_attention':
                self.assertEqual(model.monitor.active_channels,[0,1,3,4,5])
                self.assertEqual(len(set(model.monitor.feature_slots.tolist())),18)

    def test_default_54_matches_archived_registry_bit_for_bit(self):
        path=project_path('wear_monitoring_dynamics/outputs/sources/fabafe1625279f6a5bd64e41abc40a1593681dc9b041a552252a5edb4579a3d1.zip')
        with zipfile.ZipFile(path) as z:source=z.read('wear_monitoring_dynamics/models/registry.py').decode()
        ns={'__package__':'wear_monitoring_dynamics.models'};exec(compile(source,str(path),'exec'),ns)
        for cfg in variants(load_config(),specification()):
            cfg=copy.deepcopy(cfg);cfg['monitor'].pop('selected_features')
            torch.manual_seed(4);a=build_model(cfg)
            torch.manual_seed(4);b=ns['build_model'](cfg)
            for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)

    def test_matrix_and_feature_validation(self):
        spec=specification();self.assertEqual(len(build_tasks(load_config(),spec)),96)
        for item in spec['variants']:
            self.assertEqual(len(item['monitor']['selected_features']),18)
            self.assertEqual(item['dynamics']['mode'],'none')
        with self.assertRaises(ValueError):feature_indices(['bad'])
        with self.assertRaises(ValueError):feature_indices(['smcAC__std','smcAC__std'])


if __name__=='__main__':unittest.main()
