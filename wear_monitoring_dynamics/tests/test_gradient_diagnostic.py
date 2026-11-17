import unittest
import numpy as np
import torch
from wear_monitoring_dynamics.scripts.diagnose_monotonicity import gradient_vector, geometry, prepare, HOLDOUTS


class GradientDiagnosticTests(unittest.TestCase):
    def test_flat_gradient_unused_parameter_and_no_accumulation(self):
        a=torch.nn.Parameter(torch.tensor([2.,-3.]));b=torch.nn.Parameter(torch.tensor([7.]))
        result=gradient_vector(a.square().sum(),[a,b])
        torch.testing.assert_close(result,torch.tensor([4.,-6.,0.],dtype=torch.float64))
        self.assertIsNone(a.grad);self.assertIsNone(b.grad)
        torch.testing.assert_close(a.detach(),torch.tensor([2.,-3.]))

    def test_geometry_conflict_and_zero(self):
        a=torch.tensor([3.,4.],dtype=torch.float64)
        self.assertAlmostEqual(geometry(2*a,a)['ratio'],2.)
        self.assertAlmostEqual(geometry(-a,a)['cosine'],-1.)
        self.assertTrue(np.isnan(geometry(a,torch.zeros_like(a))['ratio']))
        self.assertTrue(np.isnan(geometry(torch.zeros_like(a),a)['cosine']))

    def test_fixed_small_study_matrix(self):
        _,rows,configs=prepare()
        self.assertEqual(len(rows),32);self.assertEqual(len(configs),32)
        self.assertEqual({r['weight'] for r in rows},{0.,.1})
        for key,cfg in configs.items():
            self.assertIn(key[4],HOLDOUTS[key[3]])
            self.assertEqual(cfg['experiment']['seed'],0)
            self.assertEqual(cfg['experiment']['role'],'inner')
            self.assertEqual(cfg['training']['epochs'],500)
            self.assertEqual(cfg['monotonicity']['weight'],key[2])
            self.assertIn(key[2],[1.,10.])


if __name__=='__main__':
    unittest.main()
