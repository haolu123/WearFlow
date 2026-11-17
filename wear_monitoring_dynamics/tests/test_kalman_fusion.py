import unittest
import numpy as np
from wear_monitoring_dynamics.evaluation.kalman_fusion import update
from wear_monitoring_dynamics.data.splits import make_split

class KalmanTests(unittest.TestCase):
    def test_no_observation(self):
        y,p,k,c=update(.2,.1,np.nan,.02,.1,False)
        self.assertEqual((y,p,k,c),(.2,.1,0.,False))
    def test_joseph(self):
        y,p,k,c=update(.2,.1,.4,.1,.1)
        self.assertAlmostEqual(y,.3);self.assertAlmostEqual(p,.05);self.assertEqual(k,.5)
    def test_projection(self):
        y,p,k,c=update(.21,.1,0.,.001,.2,True,True)
        self.assertEqual(y,.2);self.assertTrue(c);self.assertGreater(p,0)
    def test_zero_prior_variance(self):
        self.assertEqual(update(.2,0.,.4,.1,.1)[:3],(.2,0.,0.))
    def test_nested(self):
        s=make_split('fold_a','nested',[1,2]);self.assertEqual(s['train'],[3,4,5,7,8]);self.assertEqual(s['validation'],[1,2]);self.assertEqual(s['test'],[])
        with self.assertRaises(ValueError):make_split('fold_a','nested',[1,1])
    def test_old_splits_unchanged(self):
        self.assertEqual(make_split('fold_a','inner',1)['train'],[2,3,4,5,7,8])
        self.assertEqual(len(make_split('fold_b','final')['train']),7)

if __name__=='__main__':unittest.main()

