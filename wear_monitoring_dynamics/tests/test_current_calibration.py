import unittest
import numpy as np
from wear_monitoring_dynamics.scripts.compare_current_calibration import compare

class CalibrationTests(unittest.TestCase):
    def test_identity_and_offset(self):
        a=np.array([[0,1],[.5,2],[1,3.]])
        self.assertEqual(compare(a,a)['raw_gap'],0)
        b=a.copy();b[:,1]+=4
        self.assertAlmostEqual(compare(a,b)['raw_gap'],4)
        self.assertAlmostEqual(compare(a,b)['signed_bias'],-4)

    def test_crossing_and_rescale(self):
        a=np.array([[0,0],[1,1.]])
        b=np.array([[0,1],[1,0.]])
        r=compare(a,b)
        self.assertAlmostEqual(r['raw_gap'],.5)
        self.assertAlmostEqual(r['signed_bias'],0)
        a[:,1]*=5;b[:,1]*=5
        self.assertAlmostEqual(compare(a,b)['raw_gap'],2.5)

    def test_overlap_and_missing(self):
        a=[[0,1],[1,2],[2,3]];b=[[1,2],[2,3],[3,100],[4,np.nan]]
        r=compare(a,b)
        self.assertEqual((r['overlap_low'],r['overlap_high']),(1,2))
        self.assertEqual(r['raw_gap'],0)
        self.assertIsNone(compare(a,[[2,3],[3,4]]))

    def test_flat_and_sparse(self):
        r=compare([[0,0],[1,0]],[[0,0],[1,0]])
        self.assertEqual(r['overlap_swing'],0)
        self.assertTrue(r['sparse'])

if __name__=='__main__':unittest.main()
