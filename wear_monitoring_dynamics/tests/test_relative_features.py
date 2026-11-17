"""Tests for causal anchoring and fair curve comparisons."""
import unittest
import numpy as np
from ..scripts.diagnose_relative_features import centered, gap_between


class RelativeFeatureTests(unittest.TestCase):
    def test_removes_constant_case_offset(self):
        values=np.array([[10.],[12.],[15.],[30.],[32.],[35.]])
        case=np.array([1,1,1,2,2,2]);refs={1:np.array([0]),2:np.array([3])}
        out=centered(values,case,refs)
        np.testing.assert_array_equal(out[:3],out[3:])
        np.testing.assert_array_equal(values[:,0],[10,12,15,30,32,35])

    def test_future_values_do_not_change_past(self):
        values=np.array([[1.,2.],[2.,4.],[5.,6.],[7.,9.],[8.,12.]])
        case=np.ones(5);refs={1:np.array([0,1,2])}
        a=centered(values,case,refs)
        modified=values.copy();modified[4]*=1000
        b=centered(modified,case,refs)
        np.testing.assert_array_equal(a[:4],b[:4])

    def test_overlap_only_and_sparse_rejection(self):
        x=np.array([0.,1.,2.]);y=2*x
        out=gap_between(x,y,np.array([1.,2.,3.]),np.array([5.,7.,9.]))
        self.assertAlmostEqual(out['gap'],3.)
        self.assertEqual((out['overlap_low'],out['overlap_high']),(1.,2.))
        self.assertIsNone(gap_between(np.array([0.,1.]),np.array([0.,1.]),np.array([2.,3.]),np.array([2.,3.])))


if __name__=='__main__':unittest.main()
