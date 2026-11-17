import unittest
import numpy as np
from ..data.sensor_transforms import fit_sensor_transform,apply_sensor_transform

class SensorTransformTests(unittest.TestCase):
    def test_no_transform_equals_standardization(self):
        x=np.array([[1.,2],[2,4],[5,8]])
        state,(mean,sd)=fit_sensor_transform(x,dict(method='zscore',indices=[0,1]))
        np.testing.assert_array_equal(apply_sensor_transform(x,state),x)
        np.testing.assert_allclose(mean,x.mean(0));np.testing.assert_allclose(sd,x.std(0))

    def test_log_only_designated_features(self):
        x=np.array([[0.,-3],[2.,1],[4.,7]])
        state,_=fit_sensor_transform(x,dict(method='log_zscore',indices=[0]))
        self.assertEqual(state['log_scales'],[3.])
        out=apply_sensor_transform(x,state);np.testing.assert_allclose(out[:,0],np.log1p(x[:,0]/3));np.testing.assert_array_equal(out[:,1],x[:,1])
        self.assertTrue(np.isfinite(apply_sensor_transform(np.array([[1000.,-100.]]),state)).all())

    def test_robust_and_frozen_clipping(self):
        x=np.array([[0.],[1.],[2.],[100.]])
        _,(mean,scale)=fit_sensor_transform(x,dict(method='robust',indices=[0]))
        np.testing.assert_allclose(mean,[1.5]);np.testing.assert_allclose(scale,[25.75])
        state,_=fit_sensor_transform(x,dict(method='clip_zscore',indices=[0],quantiles=[.01,.99]))
        lo,hi=np.quantile(x[:,0],[.01,.99]);np.testing.assert_allclose(apply_sensor_transform(np.array([[-100.],[1e9]]),state).ravel(),[lo,hi])

if __name__=='__main__':unittest.main()
