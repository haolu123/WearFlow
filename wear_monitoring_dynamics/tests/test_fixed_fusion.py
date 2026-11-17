import unittest,numpy as np
from wear_monitoring_dynamics.evaluation.fixed_fusion import rollout
class FusionTests(unittest.TestCase):
    def run_filter(self,obs,valid=None,alpha=0.5,mono=False):
        return rollout([0,1,3],np.array(obs,float).reshape(3,1),np.ones((3,1),bool) if valid is None else np.array(valid).reshape(3,1),[alpha],[mono],[.1],lambda x,dt:x+.1*dt)[1].ravel()
    def test_endpoints(self):
        np.testing.assert_allclose(self.run_filter([9,8,7],alpha=0),[.1,.2,.4]);np.testing.assert_allclose(self.run_filter([9,.3,.5],alpha=1),[.1,.3,.5])
    def test_causality(self):
        a=self.run_filter([.1,.3,.5]);b=self.run_filter([.1,.3,99]);np.testing.assert_array_equal(a[:2],b[:2])
    def test_invalid_skip(self):
        np.testing.assert_allclose(self.run_filter([.1,np.nan,.5],valid=[True,False,True],alpha=1),[.1,.2,.5])
    def test_monotone(self):
        np.testing.assert_allclose(self.run_filter([.1,.4,.2],alpha=1,mono=True),[.1,.4,.4]);self.assertLess(self.run_filter([.1,.4,.2],alpha=1)[-1],.4)
if __name__=='__main__':unittest.main()

