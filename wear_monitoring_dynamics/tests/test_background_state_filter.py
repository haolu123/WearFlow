import unittest
import numpy as np
import torch
from ..scripts.run_background_state_filter import PositiveRate, dynamics_inputs, filter_sequence, fit_observation, KINDS


class BackgroundStateTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.obs=dict(beta=np.zeros((10,18)),R=np.eye(18)*.1)
        self.obs['beta'][5,0]=1
        B=np.zeros((18,2));B[1,0]=1;B[2,1]=1
        for k in KINDS[1:]:self.obs[k]=B
        self.prior=np.linspace(.1,1,10);self.sensor=rng.normal(0,.01,(10,18))
        self.sensor[:,0]=self.prior+.15;self.sensor[:,1]=2;self.sensor[:,2]=-1

    def run_filter(self,kind,**kw):
        args=dict(prior=self.prior,dt=np.ones(10),sensor=self.sensor,condition=np.zeros(4),valid=np.ones(10,bool),
                  observation=self.obs,kind=kind,q=.01,r_scale=1.)
        args.update(kw);return filter_sequence(**args)

    def test_constant_rate_and_prefix(self):
        c=dict(table=dict(time=np.array([0.,1.,3.]),case=np.ones(3),run=np.arange(3)),scalers=dict(time_scale=1),ct=torch.zeros(3,5))
        model=PositiveRate()
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            y=model(*dynamics_inputs(c)).numpy()
        np.testing.assert_allclose(y,np.log(2)*np.array([1,2,4]),rtol=1e-6)
        c['table']={k:v[:2] for k,v in c['table'].items()};c['ct']=c['ct'][:2]
        with torch.no_grad():np.testing.assert_allclose(model(*dynamics_inputs(c)).numpy(),y[:2])

    def test_off_invalid_and_causal(self):
        for k in KINDS:
            np.testing.assert_array_equal(self.run_filter(k,r_scale=0)['wear'],self.prior)
            np.testing.assert_allclose(self.run_filter(k,valid=np.zeros(10,bool))['wear'],self.prior)
            a=self.run_filter(k);b=self.run_filter(k,prior=self.prior[:5],dt=np.ones(5),sensor=self.sensor[:5],valid=np.ones(5,bool))
            for field in a:np.testing.assert_allclose(a[field][:5],b[field])

    def test_classical_kalman_equivalence_and_background_recovery(self):
        out=self.run_filter('background_filter')
        H=np.column_stack([self.obs['beta'][5],self.obs['background_filter']]);P=np.diag([.26,1.,1.])
        K=P@H.T@np.linalg.inv(self.obs['R']+H@P@H.T)
        expected=np.array([self.prior[0],0.,0.])+K@(self.sensor[0]-H@np.array([self.prior[0],0.,0.]))
        np.testing.assert_allclose(np.r_[out['wear'][0],out['background'][0]],expected,atol=1e-12)
        np.testing.assert_allclose(out['background'][-1],[2,-1],atol=.025)
        self.assertLess(abs(out['wear'][-1]-(self.prior[-1]+.15)),.01)

    def test_orthogonal_background_cannot_change_wear_update(self):
        plain=self.run_filter('sensor_filter');separated=self.run_filter('separated_background_filter')
        np.testing.assert_allclose(plain['wear'],separated['wear'],atol=1e-12)
        np.testing.assert_allclose(plain['sd'],separated['sd'],atol=1e-12)

    def test_observation_ignores_heldout_labels(self):
        rng=np.random.default_rng(9);cases=np.repeat(np.arange(5),8);n=len(cases)
        c=dict(train=np.arange(32),table=dict(case=cases,VB=rng.uniform(0,.5,n)),scalers=dict(wear=[0.,.2]),
               ct=torch.tensor(rng.normal(size=(n,5)),dtype=torch.float32),sensor=torch.tensor(rng.normal(size=(n,23)),dtype=torch.float32),split=dict(train=[0,1,2,3]))
        a=fit_observation(c);c['table']['VB'][32:]=999;b=fit_observation(c)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        self.assertLess(a['background_wear_overlap'][1],1e-6)


if __name__=='__main__':unittest.main()
