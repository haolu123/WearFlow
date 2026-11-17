"""Source-equation, causality and state-isolation checks; no NASA training."""
import json, time, unittest
from pathlib import Path
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from sun_pigpr import SunPIGPR, PIGPRConfig, gaussian_smooth, matern32, paper_grid

class TestPIGPR(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.t=np.tile(np.arange(1,13,dtype=float),2)
        self.case=np.repeat([1,2],12)
        self.X=np.column_stack([self.t, rng.normal(size=24),np.ones(24)])
        self.y=.02*self.t+.03
        self.model=SunPIGPR().fit(self.X,self.y,self.t,self.case)

    def test_equation20_linear_recovery(self):
        h=gaussian_smooth(self.X[:,0],self.t,self.case)
        y=.04*h+.02
        model=SunPIGPR().fit(self.X,y,self.t,self.case)
        np.testing.assert_allclose(model.mean_coef_,[.04,.02],atol=1e-12)
        np.testing.assert_allclose(model.predict(self.X,self.t,self.case),y,atol=1e-12)

    def test_kernel_matches_standard_matern32(self):
        X=np.array([[0.,2.],[1.,3.],[-2.,.5]])
        np.testing.assert_allclose(matern32(X,X,1.2,.7),
                                   .7**2*Matern(length_scale=1.2,nu=1.5)(X),atol=1e-14)

    def test_equation17_and12_direct_solve(self):
        m=self.model
        p=m.predict_components(self.X,self.t,self.case)
        K=matern32(m.X_train_,m.X_train_,1.5,.01)
        Knoise=K+np.eye(len(K))*(1.5**2+1e-10)
        residual=(self.y-p["physical_mean"])/m.y_scale_
        expected=p["physical_mean"]+m.y_scale_*K@np.linalg.solve(Knoise,residual)
        np.testing.assert_allclose(p["mean"],expected,atol=1e-13)
        var=np.diag(K-K@np.linalg.solve(Knoise,K))
        np.testing.assert_allclose(p["latent_std"]**2,m.y_scale_**2*var,atol=1e-14)

    def test_no_future_dependency(self):
        before=self.model.predict(self.X,self.t,self.case)
        mutated=self.X.copy(); mutated[7:12]+=1e5
        after=self.model.predict(mutated,self.t,self.case)
        np.testing.assert_allclose(before[:7],after[:7],atol=0,rtol=0)
        np.testing.assert_allclose(before[12:],after[12:],atol=0,rtol=0)

    def test_prefix_prediction_agrees(self):
        full=self.model.predict(self.X,self.t,self.case)
        prefix=self.model.predict(self.X[:6],self.t[:6],self.case[:6])
        np.testing.assert_allclose(full[:6],prefix,atol=1e-14)

    def test_centered_offline_has_future_dependency(self):
        v=np.arange(10,dtype=float);t=np.arange(10);case=np.ones(10)
        a=gaussian_smooth(v,t,case,mode="centered")
        v[5]=100
        b=gaussian_smooth(v,t,case,mode="centered")
        self.assertNotEqual(a[4],b[4])

    def test_predict_does_not_refit(self):
        before=json.dumps(self.model.audit_state(),sort_keys=True)
        self.model.predict(self.X*100,self.t,self.case)
        self.assertEqual(before,json.dumps(self.model.audit_state(),sort_keys=True))

    def test_unlabeled_records_allowed(self):
        y=self.y.copy();y[::4]=np.nan
        m=SunPIGPR().fit(self.X,y,self.t,self.case)
        self.assertEqual(m.n_train_labels_,18)
        self.assertTrue(np.isfinite(m.predict(self.X,self.t,self.case)).all())

    def test_invalid_input_rejected(self):
        X=self.X.copy();X[0,0]=np.nan
        with self.assertRaises(ValueError):SunPIGPR().fit(X,self.y,self.t,self.case)

    def test_grid_count_and_range(self):
        g=list(paper_grid())
        self.assertEqual(len(g),3375)
        self.assertEqual(g[0],dict(signal_std=.01,noise_std=.1,length_scale=.1))
        self.assertEqual(g[-1],dict(signal_std=.15,noise_std=1.5,length_scale=1.5))

if __name__=="__main__":
    start=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TestPIGPR))
    Path(__file__).with_name("smoke_results.json").write_text(json.dumps({
        "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
        "seconds":time.perf_counter()-start,"data":"synthetic only","passed":result.wasSuccessful()
    },indent=2),encoding="utf-8")
    raise SystemExit(not result.wasSuccessful())

