import unittest
import numpy as np
from wear_monitoring_dynamics.scripts.explore_current_components import component_features,choose_windows,band,DEFS
from wear_monitoring_dynamics.data.current_processing import FS,DC_RAIL

class ComponentTests(unittest.TestCase):
    def test_harmonic_recovery_and_residual(self):
        t=np.arange(2000)/FS
        x=2*np.sin(2*np.pi*12.5*t)+.4*np.cos(2*np.pi*25*t)
        f,q=component_features(x,np.ones(2000)*DC_RAIL)
        self.assertEqual(set(f),set(DEFS))
        self.assertAlmostEqual(q['frequency'],12.5,places=3)
        self.assertAlmostEqual(f['period_h1_amplitude'],2,places=3)
        self.assertAlmostEqual(f['period_h2_amplitude'],.4,places=3)
        self.assertLess(f['period_residual_rms'],.001)
        self.assertAlmostEqual(f['control_dc_mean'],DC_RAIL)
        self.assertTrue(np.isnan(f['net_dc_mean']))

    def test_idle_subtraction_rejects_offset(self):
        t=np.arange(2000)/FS;x=np.sin(2*np.pi*12*t)
        idle=(x[:500]*.2,np.ones(500)*2)
        a,_=component_features(x,np.ones(2000)*7,idle)
        b,_=component_features(x,np.ones(2000)*10,(idle[0],idle[1]+3))
        self.assertAlmostEqual(a['net_dc_mean'],5)
        self.assertAlmostEqual(a['net_dc_mean'],b['net_dc_mean'])

    def test_windows_and_no_idle(self):
        t=np.arange(9000)/FS;level=np.where((t>5)&(t<30),1,.1)
        x=level*np.sin(2*np.pi*12*t);dc=level*8
        stable,idle=choose_windows(x,dc)
        self.assertEqual(stable[1]-stable[0],2000)
        self.assertLess(idle[1],5*FS)
        self.assertGreaterEqual(idle[1]-idle[0],250)
        self.assertEqual(choose_windows(np.zeros(9000),np.zeros(9000)),(None,None))

    def test_noise_residual_and_scale(self):
        rng=np.random.default_rng(7);t=np.arange(2000)/FS
        x=2*np.sin(2*np.pi*12.7*t)+rng.normal(0,.1,len(t));d=np.ones(len(t))
        a,_=component_features(x,d);b,_=component_features(3*x,d)
        self.assertGreater(a['period_residual_rms'],.08)
        self.assertLess(a['period_residual_rms'],.12)
        self.assertAlmostEqual(b['period_residual_rms']/a['period_residual_rms'],3,places=3)
        self.assertAlmostEqual(a['period_residual_fraction'],b['period_residual_fraction'],places=4)

    def test_fractional_bin_integration(self):
        hz=np.arange(20.);p=np.ones(20)
        self.assertAlmostEqual(band(hz,p,5.1,6.6),1.5)
        self.assertAlmostEqual(band(hz,p,5.4,6.9),1.5)

if __name__=='__main__':unittest.main()
