import unittest
import numpy as np
from ..scripts.diagnose_signal_windows import plateau_window,extract


class SignalWindowTests(unittest.TestCase):
    def test_known_plateau_with_guard(self):
        dc=np.r_[np.ones(1000),np.ones(6500)*5,np.ones(1500)]
        p=plateau_window(dc)
        self.assertEqual(p['end']-p['start'],2000)
        self.assertGreaterEqual(p['start'],1250)
        self.assertLessEqual(p['end'],7250)
        # Affine amplitude changes preserve the selection rule.
        q=plateau_window(dc*3+7)
        self.assertEqual((p['start'],p['end']),(q['start'],q['end']))

    def test_no_forced_window_for_constant_or_short_plateau(self):
        self.assertIsNone(plateau_window(np.ones(9000)))
        self.assertIsNone(plateau_window(np.r_[np.zeros(4000),np.ones(1000),np.zeros(4000)]))

    def test_zero_power_not_fabricated_fraction(self):
        f=extract(np.full(2000,9.9951171875))
        self.assertEqual(f['psd_ac_power'],0)
        self.assertTrue(np.isnan(f['psd_band_0_5_fraction']))

    def test_band_power_fraction_identity(self):
        t=np.arange(2000)/250;x=3+np.sin(2*np.pi*10*t)+.5*np.sin(2*np.pi*40*t)
        f=extract(x)
        for lo,hi in [(0,5),(5,15),(15,30),(30,60),(60,125)]:
            self.assertAlmostEqual(f[f'psd_band_{lo}_{hi}_power']/f['psd_ac_power'],f[f'psd_band_{lo}_{hi}_fraction'])


if __name__=='__main__':unittest.main()
