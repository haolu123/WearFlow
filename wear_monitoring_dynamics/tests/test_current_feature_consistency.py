import unittest
import numpy as np
from ..scripts.current_feature_consistency import interruptions,extract,describe_profile
from ..data.current_processing import DC_RAIL


class CurrentFeatureTests(unittest.TestCase):
    def signal(self,drop=False):
        t=np.arange(9000)/250;load=np.zeros(9000);load[1000:7500]=1
        if drop:load[4000:4250]=0
        return (.4+2*load)*np.sin(2*np.pi*12.5*t),1.4+6*load

    def test_internal_drop_detected_but_entry_exit_kept(self):
        self.assertEqual(interruptions(*self.signal()),[])
        spans=interruptions(*self.signal(True));self.assertEqual(len(spans),1)
        self.assertLess(spans[0][0],4100);self.assertGreater(spans[0][1],4150)

    def test_saturation_features_do_not_change_input(self):
        ac,dc=self.signal();dc[1500:7000]=DC_RAIL;copy=dc.copy();f=extract(ac,dc)
        np.testing.assert_array_equal(copy,dc)
        self.assertAlmostEqual(f['dc_saturation_fraction'],5500/9000)
        self.assertAlmostEqual(f['dc_saturation_longest_seconds'],22)

    def test_unobserved_exit_and_saturation_time_are_missing(self):
        ac,dc=self.signal();dc[7500:]=7.4;f=extract(ac,dc)
        self.assertTrue(np.isnan(f['dc_level_fall90_10_seconds']))
        self.assertTrue(np.isnan(f['dc_saturation_first_seconds']))

    def test_relative_width_invariant_to_positive_affine_scale(self):
        _,dc=self.signal();a=describe_profile(dc,'x');b=describe_profile(3*dc+4,'x')
        for tag in ['20','50','80']:self.assertEqual(a['x_width'+tag+'_seconds'],b['x_width'+tag+'_seconds'])


if __name__=='__main__':unittest.main()
