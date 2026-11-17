import unittest
import numpy as np
from ..data.current_processing import process_current,lowpass_segments,DC_RAIL


class CurrentProcessingTests(unittest.TestCase):
    def test_bad_record_quarantine_preserves_length(self):
        for case in [2,12]:
            out,reason=process_current(np.ones(9000),np.ones(9000),case,1)
            self.assertTrue(reason);self.assertEqual(len(out['ac_waveform']),9000)
            self.assertTrue(np.isnan(out['ac_waveform']).all());self.assertFalse(out['record_valid'].any())

    def test_saturation_is_missing_not_reconstructed(self):
        dc=np.ones(9000)*4;dc[3000:6000]=DC_RAIL
        out,reason=process_current(np.sin(np.arange(9000)),dc,1,1)
        self.assertFalse(reason);self.assertTrue(np.isnan(out['dc_observed'][3000:6000]).all())
        self.assertTrue(np.isnan(out['dc_trend_5hz'][2950:6050]).all())
        self.assertTrue(np.isfinite(out['ac_waveform']).all())

    def test_no_filtering_across_gaps(self):
        rng=np.random.default_rng(4);x=rng.normal(size=9000);valid=np.ones(9000,bool);valid[3500:4500]=False
        a=lowpass_segments(x,valid,5);x[4500:]+=1000;b=lowpass_segments(x,valid,5)
        np.testing.assert_array_equal(a[:3500],b[:3500])
        self.assertTrue(np.isnan(a[3500:4500]).all())

    def test_low_frequency_retained_high_ripple_removed(self):
        t=np.arange(9000)/250;true=4+.5*np.sin(2*np.pi*1*t);x=true+.2*np.sin(2*np.pi*60*t)
        y=lowpass_segments(x,np.ones(len(x),bool),5);mask=np.isfinite(y)&(t>2)&(t<34)
        self.assertLess(np.sqrt(np.mean((y[mask]-true[mask])**2)),.005)

    def test_ac_carrier_unchanged_and_envelope_recovered(self):
        t=np.arange(9000)/250;amplitude=2+.4*np.sin(2*np.pi*.2*t);ac=amplitude*np.sin(2*np.pi*12.5*t)
        out,_=process_current(ac,np.ones(len(t))*4,1,1);np.testing.assert_array_equal(out['ac_waveform'],ac)
        mask=np.isfinite(out['ac_envelope_2hz'])&(t>2)&(t<34)
        self.assertLess(np.sqrt(np.mean((out['ac_envelope_2hz'][mask]-amplitude[mask]/np.sqrt(2))**2)),.03)


if __name__=='__main__':unittest.main()
