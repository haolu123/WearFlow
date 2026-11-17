import unittest
import numpy as np
import torch
from ..data.sensor_quality import raw_quality_metrics,fit_rules,assess,apply_quality
from ..data.history import add_history
from ..data.preprocessing import fit_scalers
from ..data.features import SENSORS,STATISTICS
from ..models.history import HistoryCoupled


class SensorQualityTests(unittest.TestCase):
    def test_calibration_ignores_heldout_values_and_labels(self):
        rng=np.random.default_rng(12)
        records=[dict(case=c,run=r,VB=0.,**{s:rng.normal(3,.5,200) for s in SENSORS}) for c in [1,2,3] for r in [1,2,3]]
        original=fit_rules(raw_quality_metrics(records),[1,2])
        for record in records:
            record['VB']=10000
            if record['case']==3:
                for s in SENSORS:record[s]*=1e9
        self.assertEqual(original,fit_rules(raw_quality_metrics(records),[1,2]))

    def test_hard_failure_and_length_alone(self):
        rng=np.random.default_rng(1)
        records=[dict(case=1,run=r,**{s:rng.normal(3,.5,200) for s in SENSORS}) for r in range(5)]
        rules=fit_rules(raw_quality_metrics(records),[1])
        records += [dict(case=2,run=1,**{s:rng.normal(3,.5,320) for s in SENSORS}),
                    dict(case=2,run=2,**{s:np.ones(200) for s in SENSORS})]
        d,_=assess(records,raw_quality_metrics(records),rules)
        self.assertTrue(d[(d.case==2)&(d.run==1)].sensor_valid.item())
        self.assertFalse(d[(d.case==2)&(d.run==2)].sensor_valid.item())

    def test_skipping_preserves_time_and_case_boundaries(self):
        table=dict(case=np.array([1,1,1,1,2]),run=np.array([1,2,3,4,1]),time=np.array([0.,2.,7.,11.,0.]))
        table=apply_quality(table,dict(invalid_records=[[1,2],[2,1]]))
        batch={};add_history(batch,table,3,1.,'cpu')
        np.testing.assert_array_equal(batch['history_indices'][3],[0,2,3])
        np.testing.assert_array_equal(batch['history_delta'][3,:,0],[0.,7.,4.])
        self.assertFalse(batch['history_mask'][4].any())

    def test_invalid_current_uses_fallback_and_bad_history_is_ignored(self):
        torch.manual_seed(1)
        table=dict(case=np.ones(4),run=np.arange(1,5),time=np.arange(4.),sensor_valid=np.array([False,True,True,True]))
        batch=dict(sensor=torch.randn(4,54),condition=torch.randn(4,4),time=torch.arange(4.).reshape(-1,1),
                   fallback_vb_z=torch.full((4,1),.25))
        add_history(batch,table,3,1.,'cpu')
        names=[f'{s}__{t}' for s in SENSORS for t in STATISTICS][:18]
        for arch in ['history_mlp','history_gru','history_cnn']:
            model=HistoryCoupled(dict(architecture=arch,history_length=3,selected_features=names))
            before=model.estimate(batch,np.arange(4))
            self.assertEqual(before[0].item(),.25)
            batch['sensor'][0]+=10000
            torch.testing.assert_close(before,model.estimate(batch,np.arange(4)),atol=0,rtol=0)

    def test_invalid_sensor_excluded_from_scaler_not_label(self):
        t=dict(case=np.ones(5),run=np.arange(5),time=np.arange(5.),VB=np.arange(5.)*.1,
               DOC=np.ones(5),feed=np.ones(5),material=np.ones(5),sensor=np.arange(10.).reshape(5,2),sensor_valid=np.array([False,True,True,True,True]))
        first=fit_scalers(t,[1]);t['sensor'][0]=1e9;second=fit_scalers(t,[1])
        self.assertEqual(first,second);self.assertAlmostEqual(first['wear'][0],.2)


if __name__=='__main__':unittest.main()
