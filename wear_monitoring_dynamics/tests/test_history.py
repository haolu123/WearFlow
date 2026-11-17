import unittest
import numpy as np
import torch
from ..data.history import add_history
from ..models.history import HistoryCoupled
from ..data.features import SENSORS, STATISTICS


class HistoryTests(unittest.TestCase):
    def fixture(self):
        table=dict(case=np.array([1,1,1,1,2,2]),time=np.array([0.,2.,5.,9.,0.,4.]))
        batch=dict(sensor=torch.randn(6,54),condition=torch.randn(6,4),
                   time=torch.tensor(table['time'],dtype=torch.float32).reshape(-1,1),target=torch.randn(6,1))
        return table,batch

    def test_windows_and_masks(self):
        table,batch=self.fixture();add_history(batch,table,3,1.,'cpu')
        self.assertEqual(batch['history_indices'][2].tolist(),[0,1,2])
        self.assertEqual(batch['history_indices'][3].tolist(),[1,2,3])
        self.assertEqual(batch['history_mask'][4].tolist(),[False,False,True])
        self.assertEqual(batch['history_delta'][3,:,0].tolist(),[2.,3.,4.])
        for row in range(6):
            idx=batch['history_indices'][row][batch['history_mask'][row]].numpy()
            self.assertTrue(np.all(idx<=row));self.assertTrue(np.all(table['case'][idx]==table['case'][row]))

    def test_future_and_other_case_independence(self):
        features=[f'{s}__{t}' for s in SENSORS for t in STATISTICS][:18]
        for arch in ['history_mlp','history_gru','history_cnn']:
            for length in [1,3,5,10]:
                table,batch=self.fixture();add_history(batch,table,length,1.,'cpu')
                model=HistoryCoupled(dict(architecture=arch,history_length=length,selected_features=features))
                original=model.estimate(batch,np.array([2]))
                changed={k:v.clone() for k,v in batch.items()}
                changed['sensor'][3:]=10000;changed['condition'][3:]=-10000;changed['time'][3:]=100000
                changed['target'][:]=float('nan')
                torch.testing.assert_close(original,model.estimate(changed,np.array([2])),atol=0,rtol=0)
                original.sum().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_unselected_sensor_has_no_effect(self):
        table,batch=self.fixture();add_history(batch,table,3,1.,'cpu')
        features=[f'{s}__{t}' for s in SENSORS for t in STATISTICS][:18]
        for arch in ['history_mlp','history_gru','history_cnn']:
            model=HistoryCoupled(dict(architecture=arch,history_length=3,selected_features=features))
            before=model.estimate(batch,np.arange(6))
            batch['sensor'][:,18:]+=1000
            torch.testing.assert_close(before,model.estimate(batch,np.arange(6)),atol=0,rtol=0)


if __name__=='__main__':unittest.main()
