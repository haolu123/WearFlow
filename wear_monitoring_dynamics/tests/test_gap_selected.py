import unittest
import numpy as np
import pandas as pd
from wear_monitoring_dynamics.scripts.run_gap_selected import rank_features,inputs

class GapSelectionTests(unittest.TestCase):
    def data(self):
        rows=[]
        for c in range(1,5):
            for i in range(5):
                rows.append(dict(case=c,run=i+1,VB=i/10,time=i+1,DOC=float(c),feed=.2,material=1+c%2,
                    **{f'f{k}':i*.1+c*.001*(k+1) for k in range(20)}))
        return pd.DataFrame(rows)

    def test_heldout_labels_do_not_select(self):
        d=self.data();fs=[f'f{k}' for k in range(20)];a,_=rank_features(d,[1,2,3,4],fs)
        held=d.copy();held['case']=5;held['VB']=held.VB*100+100
        b,_=rank_features(pd.concat([d,held]),[1,2,3,4],fs)
        self.assertEqual(a,b);self.assertEqual(a,fs[:18])

    def test_heldout_features_do_not_fit_scalers(self):
        d=self.data();fs=[f'f{k}' for k in range(18)]
        xs,y,ids,a=inputs(d,[1,2,3],fs)
        d.loc[d.case==4,fs]=1e12;d.loc[d.case==4,'VB']=1e6
        xx,yy,ii,b=inputs(d,[1,2,3],fs)
        self.assertEqual(a,b);np.testing.assert_array_equal(ids,ii)
        for key in xs:np.testing.assert_array_equal(xs[key][ids],xx[key][ids])
        np.testing.assert_array_equal(y[ids],yy[ids])
        self.assertEqual(xs['ct'].shape[1],5);self.assertEqual(xs['ct_top12'].shape[1],17);self.assertEqual(xs['ct_top18'].shape[1],23)

if __name__=='__main__':unittest.main()
