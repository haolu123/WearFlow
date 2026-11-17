import unittest
import numpy as np
import torch
from ..scripts.run_initial_context import initial_context, make_model, initial_to_delta_weights


class InitialContextTests(unittest.TestCase):
    def test_first_valid_not_future_or_labeled(self):
        x=torch.arange(5*18,dtype=torch.float32).reshape(5,18)
        case=np.array([1,1,1,2,2]);run=np.array([1,2,3,1,2]);valid=np.array([False,True,True,True,True])
        initial,anchor=initial_context(x,case,run,valid)
        np.testing.assert_array_equal(anchor,[-1,1,1,3,3])
        self.assertTrue(torch.equal(initial[0],torch.zeros(18)))
        self.assertTrue(torch.equal(initial[2],x[1]))

    def test_future_append_does_not_change_prefix(self):
        x=torch.randn(5,18);case=np.array([1,1,1,2,2]);run=np.array([1,2,3,1,2]);valid=np.ones(5,dtype=bool)
        a,ids=initial_context(x,case,run,valid)
        b,more=initial_context(torch.cat([x,torch.full((1,18),999.)]),np.r_[case,1],np.r_[run,4],np.r_[valid,True])
        torch.testing.assert_close(a,b[:5],atol=0,rtol=0)
        np.testing.assert_array_equal(ids,more[:5])

    def test_equivalent_information_and_parameter_count(self):
        model=make_model(4);mapped=initial_to_delta_weights(model)
        x=torch.randn(30,18);initial=torch.randn(30,18);ct=torch.randn(30,5)
        a=torch.cat([x,initial,ct],dim=1);b=torch.cat([x,x-initial,ct],dim=1)
        torch.testing.assert_close(model(a),mapped(b),atol=1e-7,rtol=1e-5)
        self.assertEqual(sum(p.numel() for p in model.parameters()),4801)


if __name__=='__main__':unittest.main()
