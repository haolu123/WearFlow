"""Execute author notebook architecture definitions only; compare dropout and gradients."""
import unittest
import torch
from torch import nn
from model import Monitor, ZhangPhysics
from wear_monitoring_dynamics.tests.test_zhang import reference_classes


class Parity(unittest.TestCase):
    def test_author_outputs_and_gradients(self):
        ref = reference_classes()
        m, f = Monitor().double(), ZhangPhysics(17).double()
        original = ref['Solution_u']().double()
        original.encoder.net[0] = nn.Linear(17, 32).double()
        original.load_state_dict(m.state_dict(), strict=True)
        original_f = ref['MLP'](input_dim=53, output_dim=1, layers_num=4,
                                hidden_dim=50, droupout=.2).double()
        original_f.load_state_dict(f.network.state_dict(), strict=True)
        holder = nn.Module()
        holder.solution_u, holder.dynamical_F = original, original_f
        x = torch.randn(5, 17, dtype=torch.float64)
        for training in (False, True):
            m.train(training); f.train(training); holder.train(training)
            m.zero_grad(); f.zero_grad(); holder.zero_grad()
            torch.manual_seed(91)
            u, r = f.residual(m, x)
            (u.square().mean()+r.square().mean()).backward()
            torch.manual_seed(91)
            ur, rr = ref['forward'](holder, x.clone())
            (ur.square().mean()+rr.square().mean()).backward()
            torch.testing.assert_close(u, ur, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(r, rr, rtol=1e-12, atol=1e-12)
            for a, b in zip(list(m.parameters())+list(f.parameters()),
                            list(original.parameters())+list(original_f.parameters())):
                torch.testing.assert_close(a.grad, b.grad, rtol=1e-10, atol=1e-10)

if __name__ == '__main__':
    unittest.main()
