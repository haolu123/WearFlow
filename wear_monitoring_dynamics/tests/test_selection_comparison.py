import unittest
import pandas as pd
from wear_monitoring_dynamics.tests import test_gap_selected as fixtures
from wear_monitoring_dynamics.scripts.run_selection_comparison import selection

class SelectionComparisonTests(unittest.TestCase):
    def test_hybrid_requires_wear_correlation(self):
        d=fixtures.GapSelectionTests().data();fs=[f'f{k}' for k in range(20)]
        d['f0']=d.run%2
        chosen,r=selection(d,[1,2,3,4],fs,'hybrid')
        self.assertEqual(len(chosen),18);self.assertNotIn('f0',chosen)
        self.assertFalse(r.set_index('feature').loc['f0','passes_correlation_gate'])

    def test_heldout_invariance_both_rules(self):
        d=fixtures.GapSelectionTests().data();fs=[f'f{k}' for k in range(20)]
        held=d.copy();held['case']=5;held['VB']=-held.VB*100
        held[fs]=held[fs]*-1000
        for rule in ['correlation','hybrid']:
            a,_=selection(d,[1,2,3,4],fs,rule)
            b,_=selection(pd.concat([d,held]),[1,2,3,4],fs,rule)
            self.assertEqual(a,b)

    def test_gate_does_not_silently_relax(self):
        d=fixtures.GapSelectionTests().data();fs=[f'f{k}' for k in range(20)]
        for f in fs[:4]:d[f]=d.run%2
        with self.assertRaises(ValueError):selection(d,[1,2,3,4],fs,'hybrid')

if __name__=='__main__':unittest.main()
