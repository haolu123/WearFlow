"""Tests for provenance safety only; no model accuracy claims."""
import unittest
from readiness import load_spec, require_reproducible, MissingPrimaryMethodError

class EvidenceGateTests(unittest.TestCase):
    def test_unknown_core_blocks_training(self):
        with self.assertRaisesRegex(MissingPrimaryMethodError, "polynomial_equation_and_fit"):
            require_reproducible()
    def test_empty_missing_list_does_not_imply_verified(self):
        with self.assertRaisesRegex(MissingPrimaryMethodError, "mapping"):
            require_reproducible({"missing_core":[]})
    def test_ensemble_members_are_not_repetitions(self):
        s=load_spec()
        self.assertEqual(s["confirmed_overview"]["ensemble_members"], 10)
        self.assertFalse(s["training_started"])
        self.assertFalse(s["model_implemented"])

if __name__ == "__main__":
    unittest.main()

