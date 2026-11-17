import numpy as np

import unittest

from wear_monitoring_dynamics.data.subsets import restrict_training_records

from wear_monitoring_dynamics.data.preprocessing import fit_scalers, forecasting_pairs





def fixture_table():

    return dict(case=np.repeat([1, 2], 6), run=np.tile(np.arange(6), 2),

                time=np.tile(np.arange(6.), 2), VB=np.tile(np.arange(6.) / 10, 2),

                DOC=np.ones(12), feed=np.ones(12), material=np.ones(12),

                sensor=np.arange(24.).reshape(12, 2))





def test_excluded_training_rows_cannot_affect_scalers_or_pairs():

    original = fixture_table()

    keep = [[1, i] for i in range(4)]

    subset = restrict_training_records(original, [1], keep)

    changed = {k: v.copy() for k, v in original.items()}

    changed['VB'][4:6] = 10000

    changed['sensor'][4:6] = 1e9

    other = restrict_training_records(changed, [1], keep)

    assert fit_scalers(subset, [1]) == fit_scalers(other, [1])

    for key in original:

        np.testing.assert_array_equal(subset[key][subset['case'] == 2], original[key][original['case'] == 2])

    pairs = forecasting_pairs(subset, [1], [1, 2, 3])

    assert set(subset['run'][pairs[:, :2]].ravel()) <= {0, 1, 2, 3}

    assert restrict_training_records(original, [1], None) is original





def test_invalid_training_membership_rejected():

    for records in ([[2, 0]], [[1, 0], [1, 0]], []):

        with unittest.TestCase().assertRaises(ValueError):

            restrict_training_records(fixture_table(), [1], records)





class TestTrainingSubsets(unittest.TestCase):

    def test_exclusion(self):

        test_excluded_training_rows_cannot_affect_scalers_or_pairs()



    def test_invalid(self):

        test_invalid_training_membership_rejected()

