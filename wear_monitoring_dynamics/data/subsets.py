"""Explicit training-record inclusion, applied before any fitted preprocessing."""
import numpy as np


def restrict_training_records(table, train_cases, records=None):
    if records is None:
        return table
    allowed = {tuple(map(int, row)) for row in records}
    available = {(int(c), int(r)) for c, r in zip(table['case'], table['run']) if c in train_cases}
    if not allowed or not allowed <= available or len(allowed) != len(records):
        raise ValueError('Training record list is empty, duplicated or outside the training pool')
    if {c for c, _ in allowed} != set(train_cases):
        raise ValueError('Every training case must retain observations')
    keep = np.array([c not in train_cases or (int(c), int(r)) in allowed
                     for c, r in zip(table['case'], table['run'])])
    return {key: value[keep] for key, value in table.items()}
