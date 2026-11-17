"""Every inner split keeps its validation case out of ODE pretraining as well."""
POOLS = {'fold_a': [1, 2, 3, 4, 5, 7, 8], 'fold_b': [9, 10, 11, 12, 13, 14, 16]}


def make_split(fold, role, holdout=None):
    pool = POOLS[fold]
    outer = POOLS['fold_b' if fold == 'fold_a' else 'fold_a']
    if role == 'nested':
        if not isinstance(holdout, list) or len(holdout) != 2 or len(set(holdout)) != 2 or not set(holdout).issubset(pool):
            raise ValueError('Nested holdout must contain two distinct development cases')
        train, val, test = [c for c in pool if c not in holdout], sorted(holdout), []
    elif role in ('inner', 'smoke'):
        if holdout not in pool:
            raise ValueError('Inner holdout must belong to the development pool')
        train, val, test = [c for c in pool if c != holdout], [holdout], []
    elif role == 'final':
        train, val, test = list(pool), [], list(outer)
    else:
        raise ValueError(f'Unknown role {role}')
    return {'train': train, 'validation': val, 'test': test, 'outer_test_reserved': list(outer)}
