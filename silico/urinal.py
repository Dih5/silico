"""Urinal protocol iteration"""

try:
    from math import prod
except ImportError:
    from functools import reduce
    import operator


    def prod(i):
        return reduce(operator.mul, i, 1)

from itertools import product
from collections import deque

import numpy as np


def distance_matrix(indices, shape, p=2):
    """
    Returns an array whose elements are the p-norm distances to the point with the given indices.

    Args:
        indices (tuple of int): The indices of the point.
        shape (tuple of int): The shape of the array.
        p (int): The p-norm to use (deafults to 2=ecuclidean)

    Returns:
        An array of the given shape whose elements are the p-norm distances to the point of the given indices.
    """
    # Create arrays of indices for each dimension.
    indices_arrays = np.indices(shape)

    # Compute the p-norm distances to the point (i_1, i_2, ..., i_n).
    diff = indices_arrays - np.array(indices).reshape(-1, *(1,) * len(shape))
    distances = np.linalg.norm(diff, ord=p, axis=0)

    return distances


def urinal_iteration(dims, p=1):
    """Yield the coordinates of urinal-like iteration"""
    n = prod(dims)
    distances = np.ones(dims) * np.inf
    to_yield = deque(list(product(*[[0, x - 1] for x in dims])))  # Corners
    i = 0
    while i < n:
        if to_yield:
            i += 1
            element = to_yield.popleft()
            yield element
            distances = np.minimum(distances, distance_matrix(element, dims, p))
        else:
            to_yield.append(np.unravel_index(np.argmax(distances, axis=None), distances.shape))
