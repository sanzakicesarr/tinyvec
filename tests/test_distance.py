"""Day 1 auto-check -- run with:  pytest

These are the "extra eyes" that confirm the math in one command, on top of your
own live look via `python -m tinyvec.distance`.
"""

import math

import pytest

from tinyvec.distance import cosine, dot, euclidean, norm


def test_cosine_signs():
    assert cosine([1, 0], [1, 0]) == 1.0  # identical -> 1
    assert math.isclose(cosine([1, 0], [0, 1]), 0.0, abs_tol=1e-9)  # orthogonal -> 0
    assert cosine([1, 0], [-1, 0]) == -1.0  # opposite -> -1


def test_cosine_is_scale_invariant():
    # same direction, different length -> cosine still 1.0 ...
    assert math.isclose(cosine([1, 0], [2, 0]), 1.0, abs_tol=1e-9)
    # ... but the euclidean distance is NOT zero (that's the whole point)
    assert euclidean([1, 0], [2, 0]) == 1.0


def test_euclidean_identical_is_zero():
    assert euclidean([3, 4], [3, 4]) == 0.0


def test_norm_3_4_5():
    assert norm([3, 4]) == 5.0  # classic 3-4-5 triangle


def test_zero_vector_is_safe():
    assert cosine([0, 0], [1, 1]) == 0.0  # no crash / divide-by-zero


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        dot([1, 2], [1, 2, 3])
