"""Day 1 auto-check -- run with:  pytest

These are the "extra eyes" that confirm the math in one command, on top of your
own live look via `python -m tinyvec.distance`.
"""

import math

import pytest

from tinyvec.distance import cosine, dot, euclidean, norm


def test_dot_known_values():
    assert dot([1, 2, 3], [4, 5, 6]) == 32  # 4 + 10 + 18
    assert dot([-1, 2], [3, -4]) == -11  # handles negatives


def test_norm_3_4_5():
    assert norm([3, 4]) == 5.0  # classic 3-4-5 triangle


def test_euclidean_basics():
    assert euclidean([3, 4], [3, 4]) == 0.0  # identical -> distance 0
    assert euclidean([1, 0], [2, 0]) == 1.0  # simple gap


def test_cosine_signs():
    assert cosine([1, 0], [1, 0]) == 1.0  # identical -> 1
    assert math.isclose(cosine([1, 0], [0, 1]), 0.0, abs_tol=1e-9)  # orthogonal -> 0
    assert cosine([1, 0], [-1, 0]) == -1.0  # opposite -> -1


def test_cosine_is_scale_invariant():
    # same direction, different length -> cosine still 1.0
    assert math.isclose(cosine([1, 0], [2, 0]), 1.0, abs_tol=1e-9)


def test_cosine_stays_in_range():
    # arbitrary 3D vectors: the result must never leave [-1, 1]
    c = cosine([0.2, -0.5, 0.9], [0.1, 0.4, -0.7])
    assert -1.0 <= c <= 1.0


def test_similarity_is_symmetric():
    a, b = [1, 2, 3], [4, 5, 6]
    assert cosine(a, b) == cosine(b, a)
    assert dot(a, b) == dot(b, a)


def test_zero_vector_is_safe():
    assert cosine([0, 0], [1, 1]) == 0.0  # one zero vector -> no divide-by-zero
    assert cosine([0, 0], [0, 0]) == 0.0  # both zero -> still safe


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        dot([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        euclidean([1, 2], [1, 2, 3])
