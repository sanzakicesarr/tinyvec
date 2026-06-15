"""Day 1 & 2 auto-checks -- run with:  pytest

These are the "extra eyes" that confirm the math in one command, on top of your
own live look via `python -m gatekeeper.distance`.
"""

import math

import pytest

from gatekeeper.distance import cosine, dot, euclidean, norm, normalize


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


# --- Day 2: normalize ---------------------------------------------------------


def test_normalize_gives_unit_length():
    # any non-zero vector, once normalized, has length 1
    assert math.isclose(norm(normalize([3, 4])), 1.0, abs_tol=1e-9)
    assert math.isclose(norm(normalize([0.2, -0.5, 0.9])), 1.0, abs_tol=1e-9)


def test_normalize_preserves_direction():
    # normalizing only changes length, not direction -> cosine with original is 1.
    # include negative components so this independently checks sign preservation.
    for v in ([3.0, 4.0], [-3.0, 4.0], [1.0, -2.0, 3.0]):
        assert math.isclose(cosine(v, normalize(v)), 1.0, abs_tol=1e-9)


def test_normalize_makes_cosine_equal_dot():
    # the key Day-2 lesson: on normalized vectors, dot == cosine.
    # Test it on several pairs, incl. negatives / different lengths / 3D,
    # not just one friendly symmetric case.
    pairs = [
        ([3.0, 4.0], [4.0, 3.0]),  # friendly, both first-quadrant
        ([1.0, -2.0, 3.0], [-4.0, 0.0, 5.0]),  # negatives, 3D, different lengths
        ([0.2, -0.5, 0.9], [0.1, 0.4, -0.7]),  # arbitrary mixed signs
    ]
    for a, b in pairs:
        assert math.isclose(dot(normalize(a), normalize(b)), cosine(a, b), abs_tol=1e-9)


def test_normalize_exact_values():
    # exact expected outputs catch sign-dropping / abs() bugs and the
    # 1D-collapse-to-±1 behavior
    assert normalize([0, -5]) == [0.0, -1.0]
    assert normalize([-3]) == [-1.0]
    assert normalize([5]) == [1.0]


def test_normalize_zero_vector_is_safe():
    # zero vector has no direction -> returned unchanged, no divide-by-zero.
    # Different lengths to catch a hard-coded output size.
    assert normalize([0, 0]) == [0.0, 0.0]
    assert normalize([0, 0, 0]) == [0.0, 0.0, 0.0]


def test_normalize_is_idempotent():
    # normalizing twice == normalizing once: a unit vector is a fixed point
    once = normalize([0.2, -0.5, 0.9])
    twice = normalize(once)
    assert all(math.isclose(x, y, abs_tol=1e-9) for x, y in zip(once, twice, strict=True))


def test_normalize_returns_new_list():
    # must not mutate the caller's input
    original = [3.0, 4.0]
    normalize(original)
    assert original == [3.0, 4.0]
