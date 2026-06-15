"""Day 1 — the geometry of similarity, by hand.

A vector is just a point / arrow in n-dimensional space. To do semantic search
we need to measure how "close" two vectors are. Three classic ways:

- dot product    : raw alignment (grows with length)
- euclidean (L2) : straight-line distance between the two points
- cosine         : the angle between them, ignoring length (scale-invariant)

A whole vector database is built on these. So we write them by hand with the
standard library only -- no magic, no numpy yet.

Day 2 adds ``normalize``: scaling a vector to length 1. The payoff is the key
trick behind fast vector search -- once every vector is normalized, cosine
similarity is just a plain dot product (no per-query division), which later
becomes a single fast matrix multiply.

Note: these are the naive textbook formulas. For very large component
magnitudes the squares in ``norm`` / ``euclidean`` can overflow to ``inf`` (or
underflow to 0 for tiny magnitudes) -- fine for learning and normal-range data;
can be hardened later if needed (e.g. ``math.hypot``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = Sequence[float]


def _check_same_length(a: Vector, b: Vector) -> None:
    if len(a) != len(b):
        raise ValueError(f"vectors must have the same length, got {len(a)} and {len(b)}")


def dot(a: Vector, b: Vector) -> float:
    """Dot product: the sum of element-wise products."""
    _check_same_length(a, b)
    return sum(x * y for x, y in zip(a, b, strict=True))


def norm(a: Vector) -> float:
    """Length (magnitude) of a vector: sqrt(a . a)."""
    return math.sqrt(dot(a, a))


def euclidean(a: Vector, b: Vector) -> float:
    """Euclidean (L2) distance: the straight-line gap between two points."""
    _check_same_length(a, b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def cosine(a: Vector, b: Vector) -> float:
    """Cosine similarity: dot(a, b) / (|a| * |b|).

    Returns a value in [-1, 1]: 1 = same direction, 0 = orthogonal,
    -1 = opposite. Returns 0.0 if either vector is all zeros (direction
    undefined). The result is clamped to [-1, 1] so tiny float-rounding errors
    can't produce e.g. 1.0000000002 (which would later crash math.acos).
    """
    na, nb = norm(a), norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot(a, b) / (na * nb)))


def normalize(a: Vector) -> list[float]:
    """Scale a vector to unit length (length 1), keeping its direction.

    Returns a NEW list whose length is 1.0 (up to floating-point rounding).
    This is the key trick behind fast vector search: divide each vector by its
    length once, and from then on cosine similarity is just a dot product --
    ``cosine(a, b) == dot(normalize(a), normalize(b))`` for non-zero vectors
    (up to floating-point rounding) -- with no per-query division. At scale that
    becomes one matrix multiply instead of N divisions.

    A zero vector has no direction, so normalizing it is undefined (it would
    divide by zero). We return it unchanged (all zeros), mirroring how
    ``cosine`` treats the zero vector as a safe special case.
    """
    length = norm(a)
    if length == 0.0:
        return [0.0] * len(a)
    return [x / length for x in a]


def _demo() -> None:
    pairs = {
        "identical": ([1.0, 0.0], [1.0, 0.0]),
        "similar": ([1.0, 0.0], [0.9, 0.1]),
        "orthogonal": ([1.0, 0.0], [0.0, 1.0]),
        "opposite": ([1.0, 0.0], [-1.0, 0.0]),
        "same dir, 2x": ([1.0, 0.0], [2.0, 0.0]),
    }
    print(f"{'pair':<14}{'cosine':>10}{'euclidean':>12}")
    for name, (a, b) in pairs.items():
        print(f"{name:<14}{cosine(a, b):>10.3f}{euclidean(a, b):>12.3f}")
    print(
        "\nNote: 'same dir, 2x' has cosine 1.0 but euclidean 1.0 -- "
        "cosine ignores length, euclidean does not."
    )

    print("\nDay 2 -- normalize makes cosine the same as a dot product:")
    a, b = [3.0, 4.0], [4.0, 3.0]
    ua, ub = normalize(a), normalize(b)
    print(f"  normalize({a}) = {[round(x, 3) for x in ua]}  (length {norm(ua):.3f})")
    print(f"  cosine(a, b)                 = {cosine(a, b):.6f}")
    print(f"  dot(normalize(a), normalize(b)) = {dot(ua, ub):.6f}  <- same number")


if __name__ == "__main__":
    _demo()
