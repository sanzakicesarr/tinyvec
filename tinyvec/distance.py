"""Day 1 — the geometry of similarity, by hand.

A vector is just a point / arrow in n-dimensional space. To do semantic search
we need to measure how "close" two vectors are. Three classic ways:

- dot product    : raw alignment (grows with length)
- euclidean (L2) : straight-line distance between the two points
- cosine         : the angle between them, ignoring length (scale-invariant)

A whole vector database is built on these. So we write them by hand with the
standard library only -- no magic, no numpy yet.

Note: these are the naive textbook formulas. For very large component
magnitudes the squares in ``norm`` / ``euclidean`` can overflow to ``inf`` --
fine for learning and normal-range data; can be hardened later if needed.
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


if __name__ == "__main__":
    _demo()
