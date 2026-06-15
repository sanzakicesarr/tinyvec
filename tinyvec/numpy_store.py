"""Day 4 — the same search, vectorized: one matrix multiply instead of a loop.

Day 3's ``VectorStore`` scored records one Python loop iteration at a time. That
is clear but slow: 1000 vectors means 1000 trips through the interpreter. Here we
let NumPy do the work. Every stored vector becomes a row of one big matrix ``M``,
and a search is a single matrix-vector product ``M @ q`` -- NumPy computes every
score at once in fast compiled code.

The reason it is this clean is Day 2. We store *normalized* (unit-length)
vectors, and for unit vectors cosine similarity is just the dot product. So
``M @ q_unit`` yields every cosine score in one shot, with no per-row division.

Same public API as ``VectorStore`` and the same ``SearchResult`` -- so the two
stores are interchangeable, and we can prove they return identical rankings
(see the tests). Later (Day 10) we will measure just how much faster this is.
"""

from __future__ import annotations

import numpy as np

from tinyvec.distance import Vector
from tinyvec.store import SearchResult


def _unit(v: np.ndarray) -> np.ndarray:
    """Scale a NumPy vector to unit length; a zero vector stays all-zeros.

    The NumPy twin of ``distance.normalize`` -- same idea, on a whole array.
    """
    length = np.linalg.norm(v)
    if length == 0.0:
        return v
    return v / length


class NumpyVectorStore:
    """A tiny vector DB like ``VectorStore``, but each search is one NumPy matmul."""

    def __init__(self) -> None:
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._vectors: list[np.ndarray] = []  # each pre-normalized to unit length

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, id: str, text: str, vector: Vector) -> None:
        """Store one record, keeping the vector normalized to unit length.

        All vectors must share the same dimension (they have to line up as rows
        of one matrix). Adding a differently-sized vector raises ValueError.
        """
        v = np.asarray(vector, dtype=float)
        if v.ndim != 1:
            raise ValueError(f"vector must be 1-D, got shape {v.shape}")
        if self._vectors and v.shape[0] != self._vectors[0].shape[0]:
            raise ValueError(
                f"vector has dim {v.shape[0]}, store holds dim {self._vectors[0].shape[0]}"
            )
        self._ids.append(id)
        self._texts.append(text)
        self._vectors.append(_unit(v))

    def search(self, query: Vector, k: int = 3) -> list[SearchResult]:
        """Return the ``k`` records most similar to ``query``, best first.

        One matrix multiply scores every record; results match ``VectorStore``.
        Returns fewer than k (or []) for a small/empty store. A zero vector
        scores 0.0 (it sorts to a stable middle), like ``distance.cosine``.
        Raises ValueError if k < 1 or the query dimension doesn't match the
        stored vectors.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if not self._vectors:
            return []
        # rebuilt from the list each call (kept simple; Day 10 will cache it).
        matrix = np.asarray(self._vectors)  # shape (n, d)
        q = _unit(np.asarray(query, dtype=float))  # shape (d,)
        if q.ndim != 1 or q.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"query has shape {q.shape}, store holds dim {matrix.shape[1]}"
            )
        # clip to [-1, 1] so float rounding can't push a score past 1.0 -- mirrors
        # the clamp in distance.cosine, keeping the two stores exactly equivalent.
        scores = np.clip(matrix @ q, -1.0, 1.0)  # shape (n,) -- all cosines at once
        # argsort is ascending, so sort by -scores; kind="stable" keeps ties in
        # insertion order, matching Day 3's stable Python sort exactly.
        order = np.argsort(-scores, kind="stable")[:k]
        return [SearchResult(self._ids[i], self._texts[i], float(scores[i])) for i in order]


def _demo() -> None:
    from tinyvec.store import VectorStore

    data = [
        ("d1", "Hund", [0.90, 0.40]),
        ("d2", "Katze", [0.80, 0.55]),
        ("d3", "Sterne", [0.20, 0.95]),
        ("d4", "Steuererklärung", [-0.70, -0.30]),
    ]
    fast = NumpyVectorStore()
    pure = VectorStore()
    for id, text, vec in data:
        fast.add(id, text, vec)
        pure.add(id, text, vec)

    query = [0.88, 0.45]  # a "Haustier"-ish query
    print("NumPy search -- all cosine scores in ONE matrix multiply:\n")
    for rank, hit in enumerate(fast.search(query, k=len(fast)), start=1):
        print(f"  {rank}. {hit.text:<16} score {hit.score:>7.3f}")

    same = [h.id for h in fast.search(query, k=4)] == [h.id for h in pure.search(query, k=4)]
    print(f"\nSame ranking as the Day-3 pure-Python store? {same}")


if __name__ == "__main__":
    _demo()
