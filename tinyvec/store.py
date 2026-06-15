"""Day 3 — brute-force top-k search: the first time tinyvec feels like a database.

A ``VectorStore`` keeps a list of records, each one an ``(id, text, vector)``.
To search, we compare the query vector against *every* stored vector with cosine
similarity, sort by score, and hand back the best ``k``.

This is "brute force": we literally check all of them, one by one. It is the
simplest *correct* search there is -- and the honest baseline that every faster
index later in the curriculum (NumPy batch, IVF, HNSW) has to beat. Slow at huge
scale, perfect for learning and small data.

Note: we score with ``cosine`` (which ignores length), so the stored vectors and
the query need not share a length-scale. Later we will ``normalize`` on add and
search with the cheaper ``dot`` -- that is the payoff of Day 2.
"""

from __future__ import annotations

from dataclasses import dataclass

from tinyvec.distance import Vector, cosine


@dataclass(frozen=True)
class SearchResult:
    """One hit: which record matched, and how similar it was (cosine score)."""

    id: str
    text: str
    score: float


@dataclass
class _Record:
    """One stored item. Internal -- callers only ever see SearchResult."""

    id: str
    text: str
    vector: list[float]


class VectorStore:
    """A tiny in-memory vector database: add records, then search by similarity."""

    def __init__(self) -> None:
        self._records: list[_Record] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, id: str, text: str, vector: Vector) -> None:
        """Store one record.

        We copy the vector into a plain list, so a later change to the caller's
        own list cannot silently mutate what we have stored. Ids are not required
        to be unique yet: a repeated id is stored as a separate record (we will
        revisit upsert / uniqueness when persistence lands on Day 8/9).
        """
        self._records.append(_Record(id, text, list(vector)))

    def search(self, query: Vector, k: int = 3) -> list[SearchResult]:
        """Return the ``k`` records most similar to ``query``, best first.

        Brute force: score every record with cosine, sort high-to-low, take k.
        Returns fewer than k if the store holds fewer records (and [] if empty).
        Ties keep insertion order (the sort is stable). A zero vector scores 0.0
        rather than raising (see ``cosine``), so it sorts to the middle.
        Raises ValueError if k < 1, or if the query length doesn't match the
        stored vectors (the latter bubbles up from cosine -> dot).
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        scored = [
            SearchResult(r.id, r.text, cosine(query, r.vector)) for r in self._records
        ]
        # stable sort: equal scores keep insertion order (matters for the Day-4
        # NumPy port, where argsort needs kind="stable" to reproduce this).
        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:k]


def _demo() -> None:
    # Tiny hand-made "embeddings" in 2D, so the geometry stays visible.
    store = VectorStore()
    store.add("d1", "Hund", [0.90, 0.40])
    store.add("d2", "Katze", [0.80, 0.55])
    store.add("d3", "Sterne", [0.20, 0.95])
    store.add("d4", "Steuererklärung", [-0.70, -0.30])

    query = [0.88, 0.45]  # a "Haustier"-ish query, pointing near Hund/Katze
    print("query ~ 'Haustier'  ->  ranked by cosine similarity:\n")
    for rank, hit in enumerate(store.search(query, k=len(store)), start=1):
        print(f"  {rank}. {hit.text:<16} score {hit.score:>7.3f}")
    print("\nHund/Katze win (small angle); Steuererklärung loses (points the other way).")


if __name__ == "__main__":
    _demo()
