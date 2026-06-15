"""Day 3 auto-checks -- run with:  pytest

Verifies the brute-force top-k search in gatekeeper/store.py: that it ranks by
similarity (closest first), respects k, and handles the awkward cases (empty
store, k bigger than the store, k < 1, wrong dimension, vector copied on add).
"""

import math

import pytest

from gatekeeper.store import SearchResult, VectorStore


def _demo_store() -> VectorStore:
    # 2D fake embeddings: Hund & Katze both point up-right (pet-ish), with Hund
    # clearly closer to the query; Auto points almost straight up (different dir).
    store = VectorStore()
    store.add("d1", "Hund", [0.9, 0.4])
    store.add("d2", "Katze", [0.7, 0.6])
    store.add("d3", "Auto", [0.1, 0.95])
    return store


def test_add_grows_the_store():
    store = VectorStore()
    assert len(store) == 0
    store.add("a", "alpha", [1.0, 0.0])
    store.add("b", "beta", [0.0, 1.0])
    assert len(store) == 2


def test_search_returns_k_results_best_first():
    store = _demo_store()
    hits = store.search([0.85, 0.45], k=2)
    assert len(hits) == 2
    assert hits[0].score > hits[1].score  # strictly sorted high-to-low
    # the two pet vectors win; Auto (different direction) is left out of the top 2
    assert [h.text for h in hits] == ["Hund", "Katze"]
    assert "Auto" not in [h.text for h in hits]


def test_search_result_fields():
    store = _demo_store()
    # a SCALED copy of the Hund vector ([0.9, 0.4] * 2): cosine ignores length,
    # so same direction -> still a perfect score of 1.0
    top = store.search([1.8, 0.8], k=1)[0]
    assert isinstance(top, SearchResult)
    assert top.id == "d1"
    assert top.text == "Hund"
    assert math.isclose(top.score, 1.0, abs_tol=1e-9)


def test_search_finds_the_semantically_closest():
    store = _demo_store()
    # a query pointing almost straight up should rank Auto first, by a clear margin
    hits = store.search([0.1, 0.9], k=3)
    assert hits[0].text == "Auto"
    assert hits[0].score > hits[1].score + 0.05  # not a hair -- a real gap


def test_score_is_the_cosine_value():
    # a concrete, hand-checkable number: [1, 0] vs [1, 1] is a 45-degree angle,
    # so cosine = 1/sqrt(2) ~= 0.7071. proves search reports the *real* similarity.
    store = VectorStore()
    store.add("x", "diagonal", [1.0, 1.0])
    top = store.search([1.0, 0.0], k=1)[0]
    assert math.isclose(top.score, 1 / math.sqrt(2), abs_tol=1e-9)


def test_ties_keep_insertion_order():
    # two records with the SAME vector tie on score; the stable sort must keep
    # the order they were added (this contract matters for the Day-4 NumPy port)
    store = VectorStore()
    store.add("first", "a", [1.0, 0.0])
    store.add("second", "b", [1.0, 0.0])
    hits = store.search([1.0, 0.0], k=2)
    assert [h.id for h in hits] == ["first", "second"]


def test_k_larger_than_store_returns_all():
    store = _demo_store()
    hits = store.search([1.0, 0.0], k=99)
    assert len(hits) == 3  # only 3 records exist, no crash
    # even when k exceeds the store, the results still come back sorted best-first
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    # k exactly == number of records also returns all 3 (slice-boundary check)
    assert len(store.search([1.0, 0.0], k=3)) == 3


def test_search_empty_store_returns_empty():
    assert VectorStore().search([1.0, 0.0], k=3) == []


def test_k_must_be_at_least_one():
    store = _demo_store()
    with pytest.raises(ValueError):
        store.search([1.0, 0.0], k=0)


def test_dimension_mismatch_raises():
    store = _demo_store()  # stored vectors are 2D
    with pytest.raises(ValueError):
        # the error bubbles up from distance.dot, not from search itself
        store.search([1.0, 0.0, 0.0], k=1)  # 3D query


def test_add_copies_the_vector():
    store = VectorStore()
    v = [1.0, 0.0]
    store.add("a", "alpha", v)
    v[1] = 1.0  # change the DIRECTION of the caller's list after adding
    top = store.search([1.0, 0.0], k=1)[0]
    # if add() had kept a reference, the stored dir would now be [1, 1]
    # and the score would drop to ~0.707; copying keeps it at 1.0
    assert math.isclose(top.score, 1.0, abs_tol=1e-9)
