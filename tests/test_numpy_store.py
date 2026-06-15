"""Day 4 auto-checks -- run with:  pytest

Verifies the vectorized NumPy search in tinyvec/numpy_store.py. The headline
test is test_matches_the_pure_python_store: the fast store must return the SAME
ranking and (near-)same scores as Day 3's VectorStore -- an optimization that
changes the answer is a bug, not an optimization.
"""

import math

import numpy as np
import pytest

from tinyvec.numpy_store import NumpyVectorStore
from tinyvec.store import SearchResult, VectorStore


def _demo_store() -> NumpyVectorStore:
    store = NumpyVectorStore()
    store.add("d1", "Hund", [0.9, 0.4])
    store.add("d2", "Katze", [0.7, 0.6])
    store.add("d3", "Auto", [0.1, 0.95])
    return store


def test_add_grows_the_store():
    store = NumpyVectorStore()
    assert len(store) == 0
    store.add("a", "alpha", [1.0, 0.0])
    store.add("b", "beta", [0.0, 1.0])
    assert len(store) == 2


def test_search_returns_k_results_best_first():
    store = _demo_store()
    hits = store.search([0.85, 0.45], k=2)
    assert len(hits) == 2
    assert hits[0].score > hits[1].score
    assert [h.text for h in hits] == ["Hund", "Katze"]


def test_matches_the_pure_python_store():
    # the whole point of Day 4: same answers as Day 3, just computed faster.
    data = [
        ("d1", "Hund", [0.9, 0.4]),
        ("d2", "Katze", [0.7, 0.6]),
        ("d3", "Auto", [0.1, 0.95]),
        ("d4", "Steuer", [-0.7, -0.3]),
        ("d5", "Wolke", [0.3, 0.8]),
    ]
    fast = NumpyVectorStore()
    pure = VectorStore()
    for id, text, vec in data:
        fast.add(id, text, vec)
        pure.add(id, text, vec)

    # the [0, 0] query makes every score 0.0 -> a full tie, which proves BOTH
    # stores resolve ties the same way (stable, insertion order).
    for query in ([0.88, 0.45], [0.1, 0.9], [-0.5, -0.5], [1.0, 0.0], [0.0, 0.0]):
        fast_hits = fast.search(query, k=5)
        pure_hits = pure.search(query, k=5)
        # identical ranking (ids in the same order)...
        assert [h.id for h in fast_hits] == [h.id for h in pure_hits]
        # ...and the same scores (cosine, up to floating-point rounding)
        for f, p in zip(fast_hits, pure_hits, strict=True):
            assert math.isclose(f.score, p.score, abs_tol=1e-9)


def test_matches_pure_store_on_random_high_dim_data():
    # the real proof that "one matmul = same answer": 20 records in 8-D, several
    # random queries, checked against the pure-Python store. seeded -> reproducible.
    rng = np.random.default_rng(0)
    points = rng.standard_normal((20, 8))
    fast = NumpyVectorStore()
    pure = VectorStore()
    for i, p in enumerate(points):
        fast.add(f"d{i}", f"text {i}", p.tolist())
        pure.add(f"d{i}", f"text {i}", p.tolist())

    for _ in range(5):
        query = rng.standard_normal(8).tolist()
        fast_hits = fast.search(query, k=20)
        pure_hits = pure.search(query, k=20)
        assert [h.id for h in fast_hits] == [h.id for h in pure_hits]
        for f, p in zip(fast_hits, pure_hits, strict=True):
            assert math.isclose(f.score, p.score, abs_tol=1e-9)


def test_search_result_fields_and_types():
    store = _demo_store()
    top = store.search([1.8, 0.8], k=1)[0]  # scaled Hund vector -> still score 1.0
    assert isinstance(top, SearchResult)
    assert top.id == "d1"
    assert top.text == "Hund"
    assert isinstance(top.score, float)  # plain float, not numpy.float64
    assert math.isclose(top.score, 1.0, abs_tol=1e-9)
    assert top.score <= 1.0  # clipped: never exceeds 1.0, even with float rounding


def test_score_is_the_cosine_value():
    # [1, 0] vs [1, 1] is a 45-degree angle -> cosine = 1/sqrt(2) ~= 0.7071
    store = NumpyVectorStore()
    store.add("x", "diagonal", [1.0, 1.0])
    top = store.search([1.0, 0.0], k=1)[0]
    assert math.isclose(top.score, 1 / math.sqrt(2), abs_tol=1e-9)


def test_accepts_float32_input():
    # a float32 numpy vector is upcast to float64 internally; result still right
    store = NumpyVectorStore()
    store.add("x", "diagonal", np.array([1.0, 1.0], dtype=np.float32))
    top = store.search(np.array([1.0, 0.0], dtype=np.float32), k=1)[0]
    assert isinstance(top.score, float)
    assert math.isclose(top.score, 1 / math.sqrt(2), abs_tol=1e-6)


def test_ties_keep_insertion_order():
    # equal scores must keep insertion order (argsort kind="stable")
    store = NumpyVectorStore()
    store.add("first", "a", [1.0, 0.0])
    store.add("second", "b", [1.0, 0.0])
    hits = store.search([1.0, 0.0], k=2)
    assert [h.id for h in hits] == ["first", "second"]


def test_k_larger_than_store_returns_all_sorted():
    store = _demo_store()
    hits = store.search([1.0, 0.0], k=99)
    assert len(hits) == 3  # only 3 records, no crash
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_empty_store_returns_empty():
    assert NumpyVectorStore().search([1.0, 0.0], k=3) == []


def test_k_must_be_at_least_one():
    with pytest.raises(ValueError):
        _demo_store().search([1.0, 0.0], k=0)


def test_query_dimension_mismatch_raises():
    store = _demo_store()  # stored vectors are 2D
    with pytest.raises(ValueError):
        store.search([1.0, 0.0, 0.0], k=1)  # 3D query


def test_add_dimension_mismatch_raises():
    store = _demo_store()  # 2D
    with pytest.raises(ValueError):
        store.add("bad", "three-d", [1.0, 0.0, 0.0])


# --- Day 8: persistence (save / load) -----------------------------------------


def test_save_and_load_round_trip(tmp_path):
    store = _demo_store()
    path = tmp_path / "store.npz"
    store.save(path)

    loaded = NumpyVectorStore.load(path)
    assert len(loaded) == len(store)
    # a loaded store searches identically to the one we saved
    q = [0.85, 0.45]
    before, after = store.search(q, k=3), loaded.search(q, k=3)
    assert [h.id for h in after] == [h.id for h in before]
    for a, b in zip(after, before, strict=True):
        assert math.isclose(a.score, b.score, abs_tol=1e-9)


def test_save_and_load_empty_store(tmp_path):
    path = tmp_path / "empty.npz"
    NumpyVectorStore().save(path)
    loaded = NumpyVectorStore.load(path)
    assert len(loaded) == 0
    assert loaded.search([1.0, 0.0], k=3) == []


def test_loaded_store_preserves_fields_as_plain_str(tmp_path):
    path = tmp_path / "s.npz"
    _demo_store().save(path)
    top = NumpyVectorStore.load(path).search([0.9, 0.4], k=1)[0]
    assert top.id == "d1"
    assert top.text == "Hund"
    assert isinstance(top.id, str)    # plain python str...
    assert isinstance(top.text, str)  # ...not numpy.str_


def test_round_trip_preserves_umlauts_and_vector_values(tmp_path):
    store = NumpyVectorStore()
    store.add("d1", "Steuererklärung", [0.6, 0.8])  # non-ASCII text
    path = tmp_path / "umlaut.npz"
    store.save(path)
    assert path.exists()  # the file really got written

    loaded = NumpyVectorStore.load(path)
    assert loaded.search([0.6, 0.8], k=1)[0].text == "Steuererklärung"  # umlaut survived
    # the stored (normalized) vectors are bit-for-bit the same
    assert np.allclose(np.asarray(loaded._vectors), np.asarray(store._vectors))


def test_can_add_to_a_loaded_store(tmp_path):
    path = tmp_path / "s.npz"
    _demo_store().save(path)
    loaded = NumpyVectorStore.load(path)
    loaded.add("new", "Vogel", [0.95, 0.3])  # adding after load must work
    assert len(loaded) == 4
    assert loaded.search([0.95, 0.3], k=1)[0].id == "new"


def test_save_load_without_npz_suffix(tmp_path):
    # save/load stay symmetric even if you omit the .npz extension
    store = _demo_store()
    base = tmp_path / "mystore"
    store.save(base)
    loaded = NumpyVectorStore.load(base)
    assert len(loaded) == len(store)
