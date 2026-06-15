"""Day 6 auto-checks -- run with:  pytest

These need the optional embeddings extra (sentence-transformers + a one-time
model download), so each test skips itself when it isn't installed. That keeps CI
(which installs only [dev]) fast and dependency-light. Run locally with:

    pip install -e ".[embeddings]"

Note: the first run downloads ~90 MB (the model) and needs network once. After
that, results are deterministic for a given model version.
"""

import math

import pytest

from tinyvec.distance import dot
from tinyvec.embed import Embedder
from tinyvec.numpy_store import NumpyVectorStore


@pytest.fixture(autouse=True)
def _require_sentence_transformers():
    # Skip every test here unless the optional model library is installed. A
    # fixture (not a module-level importorskip) defers the check to call time,
    # so it never slows down or breaks plain test collection in CI.
    pytest.importorskip("sentence_transformers")


def test_encode_returns_384_dim_unit_vector():
    emb = Embedder()
    assert emb.model_name.endswith("all-MiniLM-L6-v2")  # the 384 below depends on this model
    v = emb.encode("hello world")
    assert len(v) == 384
    length = math.sqrt(sum(x * x for x in v))
    assert math.isclose(length, 1.0, abs_tol=1e-5)  # normalized to unit length


def test_encode_is_deterministic():
    # same text -> same vector (an embedding is a fixed function of its input)
    a = Embedder().encode("a quiet morning")
    b = Embedder().encode("a quiet morning")
    assert all(math.isclose(x, y, abs_tol=1e-6) for x, y in zip(a, b, strict=True))


def test_encode_many_matches_encode():
    emb = Embedder()
    texts = ["a dog", "a cat"]
    batch = emb.encode_many(texts)
    one_by_one = [emb.encode(t) for t in texts]
    assert len(batch) == 2
    for a, b in zip(batch, one_by_one, strict=True):
        assert all(math.isclose(x, y, abs_tol=1e-5) for x, y in zip(a, b, strict=True))


def test_encode_many_empty_returns_empty():
    assert Embedder().encode_many([]) == []


def test_encode_rejects_non_string():
    with pytest.raises(TypeError):
        Embedder().encode(["not", "a", "string"])  # passing a list is a common slip


def test_related_meaning_scores_higher_than_unrelated():
    # the heart of embeddings: similar MEANING -> higher cosine, even with no
    # shared words. On unit vectors, dot == cosine.
    emb = Embedder()
    dog = emb.encode("The dog barked at the mailman.")
    puppy = emb.encode("A puppy played in the garden.")
    tax = emb.encode("Quarterly tax returns are due in April.")
    assert dot(dog, puppy) > dot(dog, tax)


def test_semantic_search_finds_related_meaning():
    # the payoff of Day 6: search by MEANING, not matching words.
    emb = Embedder()
    store = NumpyVectorStore()
    sentences = {
        "dog": "The dog barked at the mailman.",
        "puppy": "A puppy played in the garden.",
        "tax": "Quarterly tax returns are due in April.",
    }
    for doc_id, text in sentences.items():
        store.add(doc_id, text, emb.encode(text))

    hits = store.search(emb.encode("a young dog outside"), k=3)
    assert hits[0].id in {"dog", "puppy"}          # closest is dog-related...
    assert hits[-1].id == "tax"                     # ...tax is the odd one out
    assert hits[0].score > hits[-1].score + 0.2     # by a clear margin, not a hair
