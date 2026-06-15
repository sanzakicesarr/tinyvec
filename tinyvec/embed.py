"""Day 6 — real embeddings: turn actual text into vectors with a trained model.

Until now our "vectors" were hand-made toy numbers. A real *embedding model* has
learned, from huge amounts of text, to map a sentence to a vector where similar
meaning points in a similar direction. We use sentence-transformers with the
small, fast, widely-used model ``all-MiniLM-L6-v2`` (384 dimensions).

This is an OPTIONAL extra, because it pulls in torch, which is heavy. Install it:

    pip install -e ".[embeddings]"

Everything from Days 1-5 (distance, the stores) keeps working without it -- the
heavy import happens lazily, only when you actually embed something. The model is
downloaded from the Hugging Face hub on first use (needs network once), then
cached on disk by sentence-transformers.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@cache
def _load_model(name: str) -> SentenceTransformer:
    # Imported here (not at top) so the rest of tinyvec works without torch.
    # @cache keeps one model per name for the whole process (never evicts) --
    # fine for a CLI / learning tool; it would be a slow leak in a long server.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


class Embedder:
    """Turns text into vectors with a trained sentence-transformer model.

    The model is loaded once and cached, so several Embedders for the same model
    name share one loaded model.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self.model_name = model_name

    def encode(self, text: str) -> list[float]:
        """Embed one string into a unit-length vector."""
        if not isinstance(text, str):
            raise TypeError(
                f"encode() expects a single string, got {type(text).__name__}; "
                "use encode_many() for a list"
            )
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        """Embed several strings at once (faster than one call each).

        An empty list returns an empty list.
        """
        if not texts:
            return []
        model = _load_model(self.model_name)
        # normalize_embeddings=True -> unit vectors, so cosine == dot product
        # (the Day 2 / Day 4 trick); the result drops straight into our stores.
        # Values are float32 widened to float, so a re-norm reads ~1.0, not exact.
        vectors = model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()


def _demo() -> None:
    from tinyvec.numpy_store import NumpyVectorStore

    docs = [
        ("d1", "The dog barked at the mailman."),
        ("d2", "A puppy played in the garden."),
        ("d3", "The cat slept on the warm windowsill."),
        ("d4", "Quarterly tax returns are due in April."),
        ("d5", "Interest rates affect mortgage payments."),
    ]
    emb = Embedder()
    store = NumpyVectorStore()
    for doc_id, text in docs:
        store.add(doc_id, text, emb.encode(text))

    query = "a young dog outside"
    print(f"query: {query!r}\n")
    print("most similar stored sentences (real embeddings, cosine):\n")
    for rank, hit in enumerate(store.search(emb.encode(query), k=len(store)), start=1):
        print(f"  {rank}. {hit.score:.3f}  {hit.text}")
    print("\nThe dog/puppy sentences win -- the model understands meaning, not words.")


if __name__ == "__main__":
    _demo()
