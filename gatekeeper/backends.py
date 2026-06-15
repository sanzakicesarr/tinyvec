"""Pluggable embedding backends: swap the model behind one small interface.

gatekeeper computes similarity over vectors; *where those vectors come from* is a
separate concern. Day 6's :class:`~gatekeeper.embed.Embedder` makes them locally
with sentence-transformers. For a real product you also want managed options --
so here we name one interface, :class:`EmbeddingBackend`, and add a second
implementation that calls **AWS Bedrock**.

The same store works with either:

* **local** (``Embedder``) -- offline, zero cost, you own the weights;
* **Bedrock** (``BedrockEmbedder``) -- managed, pay-per-token, nothing to host.

That is the shape a customer would either self-host or rent -- and it mirrors how
Nextcloud's ``context_chat_backend`` keeps its embedding provider pluggable.

``boto3`` is an OPTIONAL extra -- it is only needed to reach the real service::

    pip install -e ".[bedrock]"

The import is lazy, so importing gatekeeper -- and the tests, which inject a fake
client -- never require boto3 or AWS credentials. gatekeeper itself never reads,
stores, or logs your credentials: boto3 picks them up from your normal AWS
environment (``~/.aws``, env vars, or an instance/role) exactly as usual.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

from gatekeeper.distance import normalize

_DEFAULT_BEDROCK_MODEL = "amazon.titan-embed-text-v2:0"
_TITAN_V2_DIMENSIONS = (256, 512, 1024)  # the only sizes Titan v2 accepts


# runtime_checkable lets isinstance() confirm a backend looks right at runtime.
# Caveat: it checks method *names* only -- not signatures or return types.
@runtime_checkable
class EmbeddingBackend(Protocol):
    """Anything that turns text into vectors the stores can hold.

    Both :class:`~gatekeeper.embed.Embedder` (local) and :class:`BedrockEmbedder`
    (managed) satisfy this. Code that only needs "text -> vectors" can depend on
    this Protocol and stay agnostic to which backend produced them.
    """

    def encode(self, text: str) -> list[float]:
        """Embed one string into a vector."""
        ...

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        """Embed several strings at once; an empty list returns an empty list."""
        ...


class BedrockEmbedder:
    """An :class:`EmbeddingBackend` backed by AWS Bedrock (managed, pay-per-token).

    Supports Amazon Titan (e.g. ``amazon.titan-embed-text-v2:0``) and Cohere
    (e.g. ``cohere.embed-multilingual-v3``) embedding models. With the default
    ``normalize_embeddings=True`` vectors come back unit-length -- like
    ``Embedder`` -- so they drop straight into the stores and "cosine == dot"
    holds. (A genuine all-zero vector stays all-zero, same as ``distance.normalize``.)

    Pass ``client`` to inject a boto3 ``bedrock-runtime`` client (the tests inject
    a fake one). Leave it ``None`` and a real client is created lazily on first
    use, from your normal AWS credentials and region. No keys are ever read or
    stored by gatekeeper itself.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_BEDROCK_MODEL,
        *,
        region_name: str | None = None,
        dimensions: int | None = None,
        normalize_embeddings: bool = True,
        client: Any | None = None,
    ) -> None:
        if dimensions is not None and dimensions not in _TITAN_V2_DIMENSIONS:
            raise ValueError(
                f"dimensions must be one of {_TITAN_V2_DIMENSIONS} (Titan v2), got {dimensions}"
            )
        self.model_id = model_id
        self.region_name = region_name
        self.dimensions = dimensions
        self.normalize_embeddings = normalize_embeddings
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            # lazy: boto3 is only needed to talk to the real service.
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._client

    def _invoke(self, body: dict[str, Any]) -> dict[str, Any]:
        response = self._get_client().invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            accept="application/json",
            contentType="application/json",
        )
        return json.loads(response["body"].read())

    def encode(self, text: str) -> list[float]:
        """Embed one string into a vector."""
        if not isinstance(text, str):
            raise TypeError(
                f"encode() expects a single string, got {type(text).__name__}; "
                "use encode_many() for a list"
            )
        return self.encode_many([text])[0]

    def encode_many(self, texts: list[str]) -> list[list[float]]:
        """Embed several strings; an empty list returns an empty list."""
        if not texts:
            return []
        if self.model_id.startswith("cohere."):
            vectors = self._encode_cohere(texts)
        else:
            vectors = self._encode_titan(texts)
        if self.normalize_embeddings:
            # reuse Day 2's normalize so a Bedrock vector behaves like a local one
            vectors = [normalize(v) for v in vectors]
        return vectors

    def _encode_titan(self, texts: list[str]) -> list[list[float]]:
        # Titan embeds ONE text per call, so we loop over the batch.
        is_v2 = "titan-embed-text-v2" in self.model_id
        out: list[list[float]] = []
        for text in texts:
            body: dict[str, Any] = {"inputText": text}
            if is_v2:
                body["normalize"] = True
                if self.dimensions is not None:
                    body["dimensions"] = self.dimensions
            data = self._invoke(body)
            try:
                vector = data["embedding"]
            except (KeyError, TypeError) as e:
                raise ValueError(
                    f"unexpected Titan response for {self.model_id}: missing 'embedding'"
                ) from e
            out.append([float(x) for x in vector])
        return out

    def _encode_cohere(self, texts: list[str]) -> list[list[float]]:
        # Cohere embeds a whole batch in a single call. The response is keyed by
        # type (embeddings["float"]) ONLY because we set embedding_types below;
        # drop that field and the response becomes a bare list instead.
        body: dict[str, Any] = {
            "texts": list(texts),
            "input_type": "search_document",
            "embedding_types": ["float"],
        }
        data = self._invoke(body)
        try:
            rows = data["embeddings"]["float"]
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"unexpected Cohere response for {self.model_id}: missing embeddings['float']"
            ) from e
        return [[float(x) for x in row] for row in rows]


def _demo() -> None:
    # Offline illustration: a fake client stands in for Bedrock so this runs with
    # no AWS account, no network, no cost -- it shows the *pluggability*, not a
    # real call. Swap BedrockEmbedder(client=...) for a real client in production.
    from gatekeeper.numpy_store import NumpyVectorStore

    canned = {
        "The dog barked.": [0.9, 0.4],
        "A puppy played.": [0.8, 0.55],
        "Tax returns are due.": [-0.7, -0.3],
        "a young dog": [0.88, 0.45],
    }

    class _Bytes:
        def __init__(self, s: str) -> None:
            self._b = s.encode()

        def read(self) -> bytes:
            return self._b

    class _FakeBedrock:
        # Titan-shaped fake; the demo uses the default Titan model.
        def invoke_model(self, modelId, body, **_):  # noqa: N803 (boto3's kwarg name)
            text = json.loads(body)["inputText"]
            return {"body": _Bytes(json.dumps({"embedding": canned[text]}))}

    emb = BedrockEmbedder(client=_FakeBedrock())
    store = NumpyVectorStore()
    for text in ["The dog barked.", "A puppy played.", "Tax returns are due."]:
        store.add(text, text, emb.encode(text))

    print("BedrockEmbedder -> NumpyVectorStore (offline fake client):\n")
    for rank, hit in enumerate(store.search(emb.encode("a young dog"), k=3), start=1):
        print(f"  {rank}. {hit.score:.3f}  {hit.text}")
    print("\nSame store, different backend -- that is the pluggable part.")


if __name__ == "__main__":
    _demo()
