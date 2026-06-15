"""Checks for the pluggable embedding backends -- run with:  pytest

These never touch AWS: a FakeBedrockClient stands in for boto3's
``bedrock-runtime`` client, so the tests need no boto3, no credentials, no
network, and cost nothing. They prove the request/response wiring for Titan and
Cohere, that vectors come back unit-length, and that BedrockEmbedder is a drop-in
EmbeddingBackend that feeds the stores.
"""

import io
import json
import math
import sys
import types

import pytest

from gatekeeper.backends import _DEFAULT_BEDROCK_MODEL, BedrockEmbedder, EmbeddingBackend
from gatekeeper.embed import Embedder
from gatekeeper.numpy_store import NumpyVectorStore


class FakeBedrockClient:
    """Stands in for a boto3 bedrock-runtime client.

    ``responder(model_id, body_dict) -> response_dict`` decides what each call
    returns; every call is recorded in ``self.calls`` for assertions, along with
    the ``accept`` / ``contentType`` of the most recent call.
    """

    def __init__(self, responder):
        self.responder = responder
        self.calls = []
        self.accept = None
        self.content_type = None

    def invoke_model(self, modelId, body, accept=None, contentType=None):  # noqa: N803
        body_dict = json.loads(body)
        self.calls.append((modelId, body_dict))
        self.accept = accept
        self.content_type = contentType
        response = self.responder(modelId, body_dict)
        # boto3 returns a streaming body with .read(); io.BytesIO matches that.
        return {"body": io.BytesIO(json.dumps(response).encode())}


def _unit_len(v):
    return math.sqrt(sum(x * x for x in v))


# --- Titan -------------------------------------------------------------------


def test_titan_encode_many_loops_and_normalizes():
    # Titan embeds one text per call -> two texts == two calls. [3,4] -> [0.6,0.8].
    client = FakeBedrockClient(lambda model, body: {"embedding": [3.0, 4.0]})
    emb = BedrockEmbedder(client=client)
    vecs = emb.encode_many(["alpha", "beta"])

    assert len(vecs) == 2
    assert len(client.calls) == 2
    for v in vecs:
        assert math.isclose(_unit_len(v), 1.0, abs_tol=1e-9)
    assert math.isclose(vecs[0][0], 0.6, abs_tol=1e-9)
    assert math.isclose(vecs[0][1], 0.8, abs_tol=1e-9)
    # the request carried the text and asked Titan v2 to normalize
    _, body = client.calls[0]
    assert body["inputText"] == "alpha"
    assert body["normalize"] is True


def test_titan_passes_dimensions_when_set():
    client = FakeBedrockClient(lambda model, body: {"embedding": [1.0, 0.0, 0.0]})
    BedrockEmbedder(dimensions=256, client=client).encode("x")
    _, body = client.calls[0]
    assert body["dimensions"] == 256


def test_titan_v1_omits_v2_only_fields():
    # v1 has no normalize/dimensions params; we must not send them.
    client = FakeBedrockClient(lambda model, body: {"embedding": [1.0, 0.0]})
    BedrockEmbedder(model_id="amazon.titan-embed-text-v1", client=client).encode("x")
    _, body = client.calls[0]
    assert "normalize" not in body
    assert "dimensions" not in body


def test_titan_coerces_ints_to_float():
    # a model returning ints must still yield plain floats
    client = FakeBedrockClient(lambda model, body: {"embedding": [3, 4]})
    v = BedrockEmbedder(normalize_embeddings=False, client=client).encode("x")
    assert v == [3.0, 4.0]
    assert all(isinstance(x, float) for x in v)


def test_titan_unexpected_response_raises_valueerror():
    client = FakeBedrockClient(lambda model, body: {"oops": "no embedding here"})
    with pytest.raises(ValueError, match="Titan"):
        BedrockEmbedder(client=client).encode("x")


def test_default_model_is_titan_v2():
    assert _DEFAULT_BEDROCK_MODEL == "amazon.titan-embed-text-v2:0"
    client = FakeBedrockClient(lambda model, body: {"embedding": [1.0]})
    BedrockEmbedder(client=client).encode("x")
    assert client.calls[0][0] == "amazon.titan-embed-text-v2:0"


# --- Cohere ------------------------------------------------------------------


def test_cohere_sends_one_batch_call_and_normalizes_each_row():
    # Cohere embeds the whole batch in ONE call. [3,4]->[0.6,0.8], [0,2]->[0,1].
    client = FakeBedrockClient(
        lambda model, body: {"embeddings": {"float": [[3.0, 4.0], [0.0, 2.0]]}}
    )
    emb = BedrockEmbedder(model_id="cohere.embed-multilingual-v3", client=client)
    vecs = emb.encode_many(["a", "b"])

    assert len(vecs) == 2
    assert len(client.calls) == 1  # one batched call, not one per text
    assert math.isclose(vecs[0][0], 0.6, abs_tol=1e-9)
    assert math.isclose(vecs[0][1], 0.8, abs_tol=1e-9)
    assert vecs[1] == [0.0, 1.0]  # row order preserved + normalized
    _, body = client.calls[0]
    assert body["texts"] == ["a", "b"]
    assert body["input_type"] == "search_document"
    assert body["embedding_types"] == ["float"]


def test_cohere_normalize_false_keeps_raw_rows():
    client = FakeBedrockClient(
        lambda model, body: {"embeddings": {"float": [[3.0, 4.0], [0.0, 2.0]]}}
    )
    vecs = BedrockEmbedder(
        model_id="cohere.embed-english-v3", normalize_embeddings=False, client=client
    ).encode_many(["a", "b"])
    assert vecs == [[3.0, 4.0], [0.0, 2.0]]


def test_cohere_ignores_dimensions():
    # dimensions is a Titan-v2 param; it must not leak into a Cohere request.
    client = FakeBedrockClient(lambda model, body: {"embeddings": {"float": [[1.0, 0.0]]}})
    BedrockEmbedder(
        model_id="cohere.embed-multilingual-v3", dimensions=256, client=client
    ).encode("x")
    assert "dimensions" not in client.calls[0][1]


def test_cohere_unexpected_response_raises_valueerror():
    client = FakeBedrockClient(lambda model, body: {"embeddings": "wrong shape"})
    with pytest.raises(ValueError, match="Cohere"):
        BedrockEmbedder(model_id="cohere.embed-english-v3", client=client).encode("x")


# --- shared behaviour --------------------------------------------------------


def test_encode_single_returns_one_vector():
    client = FakeBedrockClient(lambda model, body: {"embedding": [0.0, 5.0]})
    v = BedrockEmbedder(client=client).encode("hello")
    assert isinstance(v, list)
    assert math.isclose(v[1], 1.0, abs_tol=1e-9)
    assert len(client.calls) == 1


def test_empty_list_makes_no_calls():
    client = FakeBedrockClient(lambda model, body: {"embedding": [1.0]})
    assert BedrockEmbedder(client=client).encode_many([]) == []
    assert client.calls == []


@pytest.mark.parametrize("bad", [None, 123, b"bytes", ["x"]])
def test_encode_rejects_non_string(bad):
    client = FakeBedrockClient(lambda model, body: {"embedding": [1.0]})
    with pytest.raises(TypeError):
        BedrockEmbedder(client=client).encode(bad)


def test_normalize_false_keeps_raw_values():
    client = FakeBedrockClient(lambda model, body: {"embedding": [3.0, 4.0]})
    v = BedrockEmbedder(normalize_embeddings=False, client=client).encode("x")
    assert v == [3.0, 4.0]


def test_invalid_dimensions_rejected():
    with pytest.raises(ValueError, match="dimensions"):
        BedrockEmbedder(dimensions=768)


def test_request_sets_json_accept_and_content_type():
    client = FakeBedrockClient(lambda model, body: {"embedding": [1.0]})
    BedrockEmbedder(client=client).encode("x")
    assert client.accept == "application/json"
    assert client.content_type == "application/json"


def test_lazy_client_created_once_with_region(monkeypatch):
    # No client injected -> _get_client() imports boto3 and builds one lazily.
    # We inject a fake boto3 module so this needs no real boto3 / no creds.
    created = []

    def fake_client(service, region_name=None):
        created.append((service, region_name))
        return object()

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=fake_client))

    emb = BedrockEmbedder(region_name="eu-central-1")  # client=None
    c1 = emb._get_client()
    c2 = emb._get_client()
    assert c1 is c2  # cached, not rebuilt
    assert created == [("bedrock-runtime", "eu-central-1")]  # one build, region forwarded


def test_bedrock_and_local_both_satisfy_the_protocol():
    # the whole point of the interface: both backends are interchangeable.
    client = FakeBedrockClient(lambda model, body: {"embedding": [1.0]})
    assert isinstance(BedrockEmbedder(client=client), EmbeddingBackend)
    assert isinstance(Embedder(), EmbeddingBackend)  # local, no torch import needed here


def test_vectors_drop_straight_into_a_store():
    # end-to-end: backend -> store -> search returns the nearest document.
    canned = {
        "dog": [0.9, 0.4],
        "puppy": [0.85, 0.45],
        "tax": [-0.7, -0.3],
        "young dog outside": [0.88, 0.42],
    }
    client = FakeBedrockClient(lambda model, body: {"embedding": canned[body["inputText"]]})
    emb = BedrockEmbedder(client=client)

    store = NumpyVectorStore()
    for word in ["dog", "puppy", "tax"]:
        store.add(word, word, emb.encode(word))

    hits = store.search(emb.encode("young dog outside"), k=3)
    assert hits[0].id in {"dog", "puppy"}   # a pet wins
    assert hits[-1].id == "tax"             # tax loses
