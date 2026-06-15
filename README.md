# gatekeeper

[![CI](https://github.com/sanzakicesarr/gatekeeper/actions/workflows/ci.yml/badge.svg)](https://github.com/sanzakicesarr/gatekeeper/actions/workflows/ci.yml)

A vector database built **from scratch** — to actually understand how embeddings,
semantic search and RAG work under the hood, one small step at a time.

> Teaching implementation, not production. Each commit is one learning step —
> from *"what is cosine similarity?"* to a full Retrieval-Augmented-Generation
> loop, and finally reading + contributing to the real thing
> ([Nextcloud's `context_chat_backend`](https://github.com/nextcloud/context_chat_backend)).

## Status

Built so far — each one a single learning step:

- **Distance metrics by hand** — dot, L2, cosine (`gatekeeper/distance.py`)
- **Normalizing** — unit vectors, so cosine becomes a plain dot product
- **Brute-force search** — `VectorStore`: add records, get the top-k most similar (`gatekeeper/store.py`)
- **Vectorized search** — `NumpyVectorStore`: the same search as one matrix multiply (`gatekeeper/numpy_store.py`)
- **Real embeddings** — turn actual sentences into vectors with a trained model (`gatekeeper/embed.py`, optional `[embeddings]` extra)

Write-up: [docs/chapter-1-similarity-and-search.md](docs/chapter-1-similarity-and-search.md).

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m gatekeeper.distance       # cosine / L2 / dot on toy vectors
python -m gatekeeper.store          # brute-force semantic search demo
python -m gatekeeper.numpy_store    # the same search, vectorized with NumPy
pytest                           # run the checks

# optional: real text embeddings (pulls in torch — heavier, ~first run downloads a model)
pip install -e ".[embeddings]"
python -m gatekeeper.embed          # semantic search on real sentences
```

## Why

Most people use a vector database as a black box. This repo opens the box: the
geometry of similarity, brute-force vs. approximate search (IVF / HNSW),
chunking, and grounding an LLM with retrieved context.
