# tinyvec

A vector database built **from scratch** — to actually understand how embeddings,
semantic search and RAG work under the hood, one small step at a time.

> Teaching implementation, not production. Each commit is one learning step —
> from *"what is cosine similarity?"* to a full Retrieval-Augmented-Generation
> loop, and finally reading + contributing to the real thing
> ([Nextcloud's `context_chat_backend`](https://github.com/nextcloud/context_chat_backend)).

## Status

**Day 1** — the geometry of similarity, by hand. See [ROADMAP.md](ROADMAP.md)
for the full plan.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python -m tinyvec.distance       # cosine / L2 / dot on toy vectors
pytest                           # run the checks
```

## Why

Most people use a vector database as a black box. This repo opens the box: the
geometry of similarity, brute-force vs. approximate search (IVF / HNSW),
chunking, and grounding an LLM with retrieved context.
