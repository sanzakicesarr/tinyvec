# Chapter 1 — Similarity and Search

> The first four steps of tinyvec, written up in plain language. By the end you
> can explain how semantic search works — and you have built every piece of it
> yourself, with nothing but Python and a little NumPy.

## 1. What is a vector?

A **vector** is just a list of numbers: `[3, 4]`, or `[0.91, 0.42, 0.78]`.

You can picture a list of numbers as an **arrow**. `[3, 4]` means "go 3 right and
4 up" — draw an arrow from the origin to that point. Two numbers fit on a sheet
of paper (2-D), three fit in a room (3-D). Real text vectors have hundreds of
numbers (e.g. 384 or 1536) — an arrow in 384-dimensional space. You cannot
picture that, but the maths (length, angle) works exactly the same in any number
of dimensions. That is why we build everything first in 2-D, where you can see it.

## 2. Measuring how close two vectors are

Three classic ways, all written by hand in [`tinyvec/distance.py`](../tinyvec/distance.py):

- **Dot product** — multiply matching components and add them up. Grows with length.
- **Euclidean (L2) distance** — the straight-line gap between the two arrow tips.
- **Cosine similarity** — the **angle** between the two arrows, ignoring length.

Cosine is the one semantic search leans on. It runs from `+1` (same direction),
through `0` (at right angles), to `-1` (opposite). Length is ignored, so only the
*direction* — the meaning — matters.

```python
from tinyvec.distance import cosine

cosine([1, 0], [1, 0])    # 1.0  — same direction
cosine([1, 0], [0, 1])    # 0.0  — orthogonal
cosine([1, 0], [-1, 0])   # -1.0 — opposite
```

## 3. The normalize trick

Computing cosine means dividing by both arrows' lengths every single time. But if
we first scale every vector to **length 1** ("normalize" it), that division is
already done — and cosine collapses into a plain dot product:

```python
from tinyvec.distance import normalize, dot, cosine

a, b = [3, 4], [4, 3]
cosine(a, b)                              # 0.96
dot(normalize(a), normalize(b))          # 0.96  — same number (up to float rounding)
```

This is the key move behind fast search: **normalize once when storing, then
similarity is just a dot product** — no per-query division. We cash this in below.

## 4. Searching: brute force

Now meaning becomes geometry. An **embedding model** turns each piece of text into
a vector, and similar meanings point in similar directions. "Hund" (dog) and
"Katze" (cat) end up close; "Steuererklärung" (tax return) ends up far away.

To search, we turn the query into a vector too, then compare it against every
stored vector and keep the closest. That is [`VectorStore`](../tinyvec/store.py):

```python
from tinyvec.store import VectorStore

store = VectorStore()
store.add("d1", "Hund", [0.90, 0.40])
store.add("d2", "Katze", [0.80, 0.55])
store.add("d3", "Steuererklärung", [-0.70, -0.30])

store.search([0.88, 0.45], k=2)   # -> [Hund, Katze]   (the tax return loses)
```

This is **brute force**: we literally score every record, sort by similarity, and
take the top `k`. It is the simplest *correct* search there is — and the honest
baseline that every faster index has to beat. Slow at huge scale, perfect for
learning and small data.

> The runnable demo (`python -m tinyvec.store`) uses a slightly larger set that
> also includes "Sterne", so its live output has a few more rows than the snippet
> above — same idea, more records.

## 5. Making it fast: one matrix multiply

Brute force scores one vector per loop iteration. With 1000 vectors that is 1000
trips through the Python interpreter. We can do far better by handing the whole
job to NumPy at once.

Stack all stored (normalized) vectors as the rows of one matrix `M`. Then a search
is a **single matrix–vector product** `M @ q`, which produces every cosine score in
one fast, compiled step. This is exactly where the normalize trick from §3 pays
off: because the rows and the query are unit vectors, `M @ q` *is* the list of
cosines — no per-row division needed. That is [`NumpyVectorStore`](../tinyvec/numpy_store.py):

```python
from tinyvec.numpy_store import NumpyVectorStore

store = NumpyVectorStore()          # same API as VectorStore
store.add("d1", "Hund", [0.90, 0.40])
store.add("d2", "Katze", [0.80, 0.55])
store.search([0.88, 0.45], k=2)     # same answer as VectorStore — just faster
```

The two stores are proven to return **identical** rankings and scores (the tests
check this against random data in higher dimensions). An optimization that changes
the answer would be a bug, not an optimization.

## What you can now explain

- A vector is a list of numbers you can picture as an arrow.
- Similar meaning → similar direction → small angle → high cosine similarity.
- Normalizing turns cosine into a cheap dot product.
- Search = compare a query against stored vectors and keep the closest (top-k).
- The same search becomes one matrix multiply, which is how real vector databases
  stay fast.

Next chapters go further: persisting the store to disk, why brute force eventually
breaks, and approximate indexes (IVF, HNSW) — then chunking and a full
Retrieval-Augmented-Generation loop.
