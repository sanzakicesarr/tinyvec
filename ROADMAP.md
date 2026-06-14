# tinyvec — Fahrplan 🗺️

Eigene Vektor-Datenbank von Grund auf bauen → verstehen wie Embeddings,
semantische Suche & RAG funktionieren → am Ende ein **echter Beitrag (PR)** zu
Nextclouds `context_chat_backend`.

**Spielregeln**
- 1 Tag = 1 Commit. Lieber „die meisten Tage" als verbissene Kette — geplante Ruhetage sind erlaubt.
- Claude baut das Stück → **du startest & testest live** → committen.
- 🔒 Keine Secrets in Git (`.gitignore` steht). Keys nur in `.env` (ignoriert).
- Conventional Commits (`feat` / `fix` / `docs` / `test` / `perf` / `ci` / `chore`).
- Umgebung: WSL2 / Ubuntu, Python-venv (wegen faiss & torch).

## Woche 1 — Mathe & Suche selbst
- [x] **Tag 1** — Cosine / L2 / Dot von Hand + Repo-Gerüst
- [ ] Tag 2 — Normalisieren: Cosine == Skalarprodukt
- [ ] Tag 3 — Brute-Force Top-k-Suche
- [ ] Tag 4 — Suche vektorisieren mit NumPy (1 Matrixmultiplikation)
- [ ] Tag 5 — CI (pytest + ruff) + erstes Kapitel-Doc
- [ ] Tag 6 — Echte Embeddings (sentence-transformers, all-MiniLM-L6-v2)
- [ ] Tag 7 — Ruhetag / Mini-Task (geplanter Puffer)

## Woche 2 — Von Liste im RAM zu echter DB
- [ ] Tag 8 — Persistenz (`.npz` + id/text-Beilage)
- [ ] Tag 9 — SQLite statt JSON-Beilage
- [ ] Tag 10 — Warum Brute-Force bricht: messen
- [ ] Tag 11 — IVF selbst bauen (k-means-Zellen)
- [ ] Tag 12 — IVF Recall/Speed-Tradeoff messen
- [ ] Tag 13 — Echtes ANN mit faiss (Flat + IVF)
- [ ] Tag 14 — HNSW + 3-Index-Vergleich

## Woche 3 — Was Profis brauchen
- [ ] Tag 15 — Metadaten-Filter (pre-/post-filter)
- [ ] Tag 16 — Chunking mit Überlappung
- [ ] Tag 17 — Volle RAG-Schleife (über **Ollama / OpenRouter**, kein teurer Key)
- [ ] Tag 18 — BRÜCKE: echtes Repo klonen, deine Module zuordnen
- [ ] Tag 19 — Read-Path im echten Repo nachverfolgen
- [ ] Tag 20 — Echtes Repo lokal bauen (für #303 nur `docker build`/`exec`) + Issue wählen
- [ ] Tag 21 — README zum Portfolio-Stück machen

## Woche 4 — Erster echter Beitrag
- [ ] Tag 22 — Auf Issue #303 die Idee ankündigen (vor dem Coden)
- [ ] Tag 23 — Fix im Fork-Branch bauen (**DCO: `git commit -s`**, Lizenz-Header!)
- [ ] Tag 24 — Fix im Container testen
- [ ] Tag 25 — **Pull Request öffnen** 🎯
- [ ] Tag 26 — Zweiten Beitrag vorbereiten (#280 / #292)

> ⏳ Falls der Maintainer auf #303 nicht antwortet (~4 Tage): Standard-Fix machen
> (Postgres `logging_collector` + `log_rotation_size`), im PR transparent erwähnen.
> Deine täglichen Commits liegen so oder so in *diesem* Repo — der Streak hängt nie an jemand anderem.
