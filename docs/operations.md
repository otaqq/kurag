# kurag — operations: resources, speed, running it

What kurag uses, how fast it answers, and what to watch. The numbers are measurements on real data (Japanese and
English books and notes, 768-dimension embeddings), with the machine stated; the formulas below are fitted to them
and are estimates, not limits. How the parts work is in [architecture.md](architecture.md) and
[search.md](search.md); the settings are in [configuration.md](configuration.md).

## Where the work goes

| Request | Work |
|---|---|
| `/upsert` | one embedding call per `EMBED_BATCH` new or changed texts (unchanged text keeps its vector), a nearest-neighbour search per new chunk for its similarity edges, full-text terms, one SQLite transaction |
| `/search` vector | one embedding call for the query, then a nearest-neighbour search **in every namespace searched**, merged by cosine |
| `/search` text | full-text match and BM25 in every namespace searched; no embedding call |
| `/search` hybrid | both, fused by RRF; cosines of text-only hits from the stored copies |
| `/kg/*` | SQLite only |

The embedding API is the slowest part of adding documents and a fixed cost per vector query (27 ms p50 for one query
through a local proxy to a small model on GPU, in the measurements below). Searching many namespaces costs a search
per namespace: ask for the subtree you need (`"namespaces": ["books.math"]`) rather than everything.

## Memory

| Part | Grows with | Measured |
|---|---|---|
| kurag, `memory` backend | the vectors of the namespaces opened: about `dims` bytes per chunk (8-bit) plus fixed overhead | 320–330 MB resident with 263,000 chunks open (vectors 202 MB); 351 MiB with 285,000 |
| kurag, `txtai` or `helixdb` backend | SQLite page cache and open namespaces only | 127 MiB with 326,000 chunks in 53 namespaces, after searching all of them; 14 MiB just after start |
| kurag-txtai | its indexes: about `dims` bytes per chunk (SQ8) plus Python, numpy and faiss | 380 MiB with 326,000 chunks |
| HelixDB | its caches (it sizes them from the memory it may use) | 1.2 GiB with 47,000 chunks (vectors only), 340–410 MiB more as graph backend |

Estimate for the `memory` backend: `chunks × dims × 1 byte × 1.1 + 100 MB`. For 1 million chunks of 768 dimensions
that is about 950 MB; past a few million, use `txtai` (its IVF and memory-mapped index kinds keep memory down) on
a machine that has the room. Namespaces open on first use and stay open.

## Disk

| What | Per chunk (768 dims) | Measured |
|---|---|---|
| kurag data folder: text, full-text index, float16 vector copy (1.5 KB), similarity edges, knowledge graph | about 5.5–6 KB with typical 1,000–1,500-character chunks | 1.9 GB for 326,000 chunks in 53 namespaces |
| kurag-txtai indexes (SQ8) | about 0.8 KB | 245 MB for the same |
| sync client caches (extracted PDF/EPUB text, git trees) | the source text | 0.5 GB for 1,086 files (mostly books) |

SQLite files do not shrink when chunks are deleted; the space is reused by later upserts.

## Search latency

Measured 2026-11-07 on a 4-vCPU VM with 16 GB, `VECTOR_BACKEND=txtai`, 53 namespaces, 326,000 chunks, 10 Japanese
questions, limit 5, embedding through a local proxy:

| Scope | vector p50 | text p50 | hybrid p50 |
|---|---|---|---|
| one small namespace (272 chunks) | 31 ms | 3 ms | 33 ms |
| a subtree of 17 namespaces | 278 ms | 26 ms | 319 ms |
| all 53 namespaces | 371 ms | 57 ms | 449 ms |

On one namespace of 33,000 chunks (2-vCPU VM, 77 questions): vector 37–62 ms, hybrid 53–175 ms depending on the
backend (search.md, "Vector index backends"). `graph: n` adds a lookup of stored edges (milliseconds); `kg: n` adds
the knowledge-graph walk (tens of milliseconds in SQLite).

## Adding documents

- Throughput is set by the embedding API: kurag sends `EMBED_BATCH` texts per request and does the rest in the
  same request's transaction. Sending a chunk again with the same text costs no embedding.
- Each new chunk is linked to its `EDGE_NEIGHBOURS` nearest chunks, a vector search per chunk inside the namespace:
  noticeable on large namespaces and with external indexes. Lower `EDGE_NEIGHBOURS` for bulk loads if similarity
  edges (graph expansion, the graph view) matter less.
- kurag-sync sends at most 200 chunks per request and saves its state after each, so a stopped run goes on where it
  left off. Reading all 1,086 files of the deployment above again (no changes, text from the cache) took 2 min 57 s.
- Filling an external index from the stored vectors (after `/rebuild`, a new txtai, a copied data folder) needs no
  embedding: 3 s for 47,000 vectors into txtai, 14 s for 326,000; 50 s for 47,000 into HelixDB.

## CPU

Idle when nothing is asked. A search uses one core per request for the namespaces it touches; requests run in
parallel on a blocking pool, and each namespace is locked while it is searched or written (writes to one namespace
wait for each other; different namespaces do not). Upserts spend most CPU on edges and full-text terms.

## Watching it

- `GET /health` (no key): `{"ok": true, "namespaces": [...]}`; `kurag health` inside the container. The image and
  the chart use it.
- `GET /namespaces`: chunk counts; `kurag-sync status` adds what the client has sent.
- Logs (stderr): one line at start (address, data folder, backends) and errors that stop it. Failed requests are
  answered with their reason (`{"detail": ...}`) rather than logged; a proxy in front gives a request log.
- There is no metrics endpoint yet; container metrics (memory, CPU) and the request latency of a proxy in front
  cover the rest.

## When something is wrong

| Symptom | Likely cause |
|---|---|
| 401 on every request | wrong `X-API-Key`, or `API_KEY` changed without updating the clients |
| 502 on upsert or vector search | the embedding API is down or slow (search in `text` mode still works) |
| 400 "vectors of N dimensions, the namespace has M" | the embedding model changed; one namespace holds one model's vectors |
| the first searches after a restart are slow | namespaces open (and an external index is checked or filled) on first use |
| vector search slower than expected | the query searches many namespaces: narrow `namespaces` |
| hits from a deleted file | the sync client has not run since (deletes go with its next run), or the file is still matched by another source |
| kurag-sync sends everything again | its `state/` was lost, the source was renamed, or `[text]` changed |
| kurag-sync reads every file again | `state/files.json` was deleted or the files' times changed (a copy without preserving them) |
