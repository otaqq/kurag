# kurag — HTTP API (version 1)

Every route but `/health` (and the graph view under `/ui`, which has its own login) needs the header
`X-API-Key: <API_KEY>`. Bodies are JSON (`Content-Type: application/json`). How the scores are made is in
[search.md](search.md); the settings are in [configuration.md](configuration.md).

| Route | Does |
|---|---|
| `GET /health` | liveness, and the namespaces on disk |
| `GET /namespaces` | chunks per namespace |
| `POST /upsert` | add or replace chunks (embeds new text) |
| `POST /delete` | delete chunks by id prefix |
| `POST /search` | vector, text or hybrid search, with graph expansion and the knowledge graph |
| `POST /kg/upsert` | add entities, relations and mentions; mark chunks extracted |
| `POST /kg/todo` | chunks not extracted yet |
| `POST /kg/delete` | delete what was learned from chunks, by id prefix |
| `POST /kg/search` | entities named in a query and the relations around them |
| `POST /rebuild` | rebuild the vector index of one or all namespaces |

## Namespaces and ids

A **namespace** is one index (one folder with one SQLite database). Names are lower-case dot paths:
`^[a-z][a-z0-9-]*(\.[a-z0-9][a-z0-9-]*)*$`, e.g. `books.math.algebra`. Searching `books` covers `books` and every
namespace under it (`books.math`, `books.math.algebra`, ...); `exclude` removes namespaces the same way. A namespace
is created by its first upsert and removed when a delete leaves it empty.

A **chunk id** is chosen by the client. kurag-sync uses `<source>:<path>#<section>#<n>`, so that every chunk of a
file (`<source>:<path>#`) or of a section (`<source>:<path>#<section>#`) can be deleted by one prefix. Ids are
compared as UTF-8 bytes; any string works, the prefix rule is what matters.

## GET /health

```json
{"ok": true, "namespaces": ["books.math", "notes"]}
```

No key needed. `kurag health` (the binary) asks this route on its own port; the image uses it as health check.

## GET /namespaces

```json
{"books.math": 26547, "notes": 272}
```

## POST /upsert

```json
{"namespace": "notes",
 "documents": [{"id": "notes:a.md#Intro#0", "text": "...", "source": "a.md"}]}
```

Answer: `{"namespace": "notes", "count": 273}` (chunks in the namespace afterwards).

- A chunk with an existing id is replaced. Text that did not change keeps its stored vector (no embedding call);
  only new or changed text is sent to the embedding API, in batches of `EMBED_BATCH`.
- Each new chunk gets similarity edges to its `EDGE_NEIGHBOURS` nearest chunks with a cosine of at least
  `EDGE_MIN` (search.md, "Similarity graph").
- An empty `documents` list only reports the count.
- `source` is free text shown with hits (a path, a URL).
- A replaced chunk gets a new row id: the knowledge graph treats it as not extracted again.

## POST /delete

```json
{"namespace": "notes", "prefix": "notes:a.md#"}
{"namespace": "notes", "prefixes": ["notes:a.md#", "notes:b.md#"]}
```

Answer: `{"namespace": "notes", "deleted": 12}`, plus `"removed": true` when the namespace became empty and was
removed. Deletes reach the chunk, its vector, its full-text entry, its edges and its knowledge-graph mentions. An
empty prefix deletes nothing (use a real prefix to delete everything of a source).

## POST /search

```json
{"query": "where are chunks stored?",
 "namespaces": ["notes"], "exclude": ["notes.private"],
 "mode": "hybrid", "limit": 5, "graph": 2, "min_weight": 0.75, "kg": 0}
```

| Field | Default | Range | Meaning |
|---|---|---|---|
| `query` | (required) | | the question |
| `namespaces` | all | | namespaces to search, each with everything under it |
| `exclude` | none | | namespaces (and everything under them) to leave out |
| `mode` | `vector` | `vector`, `text`, `hybrid` | which search |
| `limit` | 5 | 1..1000 | hits to return (before graph expansion) |
| `graph` | 0 | 0..100 | add up to this many chunks linked to the hits by similarity edges |
| `min_weight` | 0.75 | | only follow edges at least this similar |
| `kg` | 0 | 0..3 (hybrid only) | walk this many steps of the knowledge graph from the entities named in the query |

Answer: a list of hits, best first.

```json
[{"namespace": "notes", "id": "notes:a.md#Intro#0", "source": "a.md", "text": "...",
  "score": 0.0328, "vector_score": 0.71, "text_score": 1.0002},
 {"namespace": "notes", "id": "notes:b.md#Setup#1", "source": "b.md", "text": "...",
  "score": 0.62, "via": "notes:a.md#Intro#0"}]
```

- `score`: in `vector` mode the cosine similarity; in `text` mode coverage plus BM25 / 10000 (the share of query
  terms found decides first); in `hybrid` mode the reciprocal rank fusion score (search.md).
- `vector_score`, `text_score`: in `hybrid` mode, the two scores the fusion started from (`vector_score` is computed
  also for hits only the text search found).
- `via`: a chunk added by graph expansion, and the hit it is linked to.
- Hits of several namespaces are merged by score; all namespaces use the same embedding model, so cosines compare.
- A query that finds nothing returns `[]`; asking for namespaces that do not exist is not an error.

## POST /kg/upsert

What an extractor found in some chunks:

```json
{"namespace": "notes",
 "entities": [{"name": "kurag", "type": "software", "description": "a retrieval server"}],
 "relations": [{"source": "kurag", "target": "SQLite", "type": "uses", "description": "...", "weight": 1.0,
                "chunk": "notes:a.md#Intro#0"}],
 "mentions": [{"entity": "kurag", "chunk": "notes:a.md#Intro#0"}],
 "done": ["notes:a.md#Intro#0"]}
```

Answer: `{"namespace": "notes", "entities": 120, "relations": 300, "mentions": 410, "done": 57}` (totals).

- `done`: the chunks this is the whole extraction of. What was learned from them before is replaced, and they
  leave the to-do list. Chunks with nothing to learn are sent in `done` with no entities.
- A relation's `chunk` is where it was learned: deleting that chunk deletes the relation. Entities no relation or
  mention refers to any more are removed.
- Entity names are matched exactly (case and width as given). A non-empty `type` or `description` sent again
  replaces the stored one; an empty one keeps it.

## POST /kg/todo

```json
{"namespace": "notes", "limit": 10}
```

Answer: `{"namespace": "notes", "chunks": [{"id": "...", "source": "...", "text": "..."}], "remaining": 9616}`.
`limit` 1..1000. A namespace that does not exist is 404.

## POST /kg/delete

`{"namespace": "notes", "prefix": "notes:a.md#"}` (or `prefixes`). Deletes relations and mentions learned from those
chunks and the entities left without any. Answer: `{"namespace": ..., "relations": n, "mentions": n, "entities": n}`.

## POST /kg/search

```json
{"query": "what does kurag use?", "namespaces": ["notes"], "limit": 10, "depth": 1}
```

`limit` 1..200 entities found from the query (same name first, then full text on names and descriptions), `depth`
0..3 steps of relations around them. Answer: `{"entities": [{"namespace", "name", "type", "description",
"chunks"}], "relations": [{"namespace", "source", "target", "type", "description", "weight", "chunks"}]}`, with
`chunks` the ids an entity is mentioned in (up to 20) or a relation was learned from.

## POST /rebuild

`{"namespace": "notes"}`, or `{}` for every namespace. Learns the 8-bit vector ranges again and refills an external
vector index (txtai, HelixDB) from the stored float16 vectors; no embedding calls. Answer: `{"notes": 273}`
(vectors per namespace). Needed only after a large change of content. With the in-memory index kurag learns the
ranges again by itself when a namespace's vector count has doubled; an external index is refilled when it differs
from the database (after a failed write, or a count that does not match).

## Errors

Errors are `{"detail": "message"}` with:

| Status | When |
|---|---|
| 400 | bad namespace name; vectors of other dimensions than the namespace's (another embedding model) |
| 401 | missing or wrong `X-API-Key` |
| 404 | `/kg/todo` or `/rebuild` of a namespace that does not exist |
| 415 | no `Content-Type: application/json` |
| 422 | a value out of range (`limit`, `graph`, `kg`, `depth`), `kg` without `mode: hybrid`, or JSON with missing or wrong fields (400 when it does not parse) |
| 500 | the database or the files failed |
| 502 | the embedding API or the external vector index failed (the request can be repeated) |

## Graph view

With `UI_PASSWORD` set, `/ui` serves a read-only browser view (Basic login with `UI_USER` / `UI_PASSWORD`): search,
hub sections, sections linked to one, the text of a section. Its JSON routes (`/ui/namespaces`, `/ui/search/<ns>`,
`/ui/hubs/<ns>`, `/ui/around/<ns>`, `/ui/section/<ns>`) are for that page and may change between versions.
