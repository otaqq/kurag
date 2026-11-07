# kurag

A retrieval server for RAG: vector, full-text and hybrid search, a similarity graph and a knowledge graph behind
one HTTP API, with a sync client that keeps it in step with your files. One static binary, SQLite inside, small
memory. The name is *kura* (蔵, storehouse) + *RAG*.

Status: v0.4.0. The HTTP API (version 1) is stable; a configuration file and more engines are still to come.
Tested with Japanese and English text, hundreds of thousands of chunks.

## What it does

- **Namespaces**: one index per shelf (`books.math.algebra`); search one, several, or a subtree.
- **Search modes**: `vector` (cosine over 8-bit vectors), `text` (words plus CJK bigrams, coverage + BM25),
  `hybrid` (both, fused by RRF), and `graph` expansion along similarity edges between chunks.
- **Knowledge graph**: entities, relations and mentions sent by clients (`/kg/upsert`), a to-do list of chunks not
  extracted yet (`/kg/todo`), `/kg/search`, and `kg: n` in hybrid search to bring in chunks the graph reaches.
- **Replaceable engines**: chunks, full text and graphs live in SQLite; the vector index can be kept in memory or
  handed to txtai (faiss) or HelixDB; graphs can be walked in SQLite or in HelixDB.
- **Graph view**: `/ui`, a read-only browser view of sections and their links.
- **kurag-sync**: reads git repositories, folders and shelves of books (Markdown, text, PDF via `pdftotext`, EPUB),
  splits them into sections, pages and chapters, and sends only what changed; also runs knowledge-graph extraction
  with any OpenAI-compatible LLM API.

## Quick start

Build the image and run it with an OpenAI-compatible embedding API (Ollama, LiteLLM, vLLM, ...):

```sh
docker build -f docker/Dockerfile -t kurag .
docker run -d -p 9650:9650 -v kurag-data:/data \
  -e API_KEY=change-me \
  -e EMBED_URL=http://embeddings.example:11434/v1 \
  -e EMBED_MODEL=embeddinggemma \
  kurag
```

Add and search:

```sh
curl -s localhost:9650/upsert -H 'X-API-Key: change-me' -H 'Content-Type: application/json' -d '{
  "namespace": "notes",
  "documents": [{"id": "notes:a.md#Intro#0", "text": "kurag keeps chunks in SQLite.", "source": "a.md"}]}'
curl -s localhost:9650/search -H 'X-API-Key: change-me' -H 'Content-Type: application/json' -d '{
  "query": "where are chunks stored?", "namespaces": ["notes"], "mode": "hybrid", "limit": 5}'
```

Chunk ids are `<source>:<path>#<section>#<n>`; `/delete` removes every chunk whose id starts with a prefix.

The binary alone: `docker build -f docker/Dockerfile --target bin --output type=local,dest=out .` writes
`out/kurag` and `out/kurag-sync`; `dist/systemd/kurag.service` is an example unit.

### Settings (environment)

| Variable | Meaning |
|---|---|
| `API_KEY` | required; clients send it as `X-API-Key` |
| `EMBED_URL`, `EMBED_MODEL`, `EMBED_API_KEY`, `EMBED_BATCH` | the embedding API (required: `EMBED_URL`) |
| `LISTEN`, `DATA_DIR` | address (default `0.0.0.0:9650`) and data folder (default `data`; `/data` in the image) |
| `VECTOR_BACKEND`, `VECTOR_URL` | `memory` (default), `txtai` (with `backends/txtai`), `helixdb` |
| `GRAPH_BACKEND`, `GRAPH_URL` | `sqlite` (default), `helixdb` |
| `EDGE_NEIGHBOURS`, `EDGE_MIN` | similarity edges per chunk |
| `UI_USER`, `UI_PASSWORD` | the graph view at `/ui` (off without a password) |

### kurag-sync

```toml
# sources.toml
[rag]
url = "http://localhost:9650"     # the key comes from RAG_API_KEY (environment or .env next to this file)

[[source]]
name = "docs"
type = "git"
url = "https://example.org/team/docs.git"
namespace = "team.docs"
include = ["**/*.md"]

[[source]]
name = "books"
type = "shelf"
sub = "book"
include = ["**/*.pdf", "**/*.epub"]
prefer = ["epub", "pdf"]
[source.map]
"math.algebra" = ["/srv/books/math/algebra"]
```

```sh
kurag-sync --config sources.toml --data ./state run --dry-run   # what would be sent
kurag-sync --config sources.toml --data ./state run             # send what changed
kurag-sync --config sources.toml --data ./state run --due       # only sources whose schedule has come (cron)
kurag-sync --config sources.toml extract --namespace team.docs --model gemma3:12b --api-base http://llm.example/v1
```

## Documentation

- [docs/design.md](docs/design.md): why kurag exists, principles, API, contract tests, plan
- [docs/architecture.md](docs/architecture.md): parts, state, storage, deployment, the sync client, measurements
- [docs/search.md](docs/search.md): how search works inside (tokenizer, scores, quantization, graphs)

## Development

`cargo test --workspace` runs the unit tests. The contract tests (`tests/contract/`) are black-box HTTP tests
against a running server: start `fake-embed` (a stand-in embedding API in the same crate), point a server at it,
then `KURAG_URL=http://localhost:9650 KURAG_API_KEY=... cargo test -p kurag-contract`.

## License

Apache License 2.0 ([LICENSE](LICENSE)). The graph view bundles Cytoscape.js (MIT, its notice is kept in
`ui/vendor/cytoscape.min.js`).
