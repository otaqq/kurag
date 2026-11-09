# kurag — configuration

The server is set up by environment variables; the sync client by a TOML file of sources. How to run either is in
[deployment.md](deployment.md); what the settings cost in memory and time is in [operations.md](operations.md).

## Server (`kurag`)

| Variable | Default | Meaning |
|---|---|---|
| `API_KEY` | (required) | clients send it as `X-API-Key`; the server does not start without it |
| `EMBED_URL` | (required) | base URL of an OpenAI-compatible API (`.../v1`); `/embeddings` is appended |
| `EMBED_MODEL` | empty | the `model` sent with every embedding request |
| `EMBED_API_KEY` | none | sent as `Authorization: Bearer` when set |
| `EMBED_BATCH` | 64 | texts per embedding request |
| `LISTEN` | `0.0.0.0:9650` | address and port |
| `DATA_DIR` | `data` (`/data` in the image) | one folder per namespace |
| `EDGE_NEIGHBOURS` | 10 | on upsert, each new chunk is linked to this many nearest chunks |
| `EDGE_MIN` | 0.4 | weaker links (cosine) are not kept |
| `VECTOR_BACKEND` | `memory` | `memory`, `txtai` or `helixdb`: where nearest-neighbour search runs |
| `VECTOR_URL` | none | the kurag-txtai service or the HelixDB server, for those backends |
| `GRAPH_BACKEND` | `sqlite` | `sqlite` or `helixdb`: where similarity edges and the knowledge graph are walked |
| `GRAPH_URL` | none | the HelixDB server, for that backend |
| `UI_USER`, `UI_PASSWORD` | none | the graph view at `/ui` (Basic login); no `/ui` without a password |

Every namespace must use the same embedding model: hits of several namespaces are merged by cosine, and the
dimensions of the first vectors stored fix those of the namespace. Changing the model means sending everything again
(delete the data folder, run the sync client).

All settings are read once at start; change them and restart. A configuration file (TOML) is planned
([design.md](design.md), "Configuration" in architecture.md).

### Choosing the backends

| | `memory` (default) | `txtai` | `helixdb` |
|---|---|---|---|
| Vector search | exact, over 8-bit vectors in kurag's memory | txtai's ANN (faiss): exact (IDMap) below `IVF_MIN` vectors per namespace, IVF above | HelixDB's HNSW |
| Extra process | none | kurag-txtai (`backends/txtai`, image `ghcr.io/otaqq/kurag-txtai`) | a HelixDB server |
| kurag's memory | grows with the vectors (about one byte per dimension per chunk) | small | small |
| When | up to a few hundred thousand chunks on one machine | when kurag's memory should stay small, or past what fits | tried; slower and larger at the sizes measured (search.md) |

Whatever the backend, chunks, full text, edges, the knowledge graph and a float16 copy of every vector stay in
SQLite: an external index can be dropped and filled again from it (`/rebuild`, or by itself when its count differs).
`GRAPH_BACKEND=helixdb` keeps a copy of the graphs in HelixDB; it gave the same answers more slowly (search.md,
"Graph backends"), so `sqlite` is the recommended setting.

### kurag-txtai (`backends/txtai`)

| Variable | Default | Meaning |
|---|---|---|
| `LISTEN` | `0.0.0.0:9670` | address and port (the chart uses `127.0.0.1:9670` next to kurag) |
| `DATA_DIR` | `data` (`/data` in the image) | one folder per namespace |
| `QUANTIZE` | 8 | faiss scalar quantizer bits (`SQ8`); 0 keeps float32 |
| `IVF_MIN` | 100000 | below this many vectors a namespace's index is exact (IDMap); above, IVF with 4·√n cells |
| `NPROBE` | a quarter of the cells | IVF cells searched |
| `SAVE_SECONDS` | 60 | indexes are written this long after a change, and on SIGTERM |

### HelixDB

The official image (`ghcr.io/helixdb/helixdb`, tested with v0.0.13), port 8080, data in `HELIX_DATA_DIR`. kurag
writes one label set for all namespaces, keyed by namespace, and creates its indexes on first use (a new index takes
about 20 s to become active).

## Sync client (`kurag-sync`)

```
kurag-sync [--config sources.toml] [--data DIR] [--env-file FILE] COMMAND
  run [--source NAME] [--kind KIND] [--namespace NS] [--due] [--dry-run]
  status
  shelves [--source NAME] [--files]
  text --file PATH
  extract --namespace NS [--model M] [--api-base URL ... | --api-bases-file FILE] [--json-mode schema|object|none]
          [--api-key-env VAR] [--parallel N] [--minutes M] [--limit N] [--dry-run]
```

- `--config`: the sources file (default `./sources.toml`); `<name>.local.toml` next to it is read too.
- `--data`: state and caches (default `$XDG_STATE_HOME/kurag-sync` or `~/.local/state/kurag-sync`).
- `--env-file`: `KEY=value` lines for secrets (default `.env` next to the config); the environment wins.
- `run --namespace NS`: only files going to `NS` or below it; what is gone is then deleted only there.
- `run --due`: only sources whose schedule has come (below); meant to run once an hour (systemd timer, cron).
- `status`: parts and chunks in the state, chunks per namespace on the server. `shelves`: which namespace each
  folder of a shelf source goes to, and which setting decided it. `text`: the text kurag-sync reads from a file.

### The sources file

```toml
walls = ["private"]                 # top level, before any table (see .rag.toml)

[rag]
url = "http://localhost:9650"
api_key_env = "RAG_API_KEY"         # default; the key itself goes in .env or the environment

[text]                              # words put into every part; changing them sends everything again
intro = "(intro)"                   # name of the text before a Markdown file's first "## "
source = "Source: {where}"          # the line saying where a part comes from

[kinds.docs]                        # a schedule shared by the sources of a kind
cron = "0 3 * * *"
[kinds.notes]
every = "1h"

[[source]]
name = "docs"                       # first part of every chunk id: "docs:<path>#<section>#<n>"
kind = "docs"
type = "git"
url = "https://example.org/team/docs.git"
branch = "main"
token_env = "GIT_TOKEN"             # optional: env var with a token
token_auth = "token"                # token (Forgejo, Gitea), bearer, or basic (GitHub: user x-access-token)
namespace = "team.docs"
label = "git/team/docs"             # shown in "Source:" lines (default: the source's name)
include = ["**/*.md"]
exclude = ["archive/**"]
```

Keys of every `[[source]]`:

| Key | Types | Meaning |
|---|---|---|
| `name` | all | unique; the first part of chunk ids, so renaming sends everything again |
| `type` | all | `git`, `dir`, `shelf` or `shelves`; other types are left out with one log line (the file can be shared with other tools) |
| `include`, `exclude` | all | fnmatch patterns on the path relative to the root (`*` matches `/` too); a leading `**/` also matches nothing. Default include `**/*.md` |
| `enabled` | all | `false` leaves the source out of everything; what it sent stays |
| `kind` | all | the `[kinds.<kind>]` schedule, and `run --kind` |
| `cron`, `every`, `at` | all | own schedule (wins over the kind's): `cron` "min hour day month weekday" (the minute is not used, runs once in a matching hour), `every` "30m" / "1h" / "2d" with 10 minutes of slack, `at` the hour an `every` source may run |
| `max_minutes` | all | stop the source after this long; the rest goes in on the next run, nothing is deleted as gone |
| `prefer` | all | `["epub", "pdf"]`: of files with the same folder and name, read only the first extension listed |
| `label` | all | the "where" of parts (default: `name`; none for shelf sources, whose paths are absolute) |
| `namespace` | git, dir | the namespace every file goes to |
| `url`, `branch`, `token_env`, `token_auth` | git | shallow clone into `<data>/cache/git/<name>` |
| `path` | dir, shelves | a folder; for `dir` also a list (ids then start with each folder's name) |
| `[source.map]` | shelf | namespace -> folder or list of folders, read in the file's order |
| `sub` | shelf | appended to every namespace (`math` + `book` -> `math.book`) |
| `prefix`, `only` | shelves | every folder under `path` named like a namespace is one: `prefix` is put before the name, `only` keeps names containing `.<only>.` |

Files are read as Markdown (`.md`, split at `## ` outside code fences), PDF (`.pdf`, one part per page through
`pdftotext`), EPUB (`.epub`, one part per chapter of the spine) or plain text (anything else, one part). PDF pages and
EPUB chapters with fewer than 50 letters are left out (scans without a text layer). Parts are cut into chunks of at
most 1,500 characters at paragraph breaks, each after the first starting with "title › section".

### `<name>.local.toml`

Read after the sources file, for what should not be in version control (private folders, other servers): a source
with the same name replaces the shared one, new names are added after the shared ones; `[rag]`, `[text]`,
`[extract]` and `[kinds]` keys replace the shared ones; `walls` replaces the list.

### Leaving a note out

A Markdown file whose YAML front matter has the top-level property `rag: false` is not indexed, in every source type.
A file that was indexed before is removed on the next run, like a deleted file; removing the property adds it again.
Other values, and files that are not Markdown, are not affected.

### `.rag.toml` (folder settings in shelf sources)

```toml
namespace = "math.algebra"          # this folder and below (older files may say shelf)
[[file]]
path = "Galois*.pdf"                # fnmatch, relative to this folder
namespace = "math.galois"
```

The nearest setting above a file wins; in one folder a `.rag.toml` wins over `[source.map]`. Only `namespace` (or
`shelf`) and `[[file]]` may be set. A `.rag.toml` that cannot be read, or that names a namespace on the other side of
a wall (a source mapped under `private.*` may only send to `private.*`, and other sources never there), stops the files
below it with an error rather than guessing. `kurag-sync shelves` shows the result per folder.

### Knowledge graph extraction (`[extract]` and `extract`)

```toml
[extract]
model = "gemma3:12b"
api_base = ["http://gpu1:11434/v1", "http://gpu2:11434/v1"]   # one per machine; list one twice for twice the requests
api_key_env = "LLM_API_KEY"         # if the API needs a key
json_mode = "schema"                # schema (default), object, or none
parallel = 1                        # requests at a time per API base
minutes = 50                        # stop after this long; the rest goes on next time
```

Options on the command line win. `--api-bases-file` reads the bases from a file before every batch ("URL" or "URL 2"
per line, `#` comments), so a caller can take machines out of a running extraction or add them. Each chunk's text is
cut at 6,000 characters; an answer may hold at most 15 entities and 20 relations; a request waits up to 300 s and is
retried after 15 and 60 s. A chunk counts as done once its result is on the server, so a stopped run goes on where it
left off.

### State and caches (`--data`)

| Path | Holds |
|---|---|
| `state/state.json` | part key -> sha256 of its text, namespace, chunk count (what was sent) |
| `state/files.json` | `<source>:<path>` -> size, mtime (ns), namespace, hash of the text settings (`label`, `[text]`): unchanged files are not opened. Changing one of those settings reads the source's files again and sends the parts whose text changed |
| `state/runs.json` | source -> start of its last run without errors (for `--due`) |
| `state/lock` | one run at a time; a second waits up to 30 minutes |
| `cache/text/` | extracted PDF and EPUB text, until the file's size or time changes |
| `cache/git/<name>` | git work trees |

Deleting `files.json` makes the next run read every file again (and send only what changed); deleting `state.json`
sends everything again.
