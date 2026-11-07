# kurag — deployment

kurag is two static binaries: `kurag` (the server) and `kurag-sync` (the client that fills it). The server keeps
everything in one data folder; it needs an OpenAI-compatible embedding API and, optionally, a txtai or HelixDB
process for the vector index. Settings are in [configuration.md](configuration.md), sizing in
[operations.md](operations.md).

## What is published

Each tag `v<version>` publishes (`.github/workflows/release.yml`):

| What | Where |
|---|---|
| Image: server and client (distroless, nonroot uid 65532, `/data` a volume, port 9650, health check `kurag health`), linux/amd64 and arm64 | `ghcr.io/otaqq/kurag:<version>`, `:<major>.<minor>`, `:latest` |
| Image: kurag-txtai, txtai's ANN behind a small HTTP API (`backends/txtai`, port 9670; distroless Python 3.13, about 300 MB, most of it numpy and faiss) | `ghcr.io/otaqq/kurag-txtai:<version>` |
| Helm chart | `oci://ghcr.io/otaqq/charts/kurag` |
| Static musl binaries for x86_64 and aarch64 with `LICENSE`, `NOTICE`, `README.md` and `dist/systemd/*`, plus `SHA256SUMS` | GitHub Releases |

Images and binaries come from the same build: the release binaries are built natively per architecture and copied
into the images.

## Docker Compose

`docker/compose.yaml`, from a clone of the repository:

```sh
export KURAG_API_KEY=change-me EMBED_URL=http://host.docker.internal:11434/v1 EMBED_MODEL=embeddinggemma
docker compose -f docker/compose.yaml up -d                      # kurag only
VECTOR_BACKEND=txtai VECTOR_URL=http://txtai:9670 \
  docker compose -f docker/compose.yaml --profile txtai up -d    # with kurag-txtai
```

| Variable | Default | Meaning |
|---|---|---|
| `KURAG_API_KEY`, `EMBED_URL` | (required) | the server's `API_KEY` and `EMBED_URL` |
| `EMBED_MODEL`, `EMBED_API_KEY` | empty | passed on |
| `VECTOR_BACKEND`, `VECTOR_URL`, `GRAPH_BACKEND`, `GRAPH_URL` | `memory`, -, `sqlite`, - | the backends; the services are `http://txtai:9670` and `http://helixdb:8080` |
| `KURAG_UI_USER`, `KURAG_UI_PASSWORD` | empty | the graph view's login |
| `KURAG_VERSION`, `HELIXDB_VERSION` | `latest`, `v0.0.13` | image tags |
| `KURAG_PORT` | 9650 | published port |

The profiles `txtai` and `helixdb` only start the services; kurag uses them when the backend variables say so. Data
is in the named volumes `kurag-data`, `txtai-data` and `helixdb-data`. `--build` builds from source instead of
pulling. An embedding API on the Docker host is reached as `host.docker.internal`.

## Kubernetes (Helm)

```sh
kubectl create secret generic kurag-api-key --from-literal=api-key=change-me
helm install kurag oci://ghcr.io/otaqq/charts/kurag \
  --set apiKey.existingSecret=kurag-api-key \
  --set embed.url=http://ollama.llm:11434/v1 --set embed.model=embeddinggemma
```

One Pod with the Recreate strategy: each namespace is a SQLite database written by one process, so there are no
replicas, and an upgrade stops the old Pod before the new one starts. The chart refuses to render without
`apiKey.existingSecret` and `embed.url`.

| Value | Default | Meaning |
|---|---|---|
| `image.repository`, `image.tag` | `ghcr.io/otaqq/kurag`, the chart's appVersion | |
| `apiKey.existingSecret`, `apiKey.secretKey` | (required), `api-key` | the Secret holding `API_KEY` |
| `embed.url`, `embed.model` | (required), empty | the embedding API |
| `embed.existingSecret`, `embed.secretKey` | none, `embed-api-key` | its key, if it needs one |
| `vector.backend`, `vector.url` | `memory`, empty | `txtai` without a URL runs kurag-txtai in the same Pod (`/data/txtai`, reached on 127.0.0.1:9670); `helixdb` needs a URL |
| `graph.backend`, `graph.url` | `sqlite`, empty | `helixdb` needs a URL |
| `txtai.image.*`, `txtai.resources` | `ghcr.io/otaqq/kurag-txtai`, 128Mi / 1Gi | the sidecar |
| `ui.existingSecret`, `ui.userKey`, `ui.passwordKey` | none, `ui-user`, `ui-password` | the graph view's login; off without |
| `extraEnv` | `[]` | more variables (`EDGE_NEIGHBOURS`, `EMBED_BATCH`, ...) |
| `persistence.enabled`, `size`, `storageClass`, `existingClaim`, `accessModes` | `true`, `10Gi`, default class, none, RWO | the data folder; the claim is kept on uninstall |
| `service.type`, `service.port` | `ClusterIP`, 9650 | |
| `ingress.*` | off | host, paths, TLS, class, annotations |
| `resources` | requests 50m / 128Mi, limit 2Gi | see operations.md for sizing |
| `podSecurityContext`, `securityContext` | uid 65532, read-only root, no capabilities | |
| `readinessProbe`, `livenessProbe` | `/health` every 10 s / 30 s, liveness tolerates 10 failures | opening large namespaces after a restart takes a while |
| `nodeSelector`, `tolerations`, `affinity`, `priorityClassName`, `podAnnotations`, `podLabels`, `imagePullSecrets` | | the usual |

Fill it with kurag-sync from wherever the documents are (a CronJob, a systemd timer on a machine that sees the
files), pointing `[rag] url` at the Service or the Ingress.

## systemd

The release archives (or `docker build -f docker/Dockerfile --target bin --output type=local,dest=out .`) give the
two binaries; `dist/systemd/` has the units.

```sh
install -m 755 kurag kurag-sync /usr/local/bin/
useradd --system --home /var/lib/kurag kurag
install -d -o kurag /var/lib/kurag /var/lib/kurag-sync /etc/kurag /etc/kurag-sync
cat > /etc/kurag/kurag.env <<'EOF'
API_KEY=change-me
EMBED_URL=http://127.0.0.1:11434/v1
EMBED_MODEL=embeddinggemma
EOF
chmod 600 /etc/kurag/kurag.env
cp kurag.service kurag-sync.service kurag-sync.timer /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now kurag kurag-sync.timer
```

- `kurag.service` listens on 127.0.0.1:9650 (put a reverse proxy in front for other machines), keeps data in
  `/var/lib/kurag`, and runs with `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, `NoNewPrivileges`.
- `kurag-sync.timer` runs `kurag-sync run --due` every hour at :15; each source's schedule in
  `/etc/kurag-sync/sources.toml` decides whether it is sent. The key goes in `/etc/kurag-sync/.env`
  (`RAG_API_KEY=...`). PDFs need `pdftotext` (poppler-utils), git sources `git`.
- With `VECTOR_BACKEND=txtai`, run kurag-txtai as a container next to it, or from `backends/txtai` with Python 3.13
  (`pip install -r requirements.txt; python server.py`).

## From source

```sh
docker build -f docker/Dockerfile -t kurag .                              # the image (default target)
docker build -f docker/Dockerfile --target bin --output type=local,dest=out .   # only the binaries
docker build -t kurag-txtai backends/txtai
```

The Dockerfile builds in `rust:1-alpine` (static musl) and caches cargo's registry and target folder, so a rebuild
after a small change takes seconds. The release profile (LTO, one codegen unit, stripped) is set by environment
variables in the Dockerfile, not in `Cargo.toml`, so local release builds stay fast. Without Docker:
`cargo build --release -p kurag -p kurag-sync` (any Rust 2024-edition toolchain; SQLite is bundled).

## Upgrades

- Read the release notes. Versions so far keep the data folder's format (tables are created when missing, nothing
  is converted); a version that changes it will say so there. The HTTP API stays at version 1.
- Compose: `docker compose pull && docker compose up -d`. Helm: `helm upgrade` (Recreate: a short stop). systemd:
  replace the binary, `systemctl restart kurag`.
- After an upgrade the vector index of each namespace is loaded (memory) or checked against the database (txtai,
  HelixDB) when the namespace is first used; the first searches are slower.

## Backup

The data folder (`/data`, `/var/lib/kurag`) is the whole state: one folder per namespace with `kurag.db` (SQLite in
WAL mode: copy `kurag.db`, `kurag.db-wal` and `kurag.db-shm` together, or use `sqlite3 kurag.db ".backup copy.db"`
while the server runs). An external vector index (txtai, HelixDB) needs no backup: kurag fills it again from the
float16 vectors in `kurag.db`. The sync client's state (`state/`) is worth keeping too: without it, the next run
sends every part again (the server keeps vectors of unchanged text, so this costs time, not embeddings).

## Security

- Every API route needs the API key; it is compared without an early exit (no timing hint). Use a long random key, and TLS in front
  (Ingress, reverse proxy) when the API crosses a network you do not trust: kurag itself speaks plain HTTP.
- The graph view shows document text: it has its own Basic login and is off without `UI_PASSWORD`.
- The images run as a non-root user with no shell; the chart sets a read-only root filesystem and drops every
  capability. Only the data folder is written.
- The embedding API sees every text that is added and every query; choose where it runs accordingly. TLS to it uses
  the bundled Mozilla roots (webpki-roots), not the system store: an API behind a private CA needs plain HTTP on a
  trusted network for now.
- CI scans both images and the repository with Trivy and the lock file with cargo-audit (`security.yml`).
