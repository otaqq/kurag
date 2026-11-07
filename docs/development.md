# kurag — development

How to build, test and release kurag. Why it is built this way is in [design.md](design.md), how it is put together
in [architecture.md](architecture.md).

## Layout

```
Cargo.toml            workspace (edition 2024, Apache-2.0)
crates/kurag-core/    library: tokenizer, scores, ids and namespaces, API types
crates/kurag/         the server (axum, rusqlite with bundled SQLite), and the import / recall tools
crates/kurag-sync/    the client: sources -> chunks -> server (run, status, shelves, text), knowledge graph extraction
ui/                   the graph view: plain HTML/JS/CSS built into the server (cargo feature "ui"); ui/vendor/ holds Cytoscape.js
tests/contract/       black-box HTTP tests, fake-embed (a stand-in embedding API) and compare (two servers side by side)
backends/txtai/       kurag-txtai: txtai's ANN behind a small HTTP API (Python), own Dockerfile
docker/               Dockerfile (image, binaries) and compose.yaml
charts/kurag/         Helm chart
dist/systemd/         example units
docs/                 design, architecture, search, api, configuration, deployment, operations, development
.github/workflows/    ci, release, security
```

## Build and test

```sh
cargo build                                   # debug, all crates
cargo test --workspace                        # unit tests (the contract tests skip without a server)
cargo fmt --all --check                       # rustfmt.toml: max_width 120
cargo clippy --workspace --all-targets -- -D warnings
```

Any Rust toolchain with edition 2024; SQLite is compiled in (rusqlite `bundled`), so no system libraries are needed.
The release profile (LTO, one codegen unit, strip) is set only in `docker/Dockerfile`, so local release builds stay
quick. CI runs exactly the four commands above; run them before a push.

## Contract tests

`tests/contract` (crate `kurag-contract`) tests the HTTP API from outside, so the same tests run against any server
with this API. They need a running server and an embedding API; `fake-embed` stands in for the latter (hashed
character trigrams, 256 dimensions, unit length, texts sharing words come out close), so every server gets the same
vectors and no model is needed.

```sh
cargo build -p kurag -p kurag-contract --bins
FAKE_EMBED_LISTEN=127.0.0.1:9651 target/debug/fake-embed &
LISTEN=127.0.0.1:9650 DATA_DIR=/tmp/kurag-ct API_KEY=ct EMBED_URL=http://127.0.0.1:9651/v1 EMBED_MODEL=fake \
  target/debug/kurag &
KURAG_URL=http://127.0.0.1:9650 KURAG_API_KEY=ct cargo test -p kurag-contract
```

Each test writes under its own root namespace (`ct<run>-<test>`) and leaves its data behind: use a throwaway data
folder. A new API behaviour gets a contract test first (design.md, "Contract tests").

## Tools for comparing and measuring

| Tool | Does |
|---|---|
| `kurag import <dir>` | copies txtai-kura v0.3 namespaces into `DATA_DIR` (chunks, float16 vectors, edges, knowledge graph; full text made again) |
| `kurag recall <tsv>` | recall@5 and top-1 of servers `A_URL` / `B_URL` against exact search over the float16 vectors in `DATA_DIR` |
| `compare` (in kurag-contract) | the same queries to two servers (`A_URL`, `B_URL`): overlap@5, same first hit, latency per mode |
| `kurag-sync text --file F` | the text kurag-sync reads from a file, without its cache |

## kurag-sync and its Python origin

kurag-sync recognises what was sent before by the sha256 of each part's text, and its state files may come from the
Python client it replaces (txtai-kura-sync). So the text must come out byte for byte the same:

- `pystr.rs` follows Python's string rules where the text depends on them: `splitlines` (also `\v`, `\f`, `\x1c`..,
  U+2028), `isspace` (Unicode White_Space plus `\x1c`..`\x1f`), lengths in code points, universal newlines when
  reading files, `fnmatch` for patterns.
- `html.rs` ports Python 3.14's `html.parser` the way the Python client used it (fed once, never closed: what it
  leaves unparsed at the end is dropped), with `html.unescape`. `entities.rs` holds the HTML5 named references and
  the invalid numeric ones; it is generated from Python (`html.entities.html5`, `html._invalid_charrefs`,
  `html._invalid_codepoints`) and not edited by hand.
- The PDF/EPUB text cache is the same file format; JSON floats are read exactly (serde_json `float_roundtrip`), or
  no cached mtime would match.
- The extraction prompt and answer schema keep the Python key order (serde_json `preserve_order`): the model writes
  its answer in the schema's order, and sorted keys gave other answers.

A change to how parts are made changes every part's sha256, so every chunk is sent again on the next run: say so in
the release notes.

## Release

1. Raise the version in `crates/kurag/Cargo.toml`, `crates/kurag-core/Cargo.toml`, `crates/kurag-sync/Cargo.toml`
   and `charts/kurag/Chart.yaml` (`version` and `appVersion`); CI checks that the chart follows the crate.
2. Note the change in the docs (architecture.md "State" names the version).
3. Commit, push, wait for CI.
4. Tag `v<version>` and push the tag: `release.yml` checks that the tag matches `Cargo.toml`, builds the static
   binaries natively for x86_64 and aarch64 and runs the tests there, pushes the multi-arch images
   (`ghcr.io/otaqq/kurag`, `ghcr.io/otaqq/kurag-txtai`) with SBOM and provenance, the chart
   (`oci://ghcr.io/otaqq/charts/kurag`), and a GitHub Release with archives and `SHA256SUMS`. A tag with `-`
   (`v0.5.0-rc.1`) is a pre-release and does not move `latest`.

## CI

| Workflow | When | Does |
|---|---|---|
| `ci.yml` | push to main, pull requests | fmt, clippy, unit tests; contract tests against a fresh server with fake-embed; both images (and `--version`, `import server`); the chart (lint, render with every option on, kubeconform, version check); tests and the image on arm64 (GitHub only) |
| `release.yml` | tag `v*.*.*` | as above (GitHub only) |
| `security.yml` | push to main, pull requests, weekly | Trivy on both images and the repository, cargo-audit on `Cargo.lock`; on GitHub as code-scanning results, on other forges as an issue |

The jobs that need GitHub (arm64 runners, ghcr.io) check `github.server_url`, so the same workflows can run on a
Forgejo or Gitea runner.
