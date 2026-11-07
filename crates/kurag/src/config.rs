//! Settings from the environment.

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    /// `LISTEN`, default `0.0.0.0:9650`
    pub listen: String,
    /// `DATA_DIR`, default `./data`: one folder per namespace
    pub data_dir: PathBuf,
    /// `API_KEY`: clients send it in the `X-API-Key` header. Required.
    pub api_key: String,
    /// `EMBED_URL`: base URL of an OpenAI-compatible API (`.../v1`); `/embeddings` is appended
    pub embed_url: String,
    /// `EMBED_MODEL`
    pub embed_model: String,
    /// `EMBED_API_KEY`, sent as a bearer token when set
    pub embed_api_key: Option<String>,
    /// `EMBED_BATCH`, default 64: texts per embedding request
    pub embed_batch: usize,
    /// `EDGE_NEIGHBOURS`, default 10: on upsert each new chunk is linked to this many nearest chunks
    pub edge_neighbours: usize,
    /// `EDGE_MIN`, default 0.4: weaker links are not kept
    pub edge_min: f32,
    /// `VECTOR_BACKEND`: memory (default), txtai or helixdb (backend.rs)
    pub vector_backend: String,
    /// `VECTOR_URL`: the txtai service or HelixDB server, for those backends
    pub vector_url: Option<String>,
    /// `GRAPH_BACKEND`: sqlite (default) or helixdb, where similarity edges and the knowledge graph are walked
    pub graph_backend: String,
    /// `GRAPH_URL`: the HelixDB server, for that backend
    pub graph_url: Option<String>,
    /// `UI_USER`, `UI_PASSWORD`: the graph view's login (Basic auth); without a password there is no `/ui`
    #[cfg_attr(not(feature = "ui"), allow(dead_code))]
    pub ui_user: String,
    #[cfg_attr(not(feature = "ui"), allow(dead_code))]
    pub ui_password: Option<String>,
}

pub(crate) fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

pub(crate) fn parsed<T: std::str::FromStr>(name: &str, default: T) -> Result<T, String> {
    match var(name) {
        None => Ok(default),
        Some(v) => v.parse().map_err(|_| format!("{name}: cannot read {v:?}")),
    }
}

impl Config {
    pub fn from_env() -> Result<Config, String> {
        Ok(Config {
            listen: var("LISTEN").unwrap_or_else(|| "0.0.0.0:9650".into()),
            data_dir: var("DATA_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("data")),
            // an empty key would let every request in
            api_key: var("API_KEY").ok_or("set API_KEY (clients send it in the X-API-Key header)")?,
            embed_url: var("EMBED_URL").ok_or("set EMBED_URL (an OpenAI-compatible API, e.g. http://host:4000/v1)")?,
            embed_model: var("EMBED_MODEL").unwrap_or_default(),
            embed_api_key: var("EMBED_API_KEY"),
            embed_batch: parsed("EMBED_BATCH", 64)?.max(1),
            edge_neighbours: parsed("EDGE_NEIGHBOURS", 10)?,
            edge_min: parsed("EDGE_MIN", 0.4)?,
            vector_backend: var("VECTOR_BACKEND").unwrap_or_default(),
            vector_url: var("VECTOR_URL"),
            graph_backend: var("GRAPH_BACKEND").unwrap_or_default(),
            graph_url: var("GRAPH_URL"),
            ui_user: var("UI_USER").unwrap_or_default(),
            ui_password: var("UI_PASSWORD"),
        })
    }
}
