//! A stand-in for an OpenAI-compatible embedding API (`POST /v1/embeddings`), answering with
//! `kurag_contract::embed`: same text, same vector, on every server under test. Listens on `FAKE_EMBED_LISTEN`
//! (default `0.0.0.0:9651`).

use axum::{
    Json, Router,
    routing::{get, post},
};
use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(untagged)]
enum Input {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct Request {
    input: Input,
    #[serde(default)]
    model: String,
    /// "base64" (what the OpenAI Python client asks for) or "float" / none
    #[serde(default)]
    encoding_format: Option<String>,
}

async fn embeddings(Json(r): Json<Request>) -> Json<Value> {
    let texts = match r.input {
        Input::One(t) => vec![t],
        Input::Many(ts) => ts,
    };
    let base64 = r.encoding_format.as_deref() == Some("base64");
    let data: Vec<Value> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let v = kurag_contract::embed(t);
            let embedding = if base64 {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                json!(base64::engine::general_purpose::STANDARD.encode(bytes))
            } else {
                json!(v)
            };
            json!({"object": "embedding", "index": i, "embedding": embedding})
        })
        .collect();
    Json(json!({"object": "list", "data": data, "model": r.model, "usage": {"prompt_tokens": 0, "total_tokens": 0}}))
}

#[tokio::main]
async fn main() {
    let addr = std::env::var("FAKE_EMBED_LISTEN").unwrap_or_else(|_| "0.0.0.0:9651".into());
    let app = Router::new()
        .route("/v1/embeddings", post(embeddings))
        .route("/embeddings", post(embeddings))
        .route("/health", get(|| async { "ok" }));
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("bind");
    eprintln!("fake-embed: {} dimensions on {addr}", kurag_contract::DIMS);
    axum::serve(listener, app).await.expect("serve");
}
