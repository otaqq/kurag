//! Contract tests for the kurag HTTP API (API v1 = txtai-kura v0.3.5), run against any server with that API.
//!
//! The server under test must use `fake-embed` (this crate's binary) as its embedding API, so that every
//! implementation gets the same vectors. Point the tests at it with `KURAG_URL` and `KURAG_API_KEY`.
//! Each test writes under its own root namespace (`ct<run>-<test>`) and leaves its data behind: run them against a
//! disposable server.

use std::collections::BTreeMap;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use md5::{Digest, Md5};
use serde_json::{Value, json};

/// Dimensions of the stand-in embedding.
pub const DIMS: usize = 256;

/// The stand-in embedding: hashed character trigrams of the lower-cased text, unit length, so texts that share
/// words get close vectors. Same as the stand-in in txtai-kura's own tests (md5 of each trigram, taken mod 256).
pub fn embed(text: &str) -> Vec<f32> {
    let chars: Vec<char> = text.to_lowercase().chars().collect();
    let mut v = vec![0f32; DIMS];
    for j in 0..chars.len().saturating_sub(2).max(1) {
        let gram: String = chars[j.min(chars.len())..(j + 3).min(chars.len())].iter().collect();
        let digest = Md5::digest(gram.as_bytes());
        v[digest[15] as usize % DIMS] += 1.0; // the last byte is the 128-bit value mod 256
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// A short id for this test run, so runs never share namespaces.
fn run_id() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
        format!("{:x}{:x}", t % 0xfff_ffff, std::process::id() % 0xfff)
    })
}

/// A client for the server under test, writing under one root namespace.
pub struct Api {
    url: String,
    key: String,
    /// The root namespace of this test; every namespace it uses is under it.
    pub root: String,
    http: reqwest::Client,
}

impl Api {
    /// None (and a note on stderr) when `KURAG_URL` is not set.
    pub fn from_env(test: &str) -> Option<Api> {
        let Ok(url) = std::env::var("KURAG_URL") else {
            eprintln!("SKIPPED {test}: set KURAG_URL (and KURAG_API_KEY) to the server under test");
            return None;
        };
        Some(Api {
            url: url.trim_end_matches('/').to_string(),
            key: std::env::var("KURAG_API_KEY").unwrap_or_default(),
            root: format!("ct{}-{}", run_id(), test.replace('_', "-")),
            http: reqwest::Client::new(),
        })
    }

    /// `name` under this test's root (`""` is the root itself).
    pub fn ns(&self, name: &str) -> String {
        if name.is_empty() {
            self.root.clone()
        } else {
            format!("{}.{name}", self.root)
        }
    }

    async fn send(&self, req: reqwest::RequestBuilder) -> (u16, Value) {
        let r = req.send().await.expect("server under test not reachable");
        let status = r.status().as_u16();
        let text = r.text().await.unwrap_or_default();
        (status, serde_json::from_str(&text).unwrap_or(Value::String(text)))
    }

    /// POST with the API key; the status and the answer.
    pub async fn post_status(&self, path: &str, body: Value) -> (u16, Value) {
        self.send(
            self.http
                .post(format!("{}{path}", self.url))
                .header("X-API-Key", &self.key)
                .json(&body),
        )
        .await
    }

    /// POST with the API key; the answer, which must be 200.
    pub async fn post(&self, path: &str, body: Value) -> Value {
        let (status, v) = self.post_status(path, body.clone()).await;
        assert_eq!(status, 200, "POST {path} {body} -> {v}");
        v
    }

    /// GET with the given API key (None: no header).
    pub async fn get_with_key(&self, path: &str, key: Option<&str>) -> (u16, Value) {
        let mut req = self.http.get(format!("{}{path}", self.url));
        if let Some(k) = key {
            req = req.header("X-API-Key", k);
        }
        self.send(req).await
    }

    /// Upsert into `ns(name)`; the answer.
    pub async fn upsert(&self, name: &str, docs: Value) -> Value {
        self.post("/upsert", json!({"namespace": self.ns(name), "documents": docs}))
            .await
    }

    /// Delete prefixes from `ns(name)`; the answer.
    pub async fn delete(&self, name: &str, prefixes: &[&str]) -> Value {
        self.post("/delete", json!({"namespace": self.ns(name), "prefixes": prefixes}))
            .await
    }

    /// `/search`; namespaces default to this test's root, and given ones are taken under it.
    pub async fn search(&self, mut body: Value) -> Vec<Value> {
        self.scope(&mut body);
        self.post("/search", body)
            .await
            .as_array()
            .expect("search answers a list")
            .clone()
    }

    /// `/kg/search`, scoped like `search`.
    pub async fn kg_search(&self, mut body: Value) -> Value {
        self.scope(&mut body);
        self.post("/kg/search", body).await
    }

    fn scope(&self, body: &mut Value) {
        for key in ["namespaces", "exclude"] {
            if let Some(list) = body.get(key).and_then(Value::as_array).cloned() {
                body[key] = list.iter().map(|n| json!(self.ns(n.as_str().unwrap()))).collect();
            }
        }
        if body.get("namespaces").is_none() {
            body["namespaces"] = json!([self.root]);
        }
    }

    /// `/namespaces` under this test's root, with the root taken off the names (`""` is the root itself).
    pub async fn namespaces(&self) -> BTreeMap<String, i64> {
        let (status, v) = self.get_with_key("/namespaces", Some(&self.key)).await;
        assert_eq!(status, 200, "GET /namespaces -> {v}");
        let prefix = format!("{}.", self.root);
        v.as_object()
            .expect("namespaces answers an object")
            .iter()
            .filter_map(|(name, count)| {
                let short = if *name == self.root {
                    Some("")
                } else {
                    name.strip_prefix(&prefix)
                };
                short.map(|s| (s.to_string(), count.as_i64().unwrap()))
            })
            .collect()
    }
}

/// Documents `<prefix>#0`, `<prefix>#1`, ... with the texts, source = prefix.
pub fn docs(prefix: &str, texts: &[&str]) -> Value {
    texts
        .iter()
        .enumerate()
        .map(|(i, t)| json!({"id": format!("{prefix}#{i}"), "text": t, "source": prefix}))
        .collect()
}

/// Ids of the hits, in order.
pub fn ids(hits: &[Value]) -> Vec<String> {
    hits.iter().map(|h| h["id"].as_str().unwrap().to_string()).collect()
}

/// A short Japanese text on several topics, one chunk each.
pub const BOOK: [&str; 5] = [
    "Kubernetes の Pod はコンテナをまとめて動かす単位です。",
    "Pod は一つ以上のコンテナを持ち、同じネットワークを共有します。",
    "設定の方法: ConfigMap と Secret でアプリに値を渡します。",
    "PostgreSQL のバックアップは pg_dump で取ります。",
    "認証には OAuth2 と OpenID Connect を使います。",
];
