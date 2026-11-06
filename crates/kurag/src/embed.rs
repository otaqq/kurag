//! The embedding API: OpenAI-compatible `POST <url>/embeddings`. The only model call kurag makes.

use serde::Deserialize;
use serde_json::json;

use crate::config::Config;

pub struct Embedder {
    http: reqwest::Client,
    url: String,
    model: String,
    key: Option<String>,
    batch: usize,
}

#[derive(Deserialize)]
struct Answer {
    data: Vec<Item>,
}

#[derive(Deserialize)]
struct Item {
    index: usize,
    embedding: Vec<f32>,
}

impl Embedder {
    pub fn new(cfg: &Config) -> Embedder {
        Embedder {
            http: reqwest::Client::builder().timeout(std::time::Duration::from_secs(300)).build().expect("http client"),
            url: format!("{}/embeddings", cfg.embed_url.trim_end_matches('/')),
            model: cfg.embed_model.clone(),
            key: cfg.embed_api_key.clone(),
            batch: cfg.embed_batch,
        }
    }

    /// Unit-length vectors for the texts, in order (inner product = cosine similarity).
    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        let mut out = Vec::with_capacity(texts.len());
        for part in texts.chunks(self.batch) {
            let mut req = self.http.post(&self.url).json(&json!({"model": self.model, "input": part}));
            if let Some(k) = &self.key {
                req = req.bearer_auth(k);
            }
            let r = req.send().await.map_err(|e| format!("embedding API: {e}"))?;
            let status = r.status();
            if !status.is_success() {
                let body = r.text().await.unwrap_or_default();
                return Err(format!("embedding API: {status}: {}", body.chars().take(300).collect::<String>()));
            }
            let mut answer: Answer = r.json().await.map_err(|e| format!("embedding API answer: {e}"))?;
            if answer.data.len() != part.len() {
                return Err(format!("embedding API: {} vectors for {} texts", answer.data.len(), part.len()));
            }
            answer.data.sort_by_key(|i| i.index);
            out.extend(answer.data.into_iter().map(|i| normalize(i.embedding)));
        }
        Ok(out)
    }
}

pub fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}
