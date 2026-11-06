//! `kurag recall <questions.tsv>`: how close servers' vector search comes to exact search.
//!
//! The truth is exact cosine search over the float16 copies of the vectors in `DATA_DIR` (as stored by kurag or
//! imported from txtai-kura). Each question (`<namespace>\t<query>`, the namespace taken with everything under it)
//! is embedded once with `EMBED_URL` / `EMBED_MODEL` / `EMBED_API_KEY`; the servers (`A_URL` / `A_KEY`, and
//! `B_URL` / `B_KEY` if set) are asked the same question in vector mode, embedding it themselves with the same
//! model. Prints recall@5 (share of the exact top 5 a server returns) and how often its first hit is the exact
//! first hit. Only numbers are printed. Run it against servers holding the same data as `DATA_DIR`.

use std::collections::HashMap;
use std::path::Path;

use half::f16;
use serde_json::{Value, json};

use crate::config::{Config, parsed, var};
use crate::embed::Embedder;
use crate::store::Store;

const K: usize = 5;

/// The float16 vectors of one namespace, kept as they are (half the memory of float32).
struct Exact {
    vids: Vec<i64>,
    data: Vec<f16>,
    dims: usize,
}

fn load(store: &Store, name: &str) -> Result<Option<Exact>, String> {
    let Some(ns) = store.get(name, false).map_err(|e| format!("{name}: {e:?}"))? else { return Ok(None) };
    let inner = ns.lock();
    let mut st = inner.db.prepare("select vid, v from vectors").map_err(|e| e.to_string())?;
    let mut rows = st.query([]).map_err(|e| e.to_string())?;
    let (mut vids, mut data, mut dims) = (Vec::new(), Vec::new(), 0);
    while let Some(r) = rows.next().map_err(|e| e.to_string())? {
        let b: Vec<u8> = r.get(1).map_err(|e| e.to_string())?;
        dims = b.len() / 2;
        vids.push(r.get(0).map_err(|e| e.to_string())?);
        data.extend(b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]])));
    }
    Ok(Some(Exact { vids, data, dims }))
}

struct Server {
    name: &'static str,
    url: String,
    key: String,
}

async fn ask(http: &reqwest::Client, s: &Server, ns: &str, q: &str) -> Result<Vec<(String, String)>, String> {
    let r = http
        .post(format!("{}/search", s.url.trim_end_matches('/')))
        .header("X-API-Key", &s.key)
        .json(&json!({"query": q, "namespaces": [ns], "mode": "vector", "limit": K}))
        .send()
        .await
        .map_err(|e| format!("{}: {e}", s.name))?;
    if !r.status().is_success() {
        return Err(format!("{}: {}", s.name, r.status()));
    }
    let hits: Vec<Value> = r.json().await.map_err(|e| format!("{}: {e}", s.name))?;
    Ok(hits.iter().map(|h| (h["namespace"].as_str().unwrap_or("").into(), h["id"].as_str().unwrap_or("").into())).collect())
}

pub async fn run(path: &Path, data: &Path) -> Result<(), String> {
    let cfg = Config {
        listen: String::new(),
        data_dir: data.to_path_buf(),
        api_key: String::new(),
        embed_url: var("EMBED_URL").ok_or("set EMBED_URL")?,
        embed_model: var("EMBED_MODEL").unwrap_or_default(),
        embed_api_key: var("EMBED_API_KEY"),
        embed_batch: parsed("EMBED_BATCH", 64)?.max(1),
        edge_neighbours: 0,
        edge_min: 0.0,
        ui_user: String::new(),
        ui_password: None,
        vector_backend: String::new(),
        vector_url: None,
    };
    let embedder = Embedder::new(&cfg);
    let store = Store::new(data.to_path_buf(), None);
    let mut servers = Vec::new();
    for (name, p) in [("A", "A"), ("B", "B")] {
        if let Some(url) = var(&format!("{p}_URL")) {
            servers.push(Server { name, url, key: var(&format!("{p}_KEY")).unwrap_or_default() });
        }
    }
    let questions: Vec<(String, String)> = std::fs::read_to_string(path)
        .map_err(|e| format!("{}: {e}", path.display()))?
        .lines()
        .filter_map(|l| l.split_once('\t').map(|(n, q)| (n.to_string(), q.to_string())))
        .collect();
    let queries: Vec<String> = questions.iter().map(|q| q.1.clone()).collect();
    let vectors = embedder.embed(&queries).await?;

    let http = reqwest::Client::new();
    let mut cache: HashMap<String, Option<Exact>> = HashMap::new();
    let mut recall = vec![0usize; servers.len()];
    let mut top1 = vec![0usize; servers.len()];
    let mut missed: Vec<Vec<usize>> = vec![Vec::new(); servers.len()];
    let mut counted = 0usize;
    for (i, ((ns, q), v)) in questions.iter().zip(&vectors).enumerate() {
        // exact top K over every namespace under ns
        let mut best: Vec<(f32, String, i64)> = Vec::new();
        for name in store.expand(Some(std::slice::from_ref(ns)), None) {
            if !cache.contains_key(&name) {
                cache.insert(name.clone(), load(&store, &name)?);
            }
            let Some(e) = cache[&name].as_ref() else { continue };
            if e.dims != v.len() {
                return Err(format!("{name}: {} dimensions stored, the query has {}", e.dims, v.len()));
            }
            for (j, vid) in e.vids.iter().enumerate() {
                let row = &e.data[j * e.dims..(j + 1) * e.dims];
                let s: f32 = row.iter().zip(v).map(|(a, b)| a.to_f32() * b).sum();
                if best.len() < K || s > best[K - 1].0 {
                    best.push((s, name.clone(), *vid));
                    best.sort_by(|a, b| b.0.total_cmp(&a.0));
                    best.truncate(K);
                }
            }
        }
        if best.is_empty() {
            continue;
        }
        let mut truth: Vec<(String, String)> = Vec::new();
        for (_, name, vid) in &best {
            let ns = store.get(name, false).map_err(|e| format!("{e:?}"))?.expect("open");
            let rows = ns.lock().chunks(&[*vid]).map_err(|e| format!("{e:?}"))?;
            truth.push((name.clone(), rows.get(vid).map(|c| c.0.clone()).unwrap_or_default()));
        }
        counted += 1;
        for (k, s) in servers.iter().enumerate() {
            let got = ask(&http, s, ns, q).await?;
            recall[k] += got.iter().filter(|g| truth.contains(g)).count();
            if got.first() == truth.first() {
                top1[k] += 1;
            } else {
                missed[k].push(i + 1);
            }
        }
    }
    println!("{counted} questions, exact search over float16 copies as the truth");
    println!("server  recall@{K}  top1-exact  questions whose first hit differs");
    for (k, s) in servers.iter().enumerate() {
        println!(
            "{:<6} {:>7.1} %  {:>8.1} %  {:?}",
            s.name,
            100.0 * recall[k] as f64 / (counted * K).max(1) as f64,
            100.0 * top1[k] as f64 / counted.max(1) as f64,
            missed[k]
        );
    }
    Ok(())
}
