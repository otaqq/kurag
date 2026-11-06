//! Vector index backends: where the nearest-neighbour search of a namespace runs.
//!
//! - `memory` (default): the 8-bit vectors in this process, searched exactly (`vectors.rs`).
//! - `txtai`: txtai's ANN (faiss: SQ8, IVF once a namespace is large), through the small service in
//!   `backends/txtai/` that wraps it.
//! - `helixdb`: a HelixDB server (HNSW, float32, on disk), through its HTTP API.
//!
//! An external index holds only (chunk row id, vector). Everything else stays in the namespace's SQLite database,
//! including the float16 copy of every vector, which is the truth: an external index is filled from it when it
//! is empty, when its count differs, after a failed write, and on `/rebuild`, without calling the embedding API.
//! What kurag adds on top of either engine (Japanese full text, scores and fusion, chunk ids and prefix deletes,
//! the knowledge graph with provenance, the graph view) is the same whatever the backend.

use std::sync::Mutex;
use std::time::Duration;

use base64::Engine;
use serde_json::{Value, json};

pub type Result<T> = std::result::Result<T, String>;

pub trait RemoteIndex: Send + Sync {
    fn name(&self) -> &'static str;
    /// Vectors held for the namespace.
    fn count(&self, ns: &str) -> Result<usize>;
    /// Removes every vector of the namespace.
    fn clear(&self, ns: &str) -> Result<()>;
    fn add(&self, ns: &str, items: &[(i64, &[f32])]) -> Result<()>;
    /// Filling an emptied index: `fill` batches, then `fill_done` once (an engine may choose its index kind
    /// from the whole set). By default the same as `add`.
    fn fill(&self, ns: &str, items: &[(i64, &[f32])]) -> Result<()> {
        self.add(ns, items)
    }
    fn fill_done(&self, _ns: &str) -> Result<()> {
        Ok(())
    }
    fn remove(&self, ns: &str, vids: &[i64]) -> Result<()>;
    /// (row id, cosine similarity), best first.
    fn nearest(&self, ns: &str, q: &[f32], k: usize) -> Result<Vec<(i64, f32)>>;
    /// The same for the similarity edges made on upsert, once per new chunk: an engine may trade some recall
    /// for speed here. By default the same as `nearest`.
    fn neighbours(&self, ns: &str, q: &[f32], k: usize) -> Result<Vec<(i64, f32)>> {
        self.nearest(ns, q, k)
    }
}

/// Vectors sent per request when an index is filled.
pub const FILL_BATCH: usize = 500;

fn client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder().timeout(Duration::from_secs(600)).build().expect("http client")
}

fn f32_b64(v: &[f32]) -> String {
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn post(http: &reqwest::blocking::Client, url: &str, body: &Value) -> Result<Value> {
    let r = http.post(url).json(body).send().map_err(|e| format!("{url}: {e}"))?;
    let status = r.status();
    let text = r.text().map_err(|e| format!("{url}: {e}"))?;
    if !status.is_success() {
        return Err(format!("{url}: {status}: {}", text.chars().take(300).collect::<String>()));
    }
    if text.is_empty() { Ok(Value::Null) } else { serde_json::from_str(&text).map_err(|e| format!("{url}: {e}")) }
}

/// The backend named by `VECTOR_BACKEND`; None for `memory`.
pub fn from_settings(kind: &str, url: Option<&str>) -> std::result::Result<Option<Box<dyn RemoteIndex>>, String> {
    let need_url = || url.map(|u| u.trim_end_matches('/').to_string()).ok_or(format!("VECTOR_BACKEND={kind} needs VECTOR_URL"));
    match kind {
        "" | "memory" => Ok(None),
        "txtai" => Ok(Some(Box::new(Txtai { url: need_url()?, http: client() }))),
        "helixdb" => Ok(Some(Box::new(Helix { url: need_url()?, http: client(), indexed: Mutex::new(false) }))),
        other => Err(format!("VECTOR_BACKEND: unknown backend {other:?} (memory, txtai, helixdb)")),
    }
}

/// txtai's ANN behind `backends/txtai/server.py`: one index per namespace, saved in its own data folder.
pub struct Txtai {
    url: String,
    http: reqwest::blocking::Client,
}

impl Txtai {
    fn at(&self, ns: &str, op: &str) -> String {
        format!("{}/ns/{ns}/{op}", self.url)
    }
}

impl RemoteIndex for Txtai {
    fn name(&self) -> &'static str {
        "txtai"
    }

    fn count(&self, ns: &str) -> Result<usize> {
        let r = post(&self.http, &self.at(ns, "count"), &json!({}))?;
        Ok(r["count"].as_u64().unwrap_or(0) as usize)
    }

    fn clear(&self, ns: &str) -> Result<()> {
        post(&self.http, &self.at(ns, "clear"), &json!({})).map(|_| ())
    }

    fn add(&self, ns: &str, items: &[(i64, &[f32])]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let ids: Vec<i64> = items.iter().map(|i| i.0).collect();
        let flat: Vec<f32> = items.iter().flat_map(|i| i.1.iter().copied()).collect();
        post(&self.http, &self.at(ns, "add"), &json!({"ids": ids, "dims": items[0].1.len(), "vectors": f32_b64(&flat)})).map(|_| ())
    }

    fn fill(&self, ns: &str, items: &[(i64, &[f32])]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let ids: Vec<i64> = items.iter().map(|i| i.0).collect();
        let flat: Vec<f32> = items.iter().flat_map(|i| i.1.iter().copied()).collect();
        let body = json!({"ids": ids, "dims": items[0].1.len(), "vectors": f32_b64(&flat), "fill": true});
        post(&self.http, &self.at(ns, "add"), &body).map(|_| ())
    }

    fn fill_done(&self, ns: &str) -> Result<()> {
        post(&self.http, &self.at(ns, "build"), &json!({})).map(|_| ())
    }

    fn remove(&self, ns: &str, vids: &[i64]) -> Result<()> {
        if vids.is_empty() {
            return Ok(());
        }
        post(&self.http, &self.at(ns, "remove"), &json!({"ids": vids})).map(|_| ())
    }

    fn nearest(&self, ns: &str, q: &[f32], k: usize) -> Result<Vec<(i64, f32)>> {
        let r = post(&self.http, &self.at(ns, "search"), &json!({"vector": f32_b64(q), "k": k}))?;
        Ok(r["hits"]
            .as_array()
            .map(|a| a.iter().filter_map(|h| Some((h[0].as_i64()?, h[1].as_f64()? as f32))).collect())
            .unwrap_or_default())
    }
}

/// A HelixDB server: every namespace's vectors are nodes `{ns, vid, v}` of one label (`KuragChunk`), with one
/// cosine vector index on `v` partitioned by tenant `ns` (a separate index per namespace inside HelixDB, made once
/// for all of them) and equality indexes on `ns` and `vid`. One server holds vectors of one dimension. Writes and
/// deletes are Cypher statements; index creation and vector search are native queries (Cypher has no syntax for
/// them).
pub struct Helix {
    url: String,
    http: reqwest::blocking::Client,
    /// whether the indexes were made (or found) by this process
    indexed: Mutex<bool>,
}

const HELIX_LABEL: &str = "KuragChunk";

/// HelixDB's HNSW search widens with the number of hits asked for: on 33,000 vectors recall@5 against exact search
/// was 81 % asking for 5 or 50, 91 % for 200 and 99 % for 800, its cap (docs/search.md, "Vector index backends").
/// Searches ask for HELIX_SEARCH_CANDIDATES and keep the best k; the edges made on upsert, one search per new chunk,
/// ask for ten times k and at least HELIX_EDGE_CANDIDATES.
const HELIX_SEARCH_CANDIDATES: usize = 800;
const HELIX_EDGE_CANDIDATES: usize = 100;

impl Helix {
    /// A request, again while HelixDB answers that it may be retried (a transaction conflict between
    /// concurrent writes), up to about five seconds.
    fn send(&self, path: &str, body: &Value) -> Result<Value> {
        let url = format!("{}{path}", self.url);
        let mut wait = 10;
        loop {
            match post(&self.http, &url, body) {
                Err(e) if e.contains("\"retryable\":true") && wait <= 1280 => {
                    std::thread::sleep(Duration::from_millis(wait));
                    wait *= 2;
                }
                other => return other,
            }
        }
    }

    fn cypher(&self, query: &str, params: Value) -> Result<Value> {
        self.send("/v2/cypher", &json!({"query": query, "parameters": params}))
    }

    fn native(&self, kind: &str, name: &str, root: Value) -> Result<Value> {
        let body = json!({
            "request_type": kind,
            "query_name": name,
            "query": {kind: {"entries": [{"query": {"name": name, "root": root}}], "returns": [name]}},
        });
        self.send("/v2/query", &body)
    }

    fn create_index(&self, name: &str, spec: Value) -> Result<()> {
        self.native("write", name, json!({"create_index": {"spec": spec, "if_not_exists": true}})).map(|_| ())
    }

    /// The indexes, made if missing; a new vector index answers only once validated and activated (after the
    /// receipt), so this waits for that.
    fn ensure_indexes(&self, dims: usize) -> Result<()> {
        let mut done = self.indexed.lock().unwrap_or_else(|e| e.into_inner());
        if *done {
            return Ok(());
        }
        self.create_index(
            "vector_index",
            json!({"node_vector": {"label": HELIX_LABEL, "property": "v", "dimension": dims, "metric": "cosine",
                   "tenant_property": "ns"}}),
        )?;
        for property in ["ns", "vid"] {
            self.create_index("equality_index", json!({"node_equality": {"label": HELIX_LABEL, "property": property, "unique": false}}))?;
        }
        let mut probe = vec![0.0f32; dims];
        probe[0] = 1.0;
        let mut waited = 0;
        loop {
            match self.search("", &probe, 1) {
                Ok(_) => break,
                Err(e) if e.contains("index_not_found") && waited < 1200 => {
                    std::thread::sleep(Duration::from_millis(100));
                    waited += 1;
                }
                Err(e) => return Err(e),
            }
        }
        *done = true;
        Ok(())
    }

    fn search(&self, ns: &str, q: &[f32], k: usize) -> Result<Value> {
        self.native(
            "read",
            "hits",
            json!({"value_map": {"input": {"vector_search_nodes": {"label": HELIX_LABEL, "property": "v",
                    "tenant_value": {"value": {"string": ns}}, "query_vector": {"value": {"f32_array": q}},
                    "k": {"literal": k.min(800)}}}, "properties": ["vid", "$distance"]}}),
        )
    }

    /// The best k of `candidates` hits (cosine distance turned into similarity).
    fn best(&self, ns: &str, q: &[f32], k: usize, candidates: usize) -> Result<Vec<(i64, f32)>> {
        self.ensure_indexes(q.len())?;
        let r = self.search(ns, q, candidates)?;
        let mut hits: Vec<(i64, f32)> = r["hits"]
            .as_array()
            .map(|a| a.iter().filter_map(|h| Some((h["vid"].as_i64()?, 1.0 - h["$distance"].as_f64()? as f32))).collect())
            .unwrap_or_default();
        hits.truncate(k);
        Ok(hits)
    }
}

impl RemoteIndex for Helix {
    fn name(&self) -> &'static str {
        "helixdb"
    }

    fn count(&self, ns: &str) -> Result<usize> {
        let r = self.cypher(&format!("MATCH (c:{HELIX_LABEL} {{ns: $ns}}) RETURN count(c) AS n"), json!({"ns": ns}))?;
        Ok(r["rows"][0][0].as_u64().unwrap_or(0) as usize)
    }

    fn clear(&self, ns: &str) -> Result<()> {
        self.cypher(&format!("MATCH (c:{HELIX_LABEL} {{ns: $ns}}) DETACH DELETE c"), json!({"ns": ns})).map(|_| ())
    }

    fn add(&self, ns: &str, items: &[(i64, &[f32])]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        let dims = items[0].1.len();
        self.ensure_indexes(dims)?;
        // a Cypher parameter holds at most 200,000 values (HelixDB crates/db/src/cypher/parameters.rs)
        let per = (190_000 / (dims + 3)).max(1);
        for part in items.chunks(per) {
            let rows: Vec<Value> = part.iter().map(|(vid, v)| json!({"vid": vid, "v": v})).collect();
            self.cypher(
                &format!("UNWIND $rows AS r CREATE (:{HELIX_LABEL} {{ns: $ns, vid: r.vid, v: r.v}})"),
                json!({"ns": ns, "rows": rows}),
            )?;
        }
        Ok(())
    }

    fn remove(&self, ns: &str, vids: &[i64]) -> Result<()> {
        if vids.is_empty() {
            return Ok(());
        }
        self.cypher(
            &format!("MATCH (c:{HELIX_LABEL} {{ns: $ns}}) WHERE c.vid IN $vids DETACH DELETE c"),
            json!({"ns": ns, "vids": vids}),
        )
        .map(|_| ())
    }

    fn nearest(&self, ns: &str, q: &[f32], k: usize) -> Result<Vec<(i64, f32)>> {
        self.best(ns, q, k, HELIX_SEARCH_CANDIDATES.max(k))
    }

    fn neighbours(&self, ns: &str, q: &[f32], k: usize) -> Result<Vec<(i64, f32)>> {
        self.best(ns, q, k, (k * 10).max(HELIX_EDGE_CANDIDATES))
    }
}
