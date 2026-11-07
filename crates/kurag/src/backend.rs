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

use std::collections::HashMap;
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
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .expect("http client")
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
        return Err(format!(
            "{url}: {status}: {}",
            text.chars().take(300).collect::<String>()
        ));
    }
    if text.is_empty() {
        Ok(Value::Null)
    } else {
        serde_json::from_str(&text).map_err(|e| format!("{url}: {e}"))
    }
}

/// The backend named by `VECTOR_BACKEND`; None for `memory`.
pub fn from_settings(kind: &str, url: Option<&str>) -> std::result::Result<Option<Box<dyn RemoteIndex>>, String> {
    let need_url = || {
        url.map(|u| u.trim_end_matches('/').to_string())
            .ok_or(format!("VECTOR_BACKEND={kind} needs VECTOR_URL"))
    };
    match kind {
        "" | "memory" => Ok(None),
        "txtai" => Ok(Some(Box::new(Txtai {
            url: need_url()?,
            http: client(),
        }))),
        "helixdb" => Ok(Some(Box::new(Helix {
            client: HelixClient::new(&need_url()?),
            indexed: Mutex::new(false),
        }))),
        other => Err(format!(
            "VECTOR_BACKEND: unknown backend {other:?} (memory, txtai, helixdb)"
        )),
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
        post(
            &self.http,
            &self.at(ns, "add"),
            &json!({"ids": ids, "dims": items[0].1.len(), "vectors": f32_b64(&flat)}),
        )
        .map(|_| ())
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
        let r = post(
            &self.http,
            &self.at(ns, "search"),
            &json!({"vector": f32_b64(q), "k": k}),
        )?;
        Ok(r["hits"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|h| Some((h[0].as_i64()?, h[1].as_f64()? as f32)))
                    .collect()
            })
            .unwrap_or_default())
    }
}

/// A HelixDB server: every namespace's vectors are nodes `{ns, vid, v}` of one label (`KuragChunk`), with one
/// cosine vector index on `v` partitioned by tenant `ns` (a separate index per namespace inside HelixDB, made once
/// for all of them) and equality indexes on `ns` and `vid`. One server holds vectors of one dimension. Writes and
/// deletes are Cypher statements; index creation and vector search are native queries (Cypher has no syntax for
/// them).
pub struct Helix {
    client: HelixClient,
    /// whether the indexes were made (or found) by this process
    indexed: Mutex<bool>,
}

/// Requests to a HelixDB server: Cypher statements and native (operation-tree) queries.
pub struct HelixClient {
    url: String,
    http: reqwest::blocking::Client,
}

const HELIX_LABEL: &str = "KuragChunk";

/// HelixDB's HNSW search widens with the number of hits asked for: on 33,000 vectors recall@5 against exact search
/// was 81 % asking for 5 or 50, 91 % for 200 and 99 % for 800, its cap (docs/search.md, "Vector index backends").
/// Searches ask for HELIX_SEARCH_CANDIDATES and keep the best k; the edges made on upsert, one search per new chunk,
/// ask for ten times k and at least HELIX_EDGE_CANDIDATES.
const HELIX_SEARCH_CANDIDATES: usize = 800;
const HELIX_EDGE_CANDIDATES: usize = 100;

impl HelixClient {
    pub fn new(url: &str) -> HelixClient {
        HelixClient {
            url: url.trim_end_matches('/').to_string(),
            http: client(),
        }
    }

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
        self.native(
            "write",
            name,
            json!({"create_index": {"spec": spec, "if_not_exists": true}}),
        )
        .map(|_| ())
    }

    /// The rows of a Cypher answer.
    fn rows(&self, query: &str, params: Value) -> Result<Vec<Value>> {
        let r = self.cypher(query, params)?;
        Ok(r["rows"].as_array().cloned().unwrap_or_default())
    }
}

impl Helix {
    /// The indexes, made if missing; a new vector index answers only once validated and activated (after the
    /// receipt), so this waits for that.
    fn ensure_indexes(&self, dims: usize) -> Result<()> {
        let mut done = self.indexed.lock().unwrap_or_else(|e| e.into_inner());
        if *done {
            return Ok(());
        }
        self.client.create_index(
            "vector_index",
            json!({"node_vector": {"label": HELIX_LABEL, "property": "v", "dimension": dims, "metric": "cosine",
                   "tenant_property": "ns"}}),
        )?;
        for property in ["ns", "vid"] {
            self.client.create_index(
                "equality_index",
                json!({"node_equality": {"label": HELIX_LABEL, "property": property, "unique": false}}),
            )?;
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
        self.client.native(
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
            .map(|a| {
                a.iter()
                    .filter_map(|h| Some((h["vid"].as_i64()?, 1.0 - h["$distance"].as_f64()? as f32)))
                    .collect()
            })
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
        let r = self.client.cypher(
            &format!("MATCH (c:{HELIX_LABEL} {{ns: $ns}}) RETURN count(c) AS n"),
            json!({"ns": ns}),
        )?;
        Ok(r["rows"][0][0].as_u64().unwrap_or(0) as usize)
    }

    fn clear(&self, ns: &str) -> Result<()> {
        self.client
            .cypher(
                &format!("MATCH (c:{HELIX_LABEL} {{ns: $ns}}) DETACH DELETE c"),
                json!({"ns": ns}),
            )
            .map(|_| ())
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
            self.client.cypher(
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
        self.client
            .cypher(
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

/// Graph backends: where the similarity edges and the knowledge graph are walked. `sqlite` (default, None here)
/// walks the namespace's own tables; `helixdb` keeps a copy of both in a HelixDB server and walks it there. The
/// namespace's database stays the truth: every write goes there first, the copy follows inside the same
/// transaction, and a copy found at another version than the database (`graph_version`) is filled again from it.
pub trait GraphStore: Send + Sync {
    fn name(&self) -> &'static str;
    /// The version the copy of the namespace was last brought to (None: no copy).
    fn version(&self, ns: &str) -> Result<Option<i64>>;
    fn set_version(&self, ns: &str, version: i64) -> Result<()>;
    /// Removes the namespace's copy.
    fn clear(&self, ns: &str) -> Result<()>;
    /// Chunks (row id, chunk id).
    fn add_chunks(&self, ns: &str, chunks: &[(i64, String)]) -> Result<()>;
    /// Removes chunks with their edges and mentions.
    fn remove_chunks(&self, ns: &str, vids: &[i64]) -> Result<()>;
    /// Similarity edges (a, b, weight), each pair once.
    fn add_edges(&self, ns: &str, edges: &[(i64, i64, f64)]) -> Result<()>;
    /// Entities by id, made if missing.
    fn add_entities(&self, ns: &str, eids: &[i64]) -> Result<()>;
    fn remove_entities(&self, ns: &str, eids: &[i64]) -> Result<()>;
    /// Relations (source, target, type, weight, chunk id it was learned from).
    fn add_relations(&self, ns: &str, relations: &[(i64, i64, String, f64, String)]) -> Result<()>;
    /// Removes the relations from these sources to these targets learned from these chunks (the caller adds
    /// back, from the tables, those of the set that remain: a relation sent again replaces itself).
    fn remove_relations(&self, ns: &str, sources: &[i64], targets: &[i64], chunk_ids: &[String]) -> Result<()>;
    /// Removes the relations learned from these chunks, and these chunks' mentions.
    fn forget_chunks(&self, ns: &str, chunk_ids: &[String]) -> Result<()>;
    /// The same for the chunk ids in [lo, hi) (hi None: to the end).
    fn forget_range(&self, ns: &str, lo: &str, hi: Option<&str>) -> Result<()>;
    /// Mentions (entity id, chunk row id).
    fn add_mentions(&self, ns: &str, mentions: &[(i64, i64)]) -> Result<()>;
    /// Removes the mentions of these entities in these chunks (the caller adds back those that remain).
    fn remove_mentions(&self, ns: &str, eids: &[i64], vids: &[i64]) -> Result<()>;
    /// (neighbour, weight, the chunk it hangs on) for the edges of these chunks at least `min_weight` similar.
    fn neighbours(&self, ns: &str, vids: &[i64], min_weight: f64) -> Result<Vec<(i64, f64, i64)>>;
    /// One step of a walk: (source, target, type, highest weight) of the relations touching these entities, the
    /// heaviest first (then by source, target, type), at most `limit`.
    fn relations_around(&self, ns: &str, eids: &[i64], limit: usize) -> Result<Vec<(i64, i64, String, f64)>>;
    /// (chunk row id, how many of these entities it mentions).
    fn chunks_mentioning(&self, ns: &str, eids: &[i64]) -> Result<Vec<(i64, usize)>>;
}

/// The graph backend named by `GRAPH_BACKEND`; None for `sqlite`.
pub fn graph_from_settings(kind: &str, url: Option<&str>) -> std::result::Result<Option<Box<dyn GraphStore>>, String> {
    match kind {
        "" | "sqlite" => Ok(None),
        "helixdb" => {
            let url = url.ok_or("GRAPH_BACKEND=helixdb needs GRAPH_URL")?;
            Ok(Some(Box::new(HelixGraph {
                client: HelixClient::new(url),
                indexed: Mutex::new(false),
            })))
        }
        other => Err(format!("GRAPH_BACKEND: unknown backend {other:?} (sqlite, helixdb)")),
    }
}

/// The graph in HelixDB: nodes `KGChunk {ns, vid, id}`, `KGEntity {ns, eid}`, `KGMeta {ns, version}`; edges
/// `SIMILAR {w}` (once per pair, walked both ways), `RELATES {type, weight, chunk}`, `MENTIONS` (chunk to entity).
/// All namespaces share the labels, keyed by `ns`, with equality indexes on `ns`, `vid`, `id` and `eid`.
///
/// HelixDB uses an index only for values given as parameters (`{ns: $ns}`, `x IN $list`), not for values taken
/// from an `UNWIND` row: matching both ends of each new edge in Cypher scanned the whole label per row and ran out
/// of time. So nodes are looked up by `IN` lists, and edges are made by node id in native write batches. And a
/// match on both `ns` and the id list started from the namespace's nodes (2.5 s for the edges of two chunks on
/// 47,000 chunks); matching the id list first and checking `ns` after a `WITH` took 0.02 s, so every read starts
/// from the ids.
pub struct HelixGraph {
    client: HelixClient,
    indexed: Mutex<bool>,
}

/// Values per Cypher list parameter, and edges per native write batch.
const GRAPH_BATCH: usize = 2000;
const EDGE_BATCH: usize = 500;
/// An edge to write to HelixDB: from node id, to node id, properties.
type EdgeRow<'a> = (i64, i64, Vec<(&'a str, Value)>);

fn int(v: &Value) -> i64 {
    v.as_i64().or_else(|| v.as_f64().map(|f| f as i64)).unwrap_or(0)
}

/// A native property value.
fn prop(v: &Value) -> Value {
    match v {
        Value::String(s) => json!({"value": {"string": s}}),
        Value::Number(n) if n.is_f64() => json!({"value": {"f64": n.as_f64()}}),
        Value::Number(n) => json!({"value": {"i64": n.as_i64()}}),
        other => json!({"value": {"string": other.to_string()}}),
    }
}

impl HelixGraph {
    fn ensure_indexes(&self) -> Result<()> {
        let mut done = self.indexed.lock().unwrap_or_else(|e| e.into_inner());
        if *done {
            return Ok(());
        }
        for (label, property) in [
            ("KGChunk", "ns"),
            ("KGChunk", "vid"),
            ("KGChunk", "id"),
            ("KGEntity", "ns"),
            ("KGEntity", "eid"),
            ("KGMeta", "ns"),
        ] {
            self.client.create_index(
                "equality_index",
                json!({"node_equality": {"label": label, "property": property, "unique": false}}),
            )?;
        }
        *done = true;
        Ok(())
    }

    fn write(&self, query: &str, params: Value) -> Result<()> {
        self.ensure_indexes()?;
        self.client.cypher(query, params).map(|_| ())
    }

    fn read(&self, query: &str, params: Value) -> Result<Vec<Value>> {
        self.ensure_indexes()?;
        self.client.rows(query, params)
    }

    /// HelixDB's node id of each key (`vid` of chunks, `eid` of entities) found in the namespace.
    fn node_ids(&self, label: &str, key: &str, ns: &str, keys: &[i64]) -> Result<HashMap<i64, i64>> {
        let mut keys: Vec<i64> = keys.to_vec();
        keys.sort_unstable();
        keys.dedup();
        let mut out = HashMap::new();
        for part in keys.chunks(GRAPH_BATCH) {
            let rows = self.read(
                &format!("MATCH (n:{label}) WHERE n.{key} IN $keys WITH n WHERE n.ns = $ns RETURN n.{key}, id(n)"),
                json!({"ns": ns, "keys": part}),
            )?;
            out.extend(rows.iter().map(|r| (int(&r[0]), int(&r[1]))));
        }
        Ok(out)
    }

    /// Edges (from node id, to node id, properties) in native write batches.
    fn add_edges_by_id(&self, label: &str, edges: &[EdgeRow<'_>]) -> Result<()> {
        for part in edges.chunks(EDGE_BATCH) {
            let entries: Vec<Value> = part
                .iter()
                .enumerate()
                .map(|(n, (from, to, props))| {
                    let props: Vec<Value> = props.iter().map(|(k, v)| json!([k, prop(v)])).collect();
                    json!({"query": {"name": format!("e{n}"), "root": {"count": {"input": {"add_e": {
                        "input": {"nodes": {"reference": {"ids": [from]}}}, "label": label,
                        "to": {"ids": [to]}, "properties": props}}}}}})
                })
                .collect();
            let body = json!({"request_type": "write", "query_name": "edges",
                "query": {"write": {"entries": entries, "returns": []}}});
            self.client.send("/v2/query", &body)?;
        }
        Ok(())
    }
}

impl GraphStore for HelixGraph {
    fn name(&self) -> &'static str {
        "helixdb"
    }

    fn version(&self, ns: &str) -> Result<Option<i64>> {
        let rows = self.read("MATCH (m:KGMeta {ns: $ns}) RETURN m.version AS v", json!({"ns": ns}))?;
        Ok(rows.first().map(|r| int(&r[0])))
    }

    fn set_version(&self, ns: &str, version: i64) -> Result<()> {
        self.write("MATCH (m:KGMeta {ns: $ns}) DELETE m", json!({"ns": ns}))?;
        self.write(
            "CREATE (:KGMeta {ns: $ns, version: $v})",
            json!({"ns": ns, "v": version}),
        )
    }

    fn clear(&self, ns: &str) -> Result<()> {
        for label in ["KGChunk", "KGEntity", "KGMeta"] {
            self.write(
                &format!("MATCH (n:{label} {{ns: $ns}}) DETACH DELETE n"),
                json!({"ns": ns}),
            )?;
        }
        Ok(())
    }

    fn add_chunks(&self, ns: &str, chunks: &[(i64, String)]) -> Result<()> {
        for part in chunks.chunks(GRAPH_BATCH) {
            let rows: Vec<Value> = part.iter().map(|(vid, id)| json!({"vid": vid, "id": id})).collect();
            self.write(
                "UNWIND $rows AS r CREATE (:KGChunk {ns: $ns, vid: r.vid, id: r.id})",
                json!({"ns": ns, "rows": rows}),
            )?;
        }
        Ok(())
    }

    fn remove_chunks(&self, ns: &str, vids: &[i64]) -> Result<()> {
        for part in vids.chunks(GRAPH_BATCH) {
            self.write(
                "MATCH (c:KGChunk) WHERE c.vid IN $vids WITH c WHERE c.ns = $ns DETACH DELETE c",
                json!({"ns": ns, "vids": part}),
            )?;
        }
        Ok(())
    }

    fn add_edges(&self, ns: &str, edges: &[(i64, i64, f64)]) -> Result<()> {
        let keys: Vec<i64> = edges.iter().flat_map(|e| [e.0, e.1]).collect();
        let id = self.node_ids("KGChunk", "vid", ns, &keys)?;
        let rows: Vec<EdgeRow> = edges
            .iter()
            .filter_map(|(a, b, w)| Some((*id.get(a)?, *id.get(b)?, vec![("w", json!(w))])))
            .collect();
        self.add_edges_by_id("SIMILAR", &rows)
    }

    fn add_entities(&self, ns: &str, eids: &[i64]) -> Result<()> {
        let have = self.node_ids("KGEntity", "eid", ns, eids)?;
        let mut new: Vec<i64> = eids.iter().copied().filter(|e| !have.contains_key(e)).collect();
        new.sort_unstable();
        new.dedup();
        for part in new.chunks(GRAPH_BATCH) {
            self.write(
                "UNWIND $eids AS e CREATE (:KGEntity {ns: $ns, eid: e})",
                json!({"ns": ns, "eids": part}),
            )?;
        }
        Ok(())
    }

    fn remove_entities(&self, ns: &str, eids: &[i64]) -> Result<()> {
        for part in eids.chunks(GRAPH_BATCH) {
            self.write(
                "MATCH (e:KGEntity) WHERE e.eid IN $eids WITH e WHERE e.ns = $ns DETACH DELETE e",
                json!({"ns": ns, "eids": part}),
            )?;
        }
        Ok(())
    }

    fn add_relations(&self, ns: &str, relations: &[(i64, i64, String, f64, String)]) -> Result<()> {
        let keys: Vec<i64> = relations.iter().flat_map(|r| [r.0, r.1]).collect();
        let id = self.node_ids("KGEntity", "eid", ns, &keys)?;
        let rows: Vec<EdgeRow> = relations
            .iter()
            .filter_map(|(s, t, ty, w, c)| {
                Some((
                    *id.get(s)?,
                    *id.get(t)?,
                    vec![("type", json!(ty)), ("weight", json!(*w)), ("chunk", json!(c))],
                ))
            })
            .collect();
        self.add_edges_by_id("RELATES", &rows)
    }

    fn remove_relations(&self, ns: &str, sources: &[i64], targets: &[i64], chunk_ids: &[String]) -> Result<()> {
        if sources.is_empty() || targets.is_empty() {
            return Ok(());
        }
        for part in sources.chunks(GRAPH_BATCH) {
            self.write(
                "MATCH (s:KGEntity) WHERE s.eid IN $s WITH s WHERE s.ns = $ns \
                 MATCH (s)-[x:RELATES]->(t:KGEntity) WHERE t.eid IN $t AND x.chunk IN $c DELETE x",
                json!({"ns": ns, "s": part, "t": targets, "c": chunk_ids}),
            )?;
        }
        Ok(())
    }

    fn forget_chunks(&self, ns: &str, chunk_ids: &[String]) -> Result<()> {
        for part in chunk_ids.chunks(GRAPH_BATCH) {
            self.write(
                "MATCH (:KGEntity {ns: $ns})-[x:RELATES]->(:KGEntity) WHERE x.chunk IN $ids DELETE x",
                json!({"ns": ns, "ids": part}),
            )?;
            self.write(
                "MATCH (c:KGChunk) WHERE c.id IN $ids WITH c WHERE c.ns = $ns MATCH (c)-[m:MENTIONS]->(:KGEntity) DELETE m",
                json!({"ns": ns, "ids": part}),
            )?;
        }
        Ok(())
    }

    fn forget_range(&self, ns: &str, lo: &str, hi: Option<&str>) -> Result<()> {
        let (cond_r, cond_m) = match hi {
            Some(_) => (
                "x.chunk <> '' AND x.chunk >= $lo AND x.chunk < $hi",
                "c.id >= $lo AND c.id < $hi",
            ),
            None => ("x.chunk <> '' AND x.chunk >= $lo", "c.id >= $lo"),
        };
        let p = json!({"ns": ns, "lo": lo, "hi": hi});
        self.write(
            &format!("MATCH (:KGEntity {{ns: $ns}})-[x:RELATES]->(:KGEntity) WHERE {cond_r} DELETE x"),
            p.clone(),
        )?;
        self.write(
            &format!("MATCH (c:KGChunk {{ns: $ns}})-[m:MENTIONS]->(:KGEntity) WHERE {cond_m} DELETE m"),
            p,
        )
    }

    fn add_mentions(&self, ns: &str, mentions: &[(i64, i64)]) -> Result<()> {
        let chunk = self.node_ids("KGChunk", "vid", ns, &mentions.iter().map(|m| m.1).collect::<Vec<_>>())?;
        let entity = self.node_ids("KGEntity", "eid", ns, &mentions.iter().map(|m| m.0).collect::<Vec<_>>())?;
        let rows: Vec<EdgeRow> = mentions
            .iter()
            .filter_map(|(e, v)| Some((*chunk.get(v)?, *entity.get(e)?, Vec::new())))
            .collect();
        self.add_edges_by_id("MENTIONS", &rows)
    }

    fn remove_mentions(&self, ns: &str, eids: &[i64], vids: &[i64]) -> Result<()> {
        if eids.is_empty() || vids.is_empty() {
            return Ok(());
        }
        for part in vids.chunks(GRAPH_BATCH) {
            self.write(
                "MATCH (c:KGChunk) WHERE c.vid IN $v WITH c WHERE c.ns = $ns \
                 MATCH (c)-[m:MENTIONS]->(e:KGEntity) WHERE e.eid IN $e DELETE m",
                json!({"ns": ns, "v": part, "e": eids}),
            )?;
        }
        Ok(())
    }

    fn neighbours(&self, ns: &str, vids: &[i64], min_weight: f64) -> Result<Vec<(i64, f64, i64)>> {
        let mut out = Vec::new();
        for part in vids.chunks(GRAPH_BATCH) {
            let rows = self.read(
                "MATCH (a:KGChunk) WHERE a.vid IN $vids WITH a WHERE a.ns = $ns \
                 MATCH (a)-[e:SIMILAR]-(b:KGChunk) WHERE e.w >= $min RETURN b.vid AS b, e.w AS w, a.vid AS a",
                json!({"ns": ns, "vids": part, "min": min_weight}),
            )?;
            out.extend(
                rows.iter()
                    .map(|r| (int(&r[0]), r[1].as_f64().unwrap_or(0.0), int(&r[2]))),
            );
        }
        Ok(out)
    }

    fn relations_around(&self, ns: &str, eids: &[i64], limit: usize) -> Result<Vec<(i64, i64, String, f64)>> {
        if eids.is_empty() {
            return Ok(Vec::new());
        }
        // both directions, each anchored on an indexed list (an OR of the two would scan)
        let mut best: HashMap<(i64, i64, String), f64> = HashMap::new();
        for pattern in [
            "(a)-[x:RELATES]->(o:KGEntity) RETURN a.eid, o.eid",
            "(o:KGEntity)-[x:RELATES]->(a) RETURN o.eid, a.eid",
        ] {
            let (path, ret) = pattern.split_once(" RETURN ").expect("pattern");
            let rows = self.read(
                &format!(
                    "MATCH (a:KGEntity) WHERE a.eid IN $eids WITH a WHERE a.ns = $ns MATCH {path} \
                     RETURN {ret}, x.type, max(x.weight)"
                ),
                json!({"ns": ns, "eids": eids}),
            )?;
            for r in rows {
                let k = (int(&r[0]), int(&r[1]), r[2].as_str().unwrap_or_default().to_string());
                let w = r[3].as_f64().unwrap_or(0.0);
                let e = best.entry(k).or_insert(w);
                *e = e.max(w);
            }
        }
        let mut out: Vec<(i64, i64, String, f64)> = best.into_iter().map(|((s, t, ty), w)| (s, t, ty, w)).collect();
        out.sort_by(|a, b| {
            b.3.total_cmp(&a.3)
                .then(a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });
        out.truncate(limit);
        Ok(out)
    }

    fn chunks_mentioning(&self, ns: &str, eids: &[i64]) -> Result<Vec<(i64, usize)>> {
        if eids.is_empty() {
            return Ok(Vec::new());
        }
        let rows = self.read(
            "MATCH (e:KGEntity) WHERE e.eid IN $eids WITH e WHERE e.ns = $ns \
             MATCH (c:KGChunk)-[:MENTIONS]->(e) RETURN c.vid AS v, count(e) AS n",
            json!({"ns": ns, "eids": eids}),
        )?;
        Ok(rows.iter().map(|r| (int(&r[0]), int(&r[1]) as usize)).collect())
    }
}
