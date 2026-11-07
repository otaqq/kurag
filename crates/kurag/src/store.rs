//! Namespaces: one folder each under the data folder, with a SQLite database (`kurag.db`, WAL) holding the chunks,
//! a float16 copy of every vector, the full-text index, the similarity graph and the knowledge graph. The vector
//! index is either the 8-bit vectors in memory (`Vectors`, loaded from the database when a namespace is opened) or
//! an external backend (`backend.rs`: txtai, HelixDB), filled from the same float16 copies.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use half::f16;
use kurag_core::{ids, score, text, types::Doc};
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};

use crate::backend::{FILL_BATCH, GraphStore, RemoteIndex};
use crate::{fts, kg, sections, vectors::{Range, Vectors}};

pub const DB: &str = "kurag.db";

const SCHEMA: &str = "
create table if not exists meta(key text primary key, value text not null);
create table if not exists chunks(vid integer primary key, id text unique not null, section text not null,
  source text not null default '', text text not null);
create index if not exists chunks_section on chunks(section);
create table if not exists vectors(vid integer primary key, v blob not null);
create table if not exists edges(a integer not null, b integer not null, w real not null, primary key (a, b)) without rowid;
";

#[derive(Debug)]
pub enum Error {
    /// a bad request: 400
    Bad(String),
    Db(rusqlite::Error),
    Io(std::io::Error),
    /// the external vector index failed: 502
    Backend(String),
}

impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Error::Db(e)
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// One namespace. Everything goes through its lock: one reader or writer at a time.
pub struct Namespace {
    pub name: String,
    inner: Mutex<Inner>,
}

pub struct Inner {
    pub db: Connection,
    /// the in-memory index (backend `memory`)
    vecs: Option<Vectors>,
    /// an external index (backend `txtai` or `helixdb`), instead of `vecs`
    remote: Option<Arc<dyn RemoteIndex>>,
    /// a copy of the similarity edges and the knowledge graph in a graph backend (`GraphStore`), walked instead of
    /// the tables
    graph: Option<Arc<dyn GraphStore>>,
    /// the copy may differ from the tables (a write to it failed): filled again before it is read
    graph_stale: bool,
    name: String,
}

fn to_f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| f16::from_f32(x).to_le_bytes()).collect()
}

fn from_f16_bytes(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2).map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32()).collect()
}

fn in_parts(n: usize) -> impl Iterator<Item = std::ops::Range<usize>> {
    (0..n).step_by(500).map(move |i| i..(i + 500).min(n))
}

impl Namespace {
    fn open(name: &str, path: &Path, remote: Option<Arc<dyn RemoteIndex>>, graph: Option<Arc<dyn GraphStore>>) -> Result<Namespace> {
        std::fs::create_dir_all(path)?;
        let db = Connection::open(path.join(DB))?;
        db.query_row("pragma journal_mode=wal", [], |_| Ok(()))?;
        db.execute_batch("pragma synchronous=normal")?;
        db.execute_batch(&format!("{SCHEMA}{}{}{}", fts::SCHEMA, kg::SCHEMA, sections::SCHEMA))?;
        let mut inner = Inner { db, vecs: None, remote, graph, graph_stale: false, name: name.to_string() };
        inner.load_vectors()?;
        if inner.graph.is_some() {
            inner.sync_graph(false)?;
        }
        Ok(Namespace { name: name.to_string(), inner: Mutex::new(inner) })
    }

    pub fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Inner {
    /// Loads the 8-bit vectors from the stored float16 copies, learning the ranges from them first (two passes,
    /// so the float copies are never all in memory at once).
    fn load_vectors(&mut self) -> Result<usize> {
        if self.remote.is_some() {
            return self.sync_remote(false);
        }
        let mut range: Option<Range> = None;
        {
            let mut st = self.db.prepare("select v from vectors")?;
            let mut rows = st.query([])?;
            while let Some(r) = rows.next()? {
                let v = from_f16_bytes(&r.get::<_, Vec<u8>>(0)?);
                range.get_or_insert_with(|| Range::new(v.len())).add(&v);
            }
        }
        let Some(range) = range else {
            self.vecs = None;
            return Ok(0);
        };
        let mut vecs = Vectors::new(&range);
        {
            let mut st = self.db.prepare("select vid, v from vectors order by vid")?;
            let mut rows = st.query([])?;
            while let Some(r) = rows.next()? {
                vecs.add(r.get(0)?, &from_f16_bytes(&r.get::<_, Vec<u8>>(1)?));
            }
        }
        let n = vecs.len();
        self.vecs = Some(vecs);
        Ok(n)
    }

    /// Fills the external index from the stored copies when it does not hold as many vectors as the database, or
    /// always when `force`; the number of vectors.
    fn sync_remote(&mut self, force: bool) -> Result<usize> {
        let remote = self.remote.clone().expect("external index");
        let n: i64 = self.db.query_row("select count(*) from vectors", [], |r| r.get(0))?;
        let held = remote.count(&self.name).map_err(Error::Backend)?;
        if !force && held == n as usize {
            return Ok(held);
        }
        remote.clear(&self.name).map_err(Error::Backend)?;
        let mut st = self.db.prepare("select vid, v from vectors order by vid")?;
        let mut rows = st.query([])?;
        let mut batch: Vec<(i64, Vec<f32>)> = Vec::with_capacity(FILL_BATCH);
        let send = |batch: &mut Vec<(i64, Vec<f32>)>| -> Result<()> {
            let items: Vec<(i64, &[f32])> = batch.iter().map(|(vid, v)| (*vid, v.as_slice())).collect();
            remote.fill(&self.name, &items).map_err(Error::Backend)?;
            batch.clear();
            Ok(())
        };
        while let Some(r) = rows.next()? {
            batch.push((r.get(0)?, from_f16_bytes(&r.get::<_, Vec<u8>>(1)?)));
            if batch.len() == FILL_BATCH {
                send(&mut batch)?;
            }
        }
        send(&mut batch)?;
        remote.fill_done(&self.name).map_err(Error::Backend)?;
        Ok(n as usize)
    }

    /// Builds the vector index again from the stored copies, and forgets the section labels; the number of
    /// vectors.
    pub fn rebuild(&mut self) -> Result<usize> {
        sections::forget_all(&self.db)?;
        if self.graph.is_some() {
            self.sync_graph(true)?;
        }
        if self.remote.is_some() {
            return self.sync_remote(true);
        }
        self.load_vectors()
    }

    // ---- the copy of the graph in a graph backend ----

    fn graph_version(&self) -> Result<i64> {
        let v: Option<String> =
            self.db.query_row("select value from meta where key = 'graph_version'", [], |r| r.get(0)).optional()?;
        Ok(v.and_then(|v| v.parse().ok()).unwrap_or(0))
    }

    /// The next version of the graph, in the caller's transaction.
    fn bump_graph(&self, tx: &Connection) -> Result<i64> {
        let v = self.graph_version()? + 1;
        tx.execute("insert or replace into meta(key, value) values ('graph_version', ?1)", [v.to_string()])?;
        Ok(v)
    }

    /// Fills the copy from the tables when it is at another version, or always when `force`.
    fn sync_graph(&mut self, force: bool) -> Result<()> {
        let graph = self.graph.clone().expect("graph backend");
        let ns = self.name.clone();
        let fail = Error::Backend;
        let version = self.graph_version()?;
        if !force && graph.version(&ns).map_err(fail)? == Some(version) {
            self.graph_stale = false;
            return Ok(());
        }
        graph.clear(&ns).map_err(fail)?;
        let all = |sql: &str| -> Result<Vec<(i64, i64, f64)>> {
            Ok(self.db.prepare(sql)?.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<rusqlite::Result<_>>()?)
        };
        let chunks: Vec<(i64, String)> =
            self.db.prepare("select vid, id from chunks")?.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<_>>()?;
        graph.add_chunks(&ns, &chunks).map_err(fail)?;
        graph.add_edges(&ns, &all("select a, b, w from edges where a < b")?).map_err(fail)?;
        let eids: Vec<i64> = self.db.prepare("select eid from entities")?.query_map([], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?;
        graph.add_entities(&ns, &eids).map_err(fail)?;
        let relations: Vec<(i64, i64, String, f64, String)> = self
            .db
            .prepare("select source, target, type, weight, chunk from relations")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?
            .collect::<rusqlite::Result<_>>()?;
        graph.add_relations(&ns, &relations).map_err(fail)?;
        let mentions: Vec<(i64, i64)> = self
            .db
            .prepare("select m.eid, c.vid from mentions m join chunks c on c.id = m.chunk")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        graph.add_mentions(&ns, &mentions).map_err(fail)?;
        graph.set_version(&ns, version).map_err(fail)?;
        self.graph_stale = false;
        Ok(())
    }

    /// A write to the copy failed (its transaction rolled back): fill it again now, or before it is next read.
    fn graph_failed(&mut self) {
        self.graph_stale = true;
        let _ = self.sync_graph(true);
    }

    /// The graph backend, its copy brought up to date first; None without one.
    fn fresh_graph(&mut self) -> Result<Option<Arc<dyn GraphStore>>> {
        if self.graph.is_none() {
            return Ok(None);
        }
        if self.graph_stale {
            self.sync_graph(true)?;
        }
        Ok(self.graph.clone())
    }

    /// (entity, chunk row id) for mentions of these chunk ids still in the tables.
    fn mentions_of(&self, chunk_ids: &[&str]) -> Result<Vec<(i64, i64)>> {
        let mut out = Vec::new();
        for part in chunk_ids.chunks(500) {
            let sql = format!("select m.eid, c.vid from mentions m join chunks c on c.id = m.chunk where m.chunk in ({})", kg::marks(part.len()));
            out.extend(self.db.prepare(&sql)?.query_map(params_from_iter(part), |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<Vec<_>>>()?);
        }
        Ok(out)
    }

    /// Upserted chunks in the copy (in the caller's transaction, before it commits): old rows out, new rows and
    /// their edges in, and the mentions the tables keep for those chunk ids.
    fn graph_chunks(&self, tx: &Connection, removed: &[i64], added: &[(i64, String)], pairs: &HashMap<(i64, i64), f64>) -> Result<()> {
        let Some(graph) = &self.graph else { return Ok(()) };
        let (ns, fail) = (&self.name, Error::Backend);
        let version = self.bump_graph(tx)?;
        graph.remove_chunks(ns, removed).map_err(fail)?;
        graph.add_chunks(ns, added).map_err(fail)?;
        let edges: Vec<(i64, i64, f64)> = pairs.iter().map(|(&(a, b), &w)| (a, b, w)).collect();
        graph.add_edges(ns, &edges).map_err(fail)?;
        let ids: Vec<&str> = added.iter().map(|(_, id)| id.as_str()).collect();
        graph.add_mentions(ns, &self.mentions_of(&ids)?).map_err(fail)?;
        graph.set_version(ns, version).map_err(fail)
    }

    /// Deleted chunks and what the knowledge graph learned from the id ranges, in the copy.
    fn graph_delete(&self, tx: &Connection, vids: &[i64], ranges: &[(String, Option<String>)], removed: &[i64]) -> Result<()> {
        let Some(graph) = &self.graph else { return Ok(()) };
        let (ns, fail) = (&self.name, Error::Backend);
        let version = self.bump_graph(tx)?;
        graph.remove_chunks(ns, vids).map_err(fail)?;
        for (lo, hi) in ranges {
            graph.forget_range(ns, lo, hi.as_deref()).map_err(fail)?;
        }
        graph.remove_entities(ns, removed).map_err(fail)?;
        graph.set_version(ns, version).map_err(fail)
    }

    /// `/kg/upsert`: the tables, then the copy, in one transaction.
    pub fn kg_upsert(&mut self, req: &kurag_core::types::KgUpsert) -> Result<kurag_core::types::KgCounts> {
        let result = (|| -> Result<kurag_core::types::KgCounts> {
            let tx = self.db.unchecked_transaction()?;
            // the sections' labels may show these entities
            let chunks: Vec<&str> = req.done.iter().map(String::as_str)
                .chain(req.mentions.iter().map(|m| m.chunk.as_str()))
                .chain(req.relations.iter().map(|r| r.chunk.as_str()).filter(|c| !c.is_empty()))
                .collect();
            sections::forget_chunks(&tx, &chunks)?;
            let (counts, change) = kg::upsert(&tx, req)?;
            if let Some(graph) = &self.graph {
                let (ns, fail) = (&self.name, Error::Backend);
                let version = self.bump_graph(&tx)?;
                graph.forget_chunks(ns, &change.done).map_err(fail)?;
                graph.add_entities(ns, &change.entities).map_err(fail)?;
                // relations and mentions sent again replace themselves: out with the set they belong to, back in
                // from the tables
                let set = |v: Vec<i64>| -> Vec<i64> { v.into_iter().collect::<HashSet<_>>().into_iter().collect() };
                if !change.relations.is_empty() {
                    let sources = set(change.relations.iter().map(|r| r.0).collect());
                    let targets = set(change.relations.iter().map(|r| r.1).collect());
                    let chunks: Vec<String> =
                        change.relations.iter().map(|r| r.4.clone()).collect::<HashSet<_>>().into_iter().collect();
                    graph.remove_relations(ns, &sources, &targets, &chunks).map_err(fail)?;
                    graph.add_relations(ns, &self.relations_among(&tx, &sources, &targets, &chunks)?).map_err(fail)?;
                }
                if !change.mentions.is_empty() {
                    let ids: Vec<&str> = change.mentions.iter().map(|(_, c)| c.as_str()).collect();
                    let vids = set(self.vids_by_id(&ids)?.into_values().collect());
                    let eids = set(change.mentions.iter().map(|m| m.0).collect());
                    graph.remove_mentions(ns, &eids, &vids).map_err(fail)?;
                    graph.add_mentions(ns, &self.mentions_among(&tx, &eids, &vids)?).map_err(fail)?;
                }
                graph.remove_entities(ns, &change.removed).map_err(fail)?;
                graph.set_version(ns, version).map_err(fail)?;
            }
            tx.commit()?;
            Ok(counts)
        })();
        if result.is_err() && self.graph.is_some() {
            self.graph_failed();
        }
        result
    }

    /// `/kg/delete`: (relations, mentions, entities) removed.
    pub fn kg_delete(&mut self, ranges: &[(String, Option<String>)]) -> Result<(i64, i64, i64)> {
        let result = (|| -> Result<(i64, i64, i64)> {
            let tx = self.db.unchecked_transaction()?;
            sections::forget_ranges(&tx, ranges)?;
            let (relations, mentions, removed) = kg::delete(&tx, ranges)?;
            self.graph_delete(&tx, &[], ranges, &removed)?;
            tx.commit()?;
            Ok((relations, mentions, removed.len() as i64))
        })();
        if result.is_err() && self.graph.is_some() {
            self.graph_failed();
        }
        result
    }

    /// The relations in the tables from these sources to these targets learned from these chunks.
    fn relations_among(&self, db: &Connection, sources: &[i64], targets: &[i64], chunks: &[String]) -> Result<Vec<(i64, i64, String, f64, String)>> {
        let sql = format!(
            "select source, target, type, weight, chunk from relations where source in ({}) and target in ({}) and chunk in ({})",
            kg::marks(sources.len()),
            kg::marks(targets.len()),
            kg::marks(chunks.len())
        );
        let mut args: Vec<rusqlite::types::Value> = sources.iter().chain(targets).map(|&v| v.into()).collect();
        args.extend(chunks.iter().map(|c| c.clone().into()));
        Ok(db.prepare(&sql)?.query_map(params_from_iter(args), |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))?.collect::<rusqlite::Result<_>>()?)
    }

    /// The mentions in the tables of these entities in these chunks: (entity, chunk row id).
    fn mentions_among(&self, db: &Connection, eids: &[i64], vids: &[i64]) -> Result<Vec<(i64, i64)>> {
        if eids.is_empty() || vids.is_empty() {
            return Ok(Vec::new());
        }
        let sql = format!(
            "select m.eid, c.vid from mentions m join chunks c on c.id = m.chunk where m.eid in ({}) and c.vid in ({})",
            kg::marks(eids.len()),
            kg::marks(vids.len())
        );
        let args: Vec<i64> = eids.iter().chain(vids).copied().collect();
        Ok(db.prepare(&sql)?.query_map(params_from_iter(args), |r| Ok((r.get(0)?, r.get(1)?)))?.collect::<rusqlite::Result<_>>()?)
    }

    fn vids_by_id(&self, chunk_ids: &[&str]) -> Result<HashMap<String, i64>> {
        let mut out = HashMap::new();
        for part in chunk_ids.chunks(500) {
            let sql = format!("select id, vid from chunks where id in ({})", kg::marks(part.len()));
            for row in self.db.prepare(&sql)?.query_map(params_from_iter(part), |r| Ok((r.get::<_, String>(0)?, r.get(1)?)))? {
                let (id, vid) = row?;
                out.insert(id, vid);
            }
        }
        Ok(out)
    }

    /// `/kg/search` around the found entities: the walk in the graph backend, or in the tables.
    pub fn kg_around(&mut self, eids: &[i64], depth: usize, limit: usize) -> Result<kg::Around> {
        let graph = self.fresh_graph()?;
        let (db, ns) = (&self.db, self.name.clone());
        let step = |e: &[i64], l: usize| match &graph {
            Some(g) => g.relations_around(&ns, e, l),
            None => kg::relations_around(db, e, l).map_err(|e| e.to_string()),
        };
        kg::around(db, eids, depth, limit, &step).map_err(Error::Backend)
    }

    /// Chunks the knowledge graph ties to the query: entities named in it (`kg::find`), walked `depth` steps, and
    /// the chunks that mention them; (chunk row id, how many of the entities it mentions).
    pub fn kg_chunks(&mut self, query: &str, depth: usize, limit: usize) -> Result<Vec<(i64, usize)>> {
        let found = kg::find(&self.db, query, limit)?;
        if found.is_empty() {
            return Ok(Vec::new());
        }
        let graph = self.fresh_graph()?;
        let (db, ns) = (&self.db, self.name.clone());
        let step = |e: &[i64], l: usize| match &graph {
            Some(g) => g.relations_around(&ns, e, l),
            None => kg::relations_around(db, e, l).map_err(|e| e.to_string()),
        };
        let (reached, _) = kg::walk(&found, depth, limit * 5, &step).map_err(Error::Backend)?;
        match &graph {
            Some(g) => g.chunks_mentioning(&ns, &reached).map_err(Error::Backend),
            None => Ok(kg::chunks_mentioning(db, &reached)?),
        }
    }

    /// Dimensions of the stored vectors (float16: two bytes each), None when there is none.
    fn stored_dims(&self) -> Result<Option<usize>> {
        Ok(self.db.query_row("select length(v) / 2 from vectors limit 1", [], |r| r.get::<_, i64>(0)).optional()?.map(|d| d as usize))
    }

    /// The stored float16 copy of a chunk's vector.
    fn stored_vector(&self, vid: i64) -> Result<Option<Vec<f32>>> {
        Ok(self
            .db
            .prepare_cached("select v from vectors where vid = ?1")?
            .query_row([vid], |r| r.get::<_, Vec<u8>>(0))
            .optional()?
            .map(|b| from_f16_bytes(&b)))
    }

    pub fn count(&self) -> Result<i64> {
        Ok(self.db.query_row("select count(*) from chunks", [], |r| r.get(0))?)
    }

    /// Nothing left: no chunk and no entity.
    pub fn empty(&self) -> Result<bool> {
        let entity: Option<i64> = self.db.query_row("select 1 from entities limit 1", [], |r| r.get(0)).optional()?;
        Ok(self.count()? == 0 && entity.is_none())
    }

    /// Stored vectors of the chunks with these ids, when their text is `text`: they need no embedding again.
    pub fn unchanged(&self, docs: &[Doc]) -> Result<HashMap<String, Vec<f32>>> {
        let mut out = HashMap::new();
        let mut st = self.db.prepare_cached("select c.text, v.v from chunks c join vectors v on v.vid = c.vid where c.id = ?1")?;
        for d in docs {
            let row = st.query_row([&d.id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, Vec<u8>>(1)?))).optional()?;
            if let Some((t, v)) = row
                && t == d.text
            {
                out.insert(d.id.clone(), from_f16_bytes(&v));
            }
        }
        Ok(out)
    }

    fn vids_of_ids(&self, chunk_ids: &[&str]) -> Result<Vec<i64>> {
        let mut out = Vec::new();
        for r in in_parts(chunk_ids.len()) {
            let part = &chunk_ids[r];
            let mut st = self.db.prepare(&format!("select vid from chunks where id in ({})", kg::marks(part.len())))?;
            out.extend(st.query_map(params_from_iter(part), |r| r.get::<_, i64>(0))?.collect::<rusqlite::Result<Vec<_>>>()?);
        }
        Ok(out)
    }

    fn vids_with_prefixes(&self, prefixes: &[String]) -> Result<Vec<i64>> {
        let mut out = Vec::new();
        for p in prefixes {
            let (lo, hi) = ids::prefix_range(p);
            let vids: Vec<i64> = match hi {
                None => self.db.prepare("select vid from chunks where id >= ?1")?.query_map([lo], |r| r.get(0))?.collect::<rusqlite::Result<_>>()?,
                Some(hi) => self
                    .db
                    .prepare("select vid from chunks where id >= ?1 and id < ?2")?
                    .query_map([lo, hi], |r| r.get(0))?
                    .collect::<rusqlite::Result<_>>()?,
            };
            out.extend(vids);
        }
        out.sort_unstable();
        out.dedup();
        Ok(out)
    }

    /// Removes chunks from the database (in the caller's transaction): rows, vectors, full text, edges, and the
    /// labels of their sections.
    fn remove_rows(&self, vids: &[i64]) -> Result<()> {
        for r in in_parts(vids.len()) {
            let part = &vids[r];
            sections::forget_vids(&self.db, part)?;
            let m = kg::marks(part.len());
            for sql in [
                format!("delete from edges where a in ({m})"),
                format!("delete from edges where b in ({m})"),
                format!("delete from chunks where vid in ({m})"),
                format!("delete from vectors where vid in ({m})"),
            ] {
                self.db.execute(&sql, params_from_iter(part))?;
            }
            fts::remove(&self.db, "fts", part)?;
        }
        Ok(())
    }

    /// Adds the chunks with their vectors (replacing chunks with the same id), links each to its nearest chunks;
    /// the number of chunks. One transaction: on failure nothing changes.
    pub fn upsert(&mut self, docs: &[Doc], vectors: &[Vec<f32>], neighbours: usize, edge_min: f32) -> Result<i64> {
        let dims = vectors.first().map_or(0, Vec::len);
        if let Some(have) = self.stored_dims()?
            && have != dims
        {
            return Err(Error::Bad(format!(
                "vectors of {dims} dimensions, the namespace has {have}: another embedding model?"
            )));
        }
        let result = self.upsert_tx(docs, vectors, neighbours, edge_min);
        if result.is_err() && self.graph.is_some() {
            self.graph_failed();
        }
        match &result {
            // the transaction was rolled back: the index follows the database
            Err(_) if self.remote.is_some() => {
                self.sync_remote(true)?;
            }
            Err(_) => {
                self.load_vectors()?;
            }
            // ranges learned on a few vectors: learn them again on all of them once their number has doubled
            Ok(_) if self.vecs.as_ref().is_some_and(|v| v.len() >= 2 * v.trained_on().max(1)) => {
                self.load_vectors()?;
            }
            Ok(_) => {}
        }
        result
    }

    fn upsert_tx(&mut self, docs: &[Doc], vectors: &[Vec<f32>], neighbours: usize, edge_min: f32) -> Result<i64> {
        let tx = self.db.unchecked_transaction()?;
        let old = self.vids_of_ids(&docs.iter().map(|d| d.id.as_str()).collect::<Vec<_>>())?;
        self.remove_rows(&old)?;
        if let Some(remote) = self.remote.clone() {
            let (vids, pairs) = self.upsert_remote(&tx, &remote, &old, docs, vectors, neighbours, edge_min)?;
            let added: Vec<(i64, String)> = vids.iter().copied().zip(docs.iter().map(|d| d.id.clone())).collect();
            self.graph_chunks(&tx, &old, &added, &pairs)?;
            tx.commit()?;
            return self.count();
        }
        let vecs = self.vecs.get_or_insert_with(|| {
            // a new namespace: ranges from this first batch
            let mut range = Range::new(vectors[0].len());
            vectors.iter().for_each(|v| range.add(v));
            Vectors::new(&range)
        });
        old.iter().for_each(|&vid| vecs.remove(vid));
        let mut vids = Vec::with_capacity(docs.len());
        {
            let mut chunk = tx.prepare_cached("insert into chunks(id, section, source, text) values (?1, ?2, ?3, ?4)")?;
            let mut vector = tx.prepare_cached("insert into vectors(vid, v) values (?1, ?2)")?;
            for (d, v) in docs.iter().zip(vectors) {
                chunk.execute(params![d.id, ids::section_of(&d.id), d.source, d.text])?;
                let vid = tx.last_insert_rowid();
                vector.execute(params![vid, to_f16_bytes(v)])?;
                vecs.add(vid, v);
                vids.push(vid);
            }
        }
        sections::forget_vids(&tx, &vids)?;
        let rows: Vec<(i64, &str)> = vids.iter().copied().zip(docs.iter().map(|d| d.text.as_str())).collect();
        fts::add(&tx, "fts", &rows)?;
        // link the new chunks to their nearest chunks, both directions
        let mut pairs: HashMap<(i64, i64), f64> = HashMap::new();
        {
            let mut edge = tx.prepare_cached("insert or replace into edges(a, b, w) values (?1, ?2, ?3)")?;
            for (&vid, v) in vids.iter().zip(vectors) {
                for (other, s) in vecs.nearest(v, neighbours + 1) {
                    if other != vid && s >= edge_min {
                        edge.execute(params![vid, other, s as f64])?;
                        edge.execute(params![other, vid, s as f64])?;
                        pairs.insert((vid.min(other), vid.max(other)), s as f64);
                    }
                }
            }
        }
        let added: Vec<(i64, String)> = vids.iter().copied().zip(docs.iter().map(|d| d.id.clone())).collect();
        self.graph_chunks(&tx, &old, &added, &pairs)?;
        tx.commit()?;
        self.count()
    }

    /// `upsert_tx` with an external index: the rows are written in the transaction, the vectors sent to the index
    /// before it commits (a failure rolls the rows back and the caller fills the index again).
    #[allow(clippy::too_many_arguments)]
    fn upsert_remote(
        &self,
        tx: &rusqlite::Transaction<'_>,
        remote: &Arc<dyn RemoteIndex>,
        old: &[i64],
        docs: &[Doc],
        vectors: &[Vec<f32>],
        neighbours: usize,
        edge_min: f32,
    ) -> Result<(Vec<i64>, HashMap<(i64, i64), f64>)> {
        remote.remove(&self.name, old).map_err(Error::Backend)?;
        let mut vids = Vec::with_capacity(docs.len());
        {
            let mut chunk = tx.prepare_cached("insert into chunks(id, section, source, text) values (?1, ?2, ?3, ?4)")?;
            let mut vector = tx.prepare_cached("insert into vectors(vid, v) values (?1, ?2)")?;
            for (d, v) in docs.iter().zip(vectors) {
                chunk.execute(params![d.id, ids::section_of(&d.id), d.source, d.text])?;
                let vid = tx.last_insert_rowid();
                vector.execute(params![vid, to_f16_bytes(v)])?;
                vids.push(vid);
            }
        }
        for part in in_parts(vids.len()) {
            let items: Vec<(i64, &[f32])> = part.clone().map(|i| (vids[i], vectors[i].as_slice())).collect();
            remote.add(&self.name, &items).map_err(Error::Backend)?;
        }
        sections::forget_vids(tx, &vids)?;
        let rows: Vec<(i64, &str)> = vids.iter().copied().zip(docs.iter().map(|d| d.text.as_str())).collect();
        fts::add(tx, "fts", &rows)?;
        let mut pairs: HashMap<(i64, i64), f64> = HashMap::new();
        {
            let mut edge = tx.prepare_cached("insert or replace into edges(a, b, w) values (?1, ?2, ?3)")?;
            for (&vid, v) in vids.iter().zip(vectors) {
                for (other, s) in remote.neighbours(&self.name, v, neighbours + 1).map_err(Error::Backend)? {
                    if other != vid && s >= edge_min {
                        edge.execute(params![vid, other, s as f64])?;
                        edge.execute(params![other, vid, s as f64])?;
                        pairs.insert((vid.min(other), vid.max(other)), s as f64);
                    }
                }
            }
        }
        Ok((vids, pairs))
    }

    /// Deletes the chunks with these id prefixes, and what the knowledge graph learned from them; how many chunks.
    pub fn delete(&mut self, prefixes: &[String]) -> Result<usize> {
        let tx = self.db.unchecked_transaction()?;
        let vids = self.vids_with_prefixes(prefixes)?;
        self.remove_rows(&vids)?;
        let ranges: Vec<_> = prefixes.iter().map(|p| ids::prefix_range(p)).collect();
        let (_, _, removed) = kg::delete(&tx, &ranges)?;
        if let Err(e) = self.graph_delete(&tx, &vids, &ranges, &removed) {
            drop(tx);
            self.graph_failed();
            return Err(e);
        }
        tx.commit()?;
        if let Some(v) = &mut self.vecs {
            vids.iter().for_each(|&vid| v.remove(vid));
        }
        if let Some(remote) = self.remote.clone() {
            let removed = in_parts(vids.len()).try_for_each(|r| remote.remove(&self.name, &vids[r]));
            if let Err(e) = removed {
                // the rows are gone: fill the index again from what is left, or report both failures
                self.sync_remote(true).map_err(|again| Error::Backend(format!("{e}; filling it again: {again:?}")))?;
            }
        }
        Ok(vids.len())
    }

    /// (id, source, text) of the chunks.
    pub fn chunks(&self, vids: &[i64]) -> Result<HashMap<i64, (String, String, String)>> {
        let mut out = HashMap::new();
        for r in in_parts(vids.len()) {
            let part = &vids[r];
            let mut st = self.db.prepare(&format!("select vid, id, source, text from chunks where vid in ({})", kg::marks(part.len())))?;
            let rows = st.query_map(params_from_iter(part), |r| Ok((r.get::<_, i64>(0)?, (r.get(1)?, r.get(2)?, r.get(3)?))))?;
            for row in rows {
                let (vid, c) = row?;
                out.insert(vid, c);
            }
        }
        Ok(out)
    }

    /// (vid, cosine similarity), best first.
    pub fn vector_search(&self, q: &[f32], limit: usize) -> Result<Vec<(i64, f64)>> {
        if let Some(remote) = &self.remote {
            let hits = remote.nearest(&self.name, q, limit).map_err(Error::Backend)?;
            return Ok(hits.into_iter().map(|(vid, s)| (vid, s as f64)).collect());
        }
        Ok(self.vecs.as_ref().map_or(Vec::new(), |v| v.nearest(q, limit).into_iter().map(|(vid, s)| (vid, s as f64)).collect()))
    }

    /// Cosine similarity of the query to a chunk (with an external index, from the stored float16 copy).
    pub fn similarity(&self, q: &[f32], vid: i64) -> Option<f64> {
        if self.remote.is_some() {
            let v = self.stored_vector(vid).ok()??;
            return Some(q.iter().zip(&v).map(|(a, b)| (a * b) as f64).sum());
        }
        self.vecs.as_ref()?.similarity(q, vid).map(f64::from)
    }

    /// (vid, text score), best first.
    pub fn text_search(&self, query: &str, limit: usize) -> Result<Vec<(i64, f64)>> {
        let terms = text::terms(query);
        let hits = fts::search(&self.db, "fts", query, limit)?;
        let texts = self.chunks(&hits.iter().map(|h| h.0).collect::<Vec<_>>())?;
        let mut scored: Vec<(i64, f64)> = hits
            .into_iter()
            .map(|(vid, bm25)| {
                let t = texts.get(&vid).map_or("", |c| c.2.as_str());
                (vid, score::text_score(text::coverage(&terms, t), bm25))
            })
            .collect();
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(scored)
    }

    /// (neighbour, weight, the chunk it hangs on) for the edges from the given chunks.
    pub fn neighbours(&mut self, vids: &[i64], min_weight: f64) -> Result<Vec<(i64, f64, i64)>> {
        if vids.is_empty() {
            return Ok(Vec::new());
        }
        if let Some(graph) = self.fresh_graph()? {
            return graph.neighbours(&self.name, vids, min_weight).map_err(Error::Backend);
        }
        let mut st = self.db.prepare(&format!("select b, w, a from edges where a in ({}) and w >= ?", kg::marks(vids.len())))?;
        let mut args: Vec<rusqlite::types::Value> = vids.iter().map(|&v| v.into()).collect();
        args.push(min_weight.into());
        Ok(st.query_map(params_from_iter(args), |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?.collect::<rusqlite::Result<_>>()?)
    }
}

/// All namespaces under the data folder; opened on first use.
pub struct Store {
    data: PathBuf,
    open: Mutex<HashMap<String, Arc<Namespace>>>,
    remote: Option<Arc<dyn RemoteIndex>>,
    graph: Option<Arc<dyn GraphStore>>,
}

impl Store {
    /// The store; `remote` is the external vector index, None for the in-memory one.
    pub fn new(data: PathBuf, remote: Option<Arc<dyn RemoteIndex>>) -> Store {
        Store { data, open: Mutex::new(HashMap::new()), remote, graph: None }
    }

    /// The store with a graph backend (None: the namespaces' own tables).
    pub fn with_graph(mut self, graph: Option<Arc<dyn GraphStore>>) -> Store {
        self.graph = graph;
        self
    }

    fn registry(&self) -> MutexGuard<'_, HashMap<String, Arc<Namespace>>> {
        self.open.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn check(name: &str) -> Result<()> {
        if ids::valid_namespace(name) { Ok(()) } else { Err(Error::Bad(format!("bad namespace name: {name}"))) }
    }

    /// Names of the namespaces on disk, sorted.
    pub fn names(&self) -> Vec<String> {
        let Ok(dir) = std::fs::read_dir(&self.data) else { return Vec::new() };
        let mut out: Vec<String> = dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().join(DB).exists())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| ids::valid_namespace(n))
            .collect();
        out.sort();
        out
    }

    /// The namespace, opened if needed; None when it does not exist and `create` is false.
    pub fn get(&self, name: &str, create: bool) -> Result<Option<Arc<Namespace>>> {
        Store::check(name)?;
        let mut open = self.registry();
        if let Some(ns) = open.get(name) {
            return Ok(Some(ns.clone()));
        }
        let path = self.data.join(name);
        if !path.join(DB).exists() && !create {
            return Ok(None);
        }
        let ns = Arc::new(Namespace::open(name, &path, self.remote.clone(), self.graph.clone())?);
        open.insert(name.to_string(), ns.clone());
        Ok(Some(ns))
    }

    /// Removes a namespace and its folder.
    pub fn drop_namespace(&self, name: &str) -> Result<()> {
        self.registry().remove(name);
        if let Some(remote) = &self.remote {
            remote.clear(name).map_err(Error::Backend)?;
        }
        if let Some(graph) = &self.graph {
            graph.clear(name).map_err(Error::Backend)?;
        }
        let path = self.data.join(name);
        if path.exists() {
            std::fs::remove_dir_all(path)?;
        }
        Ok(())
    }

    /// Namespaces to search: all of them, or each asked one with everything under it, minus the excluded ones.
    pub fn expand(&self, asked: Option<&[String]>, exclude: Option<&[String]>) -> Vec<String> {
        ids::expand(&self.names(), asked, exclude)
    }
}
