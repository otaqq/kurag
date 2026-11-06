//! Namespaces: one folder each under the data folder, with a SQLite database (`kurag.db`, WAL) holding the chunks,
//! a float16 copy of every vector, the full-text index, the similarity graph and the knowledge graph. The 8-bit
//! vectors in memory (`Vectors`) are loaded from the database when a namespace is opened.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use half::f16;
use kurag_core::{ids, score, text, types::Doc};
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};

use crate::{fts, kg, vectors::Vectors};

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
    vecs: Option<Vectors>,
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
    fn open(name: &str, path: &Path) -> Result<Namespace> {
        std::fs::create_dir_all(path)?;
        let db = Connection::open(path.join(DB))?;
        db.query_row("pragma journal_mode=wal", [], |_| Ok(()))?;
        db.execute_batch("pragma synchronous=normal")?;
        db.execute_batch(&format!("{SCHEMA}{}{}", fts::SCHEMA, kg::SCHEMA))?;
        let mut inner = Inner { db, vecs: None };
        inner.load_vectors()?;
        Ok(Namespace { name: name.to_string(), inner: Mutex::new(inner) })
    }

    pub fn lock(&self) -> MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Inner {
    /// Loads the 8-bit vectors from the stored float16 copies.
    fn load_vectors(&mut self) -> Result<usize> {
        let mut st = self.db.prepare("select vid, v from vectors order by vid")?;
        let mut rows = st.query([])?;
        let mut vecs: Option<Vectors> = None;
        while let Some(r) = rows.next()? {
            let v = from_f16_bytes(&r.get::<_, Vec<u8>>(1)?);
            vecs.get_or_insert_with(|| Vectors::new(v.len())).add(r.get(0)?, &v);
        }
        let n = vecs.as_ref().map_or(0, Vectors::len);
        drop(rows);
        drop(st);
        self.vecs = vecs;
        Ok(n)
    }

    /// Builds the vectors in memory again from the stored copies; the number of vectors.
    pub fn rebuild(&mut self) -> Result<usize> {
        self.load_vectors()
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

    /// Removes chunks from the database (in the caller's transaction): rows, vectors, full text, edges.
    fn remove_rows(&self, vids: &[i64]) -> Result<()> {
        for r in in_parts(vids.len()) {
            let part = &vids[r];
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
        if let Some(v) = &self.vecs
            && v.len() > 0
            && v.dims() != dims
        {
            return Err(Error::Bad(format!(
                "vectors of {dims} dimensions, the namespace has {}: another embedding model?",
                v.dims()
            )));
        }
        let result = self.upsert_tx(docs, vectors, neighbours, edge_min);
        if result.is_err() {
            self.load_vectors()?; // the transaction was rolled back: memory follows the database
        }
        result
    }

    fn upsert_tx(&mut self, docs: &[Doc], vectors: &[Vec<f32>], neighbours: usize, edge_min: f32) -> Result<i64> {
        let tx = self.db.unchecked_transaction()?;
        let old = self.vids_of_ids(&docs.iter().map(|d| d.id.as_str()).collect::<Vec<_>>())?;
        self.remove_rows(&old)?;
        let vecs = self.vecs.get_or_insert_with(|| Vectors::new(vectors[0].len()));
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
        let rows: Vec<(i64, &str)> = vids.iter().copied().zip(docs.iter().map(|d| d.text.as_str())).collect();
        fts::add(&tx, "fts", &rows)?;
        // link the new chunks to their nearest chunks, both directions
        {
            let mut edge = tx.prepare_cached("insert or replace into edges(a, b, w) values (?1, ?2, ?3)")?;
            for (&vid, v) in vids.iter().zip(vectors) {
                for (other, s) in vecs.nearest(v, neighbours + 1) {
                    if other != vid && s >= edge_min {
                        edge.execute(params![vid, other, s as f64])?;
                        edge.execute(params![other, vid, s as f64])?;
                    }
                }
            }
        }
        tx.commit()?;
        self.count()
    }

    /// Deletes the chunks with these id prefixes, and what the knowledge graph learned from them; how many chunks.
    pub fn delete(&mut self, prefixes: &[String]) -> Result<usize> {
        let tx = self.db.unchecked_transaction()?;
        let vids = self.vids_with_prefixes(prefixes)?;
        self.remove_rows(&vids)?;
        kg::delete(&tx, &prefixes.iter().map(|p| ids::prefix_range(p)).collect::<Vec<_>>())?;
        tx.commit()?;
        if let Some(v) = &mut self.vecs {
            vids.iter().for_each(|&vid| v.remove(vid));
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
    pub fn vector_search(&self, q: &[f32], limit: usize) -> Vec<(i64, f64)> {
        self.vecs.as_ref().map_or(Vec::new(), |v| v.nearest(q, limit).into_iter().map(|(vid, s)| (vid, s as f64)).collect())
    }

    /// Cosine similarity of the query to a chunk.
    pub fn similarity(&self, q: &[f32], vid: i64) -> Option<f64> {
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
    pub fn neighbours(&self, vids: &[i64], min_weight: f64) -> Result<Vec<(i64, f64, i64)>> {
        if vids.is_empty() {
            return Ok(Vec::new());
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
}

impl Store {
    pub fn new(data: PathBuf) -> Store {
        Store { data, open: Mutex::new(HashMap::new()) }
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
        let ns = Arc::new(Namespace::open(name, &path)?);
        open.insert(name.to_string(), ns.clone());
        Ok(Some(ns))
    }

    /// Removes a namespace and its folder.
    pub fn drop_namespace(&self, name: &str) -> Result<()> {
        self.registry().remove(name);
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
