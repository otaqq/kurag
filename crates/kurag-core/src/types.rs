//! Requests and answers of the HTTP API, version 1 (the txtai-kura v0.3.5 API).

use serde::{Deserialize, Serialize};

/// A chunk to store.
#[derive(Debug, Clone, Deserialize)]
pub struct Doc {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub source: String,
}

#[derive(Debug, Deserialize)]
pub struct Upsert {
    pub namespace: String,
    pub documents: Vec<Doc>,
}

/// `/delete` and `/kg/delete`.
#[derive(Debug, Deserialize)]
pub struct Delete {
    pub namespace: String,
    #[serde(default)]
    pub prefix: Option<String>,
    /// several at once
    #[serde(default)]
    pub prefixes: Option<Vec<String>>,
}

impl Delete {
    /// `prefix` (unless empty) and then `prefixes`.
    pub fn all_prefixes(&self) -> Vec<String> {
        let mut out: Vec<String> = self.prefix.iter().filter(|p| !p.is_empty()).cloned().collect();
        out.extend(self.prefixes.iter().flatten().cloned());
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Vector,
    Text,
    Hybrid,
}

#[derive(Debug, Deserialize)]
pub struct Search {
    pub query: String,
    #[serde(default)]
    pub namespaces: Option<Vec<String>>,
    #[serde(default)]
    pub exclude: Option<Vec<String>>,
    #[serde(default = "five")]
    pub limit: usize,
    #[serde(default)]
    pub mode: Mode,
    /// graph RAG: add up to this many chunks the similarity graph links to the hits
    #[serde(default)]
    pub graph: usize,
    /// only follow edges at least this similar
    #[serde(default = "min_weight")]
    pub min_weight: f64,
    /// knowledge graph (hybrid only): walk this many steps from the entities named in the query, and fuse the
    /// chunks that mention the entities reached as a third list (0: off)
    #[serde(default)]
    pub kg: usize,
}

fn five() -> usize {
    5
}
fn ten() -> usize {
    10
}
fn one() -> usize {
    1
}
fn min_weight() -> f64 {
    0.75
}
fn weight() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct Entity {
    pub name: String,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Relation {
    pub source: String,
    pub target: String,
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub description: String,
    #[serde(default = "weight")]
    pub weight: f64,
    /// the chunk it was learned from: deleting that chunk removes it
    #[serde(default)]
    pub chunk: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Mention {
    pub entity: String,
    pub chunk: String,
}

#[derive(Debug, Deserialize)]
pub struct KgUpsert {
    pub namespace: String,
    #[serde(default)]
    pub entities: Vec<Entity>,
    #[serde(default)]
    pub relations: Vec<Relation>,
    #[serde(default)]
    pub mentions: Vec<Mention>,
    /// chunks whose whole extraction this is: replaces what was learned from them, marks them done
    #[serde(default)]
    pub done: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct KgTodo {
    pub namespace: String,
    #[serde(default = "ten")]
    pub limit: usize,
}

#[derive(Debug, Deserialize)]
pub struct KgSearch {
    pub query: String,
    #[serde(default)]
    pub namespaces: Option<Vec<String>>,
    #[serde(default)]
    pub exclude: Option<Vec<String>>,
    #[serde(default = "ten")]
    pub limit: usize,
    #[serde(default = "one")]
    pub depth: usize,
}

#[derive(Debug, Deserialize)]
pub struct Rebuild {
    /// all namespaces when left out
    #[serde(default)]
    pub namespace: Option<String>,
}

/// Range checks of the requests (an error message when out of range).
pub trait Check {
    fn check(&self) -> Result<(), String>;
}

fn within(name: &str, v: usize, lo: usize, hi: usize) -> Result<(), String> {
    if (lo..=hi).contains(&v) {
        Ok(())
    } else {
        Err(format!("{name} must be {lo}..{hi}, not {v}"))
    }
}

impl Check for Search {
    fn check(&self) -> Result<(), String> {
        within("limit", self.limit, 1, 1000)?;
        within("graph", self.graph, 0, 100)?;
        within("kg", self.kg, 0, 3)?;
        if self.kg > 0 && self.mode != Mode::Hybrid {
            return Err("kg works with mode hybrid only".into());
        }
        Ok(())
    }
}

impl Check for KgTodo {
    fn check(&self) -> Result<(), String> {
        within("limit", self.limit, 1, 1000)
    }
}

impl Check for KgSearch {
    fn check(&self) -> Result<(), String> {
        within("limit", self.limit, 1, 200)?;
        within("depth", self.depth, 0, 3)
    }
}

/// A search hit.
#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub namespace: String,
    pub id: String,
    pub source: String,
    pub text: String,
    pub score: f64,
    /// hybrid: cosine similarity (also for hits only the text search found)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f64>,
    /// hybrid: text score, when the text search found it
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text_score: Option<f64>,
    /// graph expansion: the hit this chunk is linked to
    #[serde(skip_serializing_if = "Option::is_none")]
    pub via: Option<String>,
}

/// A chunk still to extract (`/kg/todo`).
#[derive(Debug, Clone, Serialize)]
pub struct TodoChunk {
    pub id: String,
    pub source: String,
    pub text: String,
}

/// Sizes of a namespace's knowledge graph.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct KgCounts {
    pub entities: i64,
    pub relations: i64,
    pub mentions: i64,
    pub done: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct KgEntity {
    pub namespace: String,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub description: String,
    /// chunks that mention it (up to 20)
    pub chunks: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KgRelation {
    pub namespace: String,
    pub source: String,
    pub target: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub description: String,
    pub weight: f64,
    /// chunks it was learned from
    pub chunks: Vec<String>,
}
