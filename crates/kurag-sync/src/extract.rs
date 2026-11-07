//! Knowledge graph extraction: ask an LLM for the entities and relations in each chunk the server has not extracted
//! yet (`/kg/todo`), and send them back (`/kg/upsert` with `done`). Ported from txtai-kura-sync 0.4.1 with the same
//! prompt, answer schema, limits and command line.
//!
//! Resumable: a chunk counts as done on the server once its result is sent, so a run stopped by `--minutes`,
//! `--limit` or an error goes on where it left off. A chunk sent again (changed text) is to do again, and its new
//! extraction replaces the old one. Any OpenAI-compatible chat API works (OpenAI, Ollama's /v1, a LiteLLM proxy);
//! with several API bases (one per machine) each gets `--parallel` requests at a time, and a request slot that frees
//! up gets the next chunk at once, so a fast machine does more. The list of bases can be read from a file again
//! before every batch (`--api-bases-file`), so a caller can take a machine out of a running extraction or add one.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};

pub const PROMPT: &str = r#"You extract a knowledge graph from one passage of a document. Answer with JSON only, in this form:
{"entities": [{"name": "...", "type": "...", "description": "..."}],
 "relations": [{"source": "...", "target": "...", "type": "...", "description": "..."}]}
Rules:
- entities: the specific things the passage is about: software, tools, concepts, people, organizations, places,
  machines, works. Leave out generic words (system, data, method, example).
- name: the usual name of the thing as the passage writes it, in the passage's language; always the same name for the
  same thing; no articles.
- type: one lower-case English word: software, concept, person, organization, place, machine, document, event, other.
- description: one short sentence about the entity, from the passage, in the passage's language.
- relations: only between names in your entities list, only what the passage states or clearly implies; type: a short
  lower-case English verb phrase (uses, runs on, part of, depends on, written by, ...).
- At most 15 entities and 20 relations. If the passage names nothing specific: {"entities": [], "relations": []}."#;

const TYPES: [&str; 9] = [
    "software",
    "concept",
    "person",
    "organization",
    "place",
    "machine",
    "document",
    "event",
    "other",
];
/// characters of a chunk sent to the model
pub const MAX_TEXT: usize = 6000;
/// An answer has at most 15 entities and 20 relations (well under 2,000 tokens). The cap stops a model that repeats
/// itself: without it, a server that shifts its context (Ollama) generates until the client times out.
pub const MAX_TOKENS: u32 = 4096;
pub const MAX_ENTITIES: usize = 15;
pub const MAX_RELATIONS: usize = 20;
/// seconds for one answer; chunks took up to about 2 minutes on a 12B model
pub const TIMEOUT: u64 = 300;
/// seconds to wait before the 2nd and 3rd try when the model's machine refuses or fails (restarting)
pub const RETRIES: [u64; 2] = [15, 60];

/// How the answer is held to JSON: `schema` (response_format json_schema with item and length limits: a model that
/// starts repeating itself has to close the JSON), `object` (json_object) or `none`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum JsonMode {
    Schema,
    Object,
    None,
}

impl JsonMode {
    pub fn parse(s: &str) -> Option<JsonMode> {
        match s {
            "schema" => Some(JsonMode::Schema),
            "object" => Some(JsonMode::Object),
            "none" => Some(JsonMode::None),
            _ => None,
        }
    }
}

pub fn schema() -> Value {
    let text = |n: usize| json!({"type": "string", "maxLength": n});
    json!({
        "type": "object", "required": ["entities", "relations"], "additionalProperties": false,
        "properties": {
            "entities": {"type": "array", "maxItems": MAX_ENTITIES, "items": {
                "type": "object", "required": ["name", "type", "description"], "additionalProperties": false,
                "properties": {"name": text(100), "type": text(20), "description": text(200)}}},
            "relations": {"type": "array", "maxItems": MAX_RELATIONS, "items": {
                "type": "object", "required": ["source", "target", "type", "description"], "additionalProperties": false,
                "properties": {"source": text(100), "target": text(100), "type": text(40), "description": text(200)}}},
        },
    })
}

pub fn log(msg: &str) {
    println!("{} {msg}", chrono::Local::now().format("%F %T"));
}

/// Why a call to the model failed: `Unreachable` (refused connection or 5xx: the machine is restarting or busy) is
/// tried again; a timeout is not (the model was working on this chunk and would only work on it again), nor a
/// refusal (4xx) or a bad answer.
#[derive(Debug)]
pub enum CallError {
    Unreachable(String),
    Timeout(String),
    Failed(String),
}

impl std::fmt::Display for CallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CallError::Unreachable(m) => write!(f, "unreachable: {m}"),
            CallError::Timeout(m) => write!(f, "timed out: {m}"),
            CallError::Failed(m) => write!(f, "{m}"),
        }
    }
}

/// A chat model.
pub trait Model: Send + Sync {
    fn chat(&self, system: &str, user: &str) -> Result<String, CallError>;
    /// Where it runs, for the log.
    fn place(&self) -> String;
}

/// An OpenAI-compatible chat completions API.
pub struct Llm {
    url: String,
    model: String,
    api_key: String,
    json_mode: JsonMode,
    http: reqwest::blocking::Client,
}

impl Llm {
    pub fn new(api_base: &str, model: &str, api_key: &str, json_mode: JsonMode) -> Llm {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(TIMEOUT))
            .build()
            .expect("http client");
        Llm {
            url: format!("{}/chat/completions", api_base.trim_end_matches('/')),
            model: model.to_string(),
            api_key: api_key.to_string(),
            json_mode,
            http,
        }
    }

    pub fn body(&self, system: &str, user: &str) -> Value {
        let mut body = json!({"model": self.model, "temperature": 0, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]});
        match self.json_mode {
            JsonMode::Schema => {
                body["response_format"] =
                    json!({"type": "json_schema", "json_schema": {"name": "knowledge_graph", "schema": schema()}})
            }
            JsonMode::Object => body["response_format"] = json!({"type": "json_object"}),
            JsonMode::None => {}
        }
        body
    }
}

impl Model for Llm {
    fn chat(&self, system: &str, user: &str) -> Result<String, CallError> {
        let mut req = self.http.post(&self.url).json(&self.body(system, user));
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }
        let classify = |e: reqwest::Error| {
            if e.is_timeout() {
                CallError::Timeout(e.to_string())
            } else if e.is_connect() || e.is_request() {
                CallError::Unreachable(e.to_string())
            } else {
                CallError::Failed(e.to_string())
            }
        };
        let r = req.send().map_err(classify)?;
        let status = r.status();
        if status.is_server_error() {
            return Err(CallError::Unreachable(format!("HTTP {status}")));
        }
        if !status.is_success() {
            return Err(CallError::Failed(format!("HTTP {status}")));
        }
        let v: Value = r.json().map_err(classify)?;
        let content = v["choices"][0]["message"]["content"].clone();
        match content {
            Value::String(s) => Ok(s),
            Value::Null if v["choices"][0]["message"].is_object() => Ok(String::new()),
            _ => Err(CallError::Failed("no choices[0].message.content in the answer".into())),
        }
    }

    fn place(&self) -> String {
        reqwest::Url::parse(&self.url)
            .ok()
            .and_then(|u| {
                u.host_str()
                    .map(|h| u.port().map_or(h.to_string(), |p| format!("{h}:{p}")))
            })
            .unwrap_or_else(|| self.url.clone())
    }
}

/// A JSON value as text, as Python's str() would give it for the values a model sends.
fn text_of(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(false)) => String::new(),
        Some(other) => other.to_string(),
    }
}

fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn cut(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// The model's answer as `/kg/upsert` lists for one chunk: {entities, relations, mentions}. Fails when it is not the
/// JSON object asked for.
pub fn parse(answer: &str, chunk_id: &str) -> Result<Value, String> {
    let mut text = answer.trim();
    if let Some(rest) = text.strip_prefix("```") {
        text = rest.strip_prefix("json").unwrap_or(rest).trim_start();
    }
    if let Some(rest) = text.strip_suffix("```") {
        text = rest.trim_end();
    }
    let data: Value = serde_json::from_str(text).map_err(|e| format!("not JSON: {e}"))?;
    let Value::Object(data) = data else {
        return Err("not a JSON object".into());
    };
    let empty = Vec::new();
    let list = |key: &str| match data.get(key) {
        Some(Value::Array(a)) => a,
        _ => &empty,
    };
    let mut entities = Vec::new();
    let mut names: HashMap<String, String> = HashMap::new();
    for e in list("entities") {
        let Value::Object(e) = e else { continue };
        let name = cut(&squash(&text_of(e.get("name"))), 200);
        if name.is_empty() || names.contains_key(&name.to_lowercase()) {
            continue;
        }
        let kind = text_of(e.get("type")).trim().to_lowercase();
        let kind = if TYPES.contains(&kind.as_str()) {
            kind
        } else {
            "other".to_string()
        };
        names.insert(name.to_lowercase(), name.clone());
        entities.push(
            json!({"name": name, "type": kind, "description": cut(&squash(&text_of(e.get("description"))), 500)}),
        );
    }
    let mut relations = Vec::new();
    let mut seen: HashSet<(String, String, String)> = HashSet::new();
    for r in list("relations") {
        let Value::Object(r) = r else { continue };
        let find = |key: &str| names.get(&squash(&text_of(r.get(key))).to_lowercase()).cloned();
        let (Some(s), Some(t)) = (find("source"), find("target")) else {
            continue;
        };
        let mut kind = squash(&text_of(r.get("type")).to_lowercase());
        if kind.is_empty() {
            kind = "related to".into();
        }
        let kind = cut(&kind, 80);
        // a model that repeats itself sends the same one again
        if s != t && seen.insert((s.clone(), kind.clone(), t.clone())) {
            relations.push(json!({"source": s, "target": t, "type": kind,
                "description": cut(&squash(&text_of(r.get("description"))), 500), "chunk": chunk_id}));
        }
    }
    let mentions: Vec<Value> = entities
        .iter()
        .map(|e| json!({"entity": e["name"], "chunk": chunk_id}))
        .collect();
    Ok(json!({"entities": entities, "relations": relations, "mentions": mentions}))
}

#[derive(Clone, Debug, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub text: String,
}

/// One chunk: the model's answer, parsed, and the seconds it took. `Unreachable` is tried again after each of
/// `retries` seconds.
pub fn extract_one(model: &dyn Model, chunk: &Chunk, retries: &[u64]) -> Result<(Value, f64), String> {
    let text: String = chunk.text.chars().take(MAX_TEXT).collect();
    let mut waits = retries.iter();
    loop {
        let started = Instant::now();
        match model.chat(PROMPT, &text) {
            Ok(answer) => return parse(&answer, &chunk.id).map(|r| (r, started.elapsed().as_secs_f64())),
            Err(CallError::Unreachable(e)) => match waits.next() {
                Some(&wait) => {
                    log(&format!("{}: model unreachable, trying again in {wait} s", chunk.id));
                    std::thread::sleep(Duration::from_secs(wait));
                }
                None => return Err(format!("unreachable: {e}")),
            },
            Err(e) => return Err(e.to_string()),
        }
    }
}

/// The server's side of extraction.
pub trait Server: Sync {
    /// (chunks not extracted yet, how many are left in all)
    fn todo(&self, namespace: &str, limit: usize) -> Result<(Vec<Chunk>, i64), String>;
    fn upsert(&self, namespace: &str, result: &Value, done: &str) -> Result<(), String>;
}

/// The kurag (or txtai-kura) API.
pub struct Rag {
    url: String,
    api_key: String,
    http: reqwest::blocking::Client,
}

impl Rag {
    pub fn new(url: &str, api_key: &str) -> Rag {
        let http = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("http client");
        Rag {
            url: url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            http,
        }
    }

    fn call(&self, path: &str, body: &Value) -> Result<Value, String> {
        let url = format!("{}{path}", self.url);
        let r = self
            .http
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .json(body)
            .send()
            .map_err(|e| format!("{url}: {e}"))?;
        let status = r.status();
        let text = r.text().map_err(|e| format!("{url}: {e}"))?;
        if !status.is_success() {
            return Err(format!("{url}: HTTP {status}: {}", cut(&text, 300)));
        }
        serde_json::from_str(&text).map_err(|e| format!("{url}: {e}"))
    }
}

impl Server for Rag {
    fn todo(&self, namespace: &str, limit: usize) -> Result<(Vec<Chunk>, i64), String> {
        let v = self.call("/kg/todo", &json!({"namespace": namespace, "limit": limit}))?;
        let chunks: Vec<Chunk> = serde_json::from_value(v["chunks"].clone()).map_err(|e| format!("/kg/todo: {e}"))?;
        Ok((chunks, v["remaining"].as_i64().unwrap_or(0)))
    }

    fn upsert(&self, namespace: &str, result: &Value, done: &str) -> Result<(), String> {
        let mut body = result.clone();
        body["namespace"] = json!(namespace);
        body["done"] = json!([done]);
        self.call("/kg/upsert", &body).map(|_| ())
    }
}

/// The API bases (one per machine; listed twice, twice the requests), fixed or read from a file before every batch.
pub enum Bases {
    Fixed(Vec<String>),
    /// the file, and the last list read from it (kept when the file cannot be read: being replaced)
    File(PathBuf, Vec<String>),
}

/// The bases in the file: one per line, "URL N" counts it N times; blank lines and # comments left out.
pub fn read_api_bases(path: &Path) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("");
        let mut words = line.split_whitespace();
        let Some(base) = words.next() else { continue };
        let n: usize = match words.next() {
            Some(n) => n.parse().map_err(|_| format!("{}: not a count: {n}", path.display()))?,
            None => 1,
        };
        out.extend(std::iter::repeat_n(base.to_string(), n));
    }
    Ok(out)
}

impl Bases {
    fn current(&mut self) -> Vec<String> {
        match self {
            Bases::Fixed(b) => b.clone(),
            Bases::File(path, last) => {
                match read_api_bases(path) {
                    Ok(b) => *last = b,
                    Err(e) => log(&format!("api bases file not read, the last list is kept: {e}")),
                }
                last.clone()
            }
        }
    }
}

pub struct Settings {
    pub namespace: String,
    /// requests at a time per API base
    pub parallel: usize,
    pub minutes: Option<f64>,
    pub limit: Option<usize>,
    pub dry_run: bool,
    pub retries: Vec<u64>,
}

type Done = (Chunk, String, Result<(Value, f64), String>);

/// Extracts until nothing is left, the time is up or `limit` chunks are done: 0, or 1 when chunks failed. `make`
/// turns an API base into a model (one per base for the whole run). An API no longer listed gets no new chunk (its
/// running requests finish); an API listed anew gets chunks at once. The run ends when no API is left and nothing is
/// running. A failure of the server ends it with an error.
pub fn run(
    server: &dyn Server,
    bases: &mut Bases,
    make: &dyn Fn(&str) -> Arc<dyn Model>,
    s: &Settings,
) -> Result<i32, String> {
    let several = matches!(bases, Bases::File(..)) || {
        let b = bases.current();
        b.iter().collect::<HashSet<_>>().len() > 1
    };
    let mut models: HashMap<String, Arc<dyn Model>> = HashMap::new();
    let capacity = |list: &[String]| {
        let mut cap: HashMap<String, usize> = HashMap::new();
        for b in list {
            *cap.entry(b.clone()).or_default() += s.parallel;
        }
        cap
    };
    let mut cap = capacity(&bases.current());
    let mut limit = s.limit;
    if s.dry_run && limit.is_none() {
        // nothing is marked done: without a limit, one round of chunks only
        limit = Some(cap.values().sum::<usize>() * 2);
    }
    let deadline = s.minutes.map(|m| Instant::now() + Duration::from_secs_f64(m * 60.0));
    let mut failed: HashSet<String> = HashSet::new();
    let mut seen: HashSet<String> = HashSet::new(); // sent to a model in this run (a dry run marks nothing done)
    let mut busy: HashMap<String, usize> = HashMap::new();
    let mut pending: HashSet<String> = HashSet::new();
    let (mut done, mut entities, mut relations) = (0usize, 0usize, 0usize);
    let mut seconds: Vec<f64> = Vec::new();
    let started = Instant::now();
    let mut exhausted = false;
    let (tx, rx) = mpsc::channel::<Done>();
    loop {
        let new_cap = capacity(&bases.current());
        for gone in cap.keys().filter(|b| !new_cap.contains_key(*b)) {
            log(&format!(
                "api {} removed: no new chunks, {} running request(s) finish",
                place(&models, gone),
                busy.get(gone).copied().unwrap_or(0)
            ));
        }
        for added in new_cap.keys().filter(|b| !cap.contains_key(*b)) {
            log(&format!("api {} added", place(&models, added)));
        }
        cap = new_cap;
        let mut room: usize = cap
            .iter()
            .map(|(b, n)| n.saturating_sub(busy.get(b).copied().unwrap_or(0)))
            .sum();
        if let Some(l) = limit {
            room = room.min(l.saturating_sub(seen.len()));
        }
        if room > 0 && !exhausted && deadline.is_none_or(|d| Instant::now() < d) {
            let mut skip: HashSet<String> = pending.iter().chain(failed.iter()).cloned().collect();
            if s.dry_run {
                skip.extend(seen.iter().cloned());
            }
            let (chunks, _) = server.todo(&s.namespace, room + skip.len())?;
            let asked = room + skip.len();
            let got = chunks.len();
            let fresh: Vec<Chunk> = chunks
                .into_iter()
                .filter(|c| !skip.contains(&c.id))
                .take(room)
                .collect();
            if fresh.is_empty() && pending.is_empty() {
                break;
            }
            exhausted = fresh.is_empty() && got < asked;
            for c in fresh {
                let base = cap
                    .iter()
                    .filter(|(b, n)| busy.get(*b).copied().unwrap_or(0) < **n)
                    .min_by(|a, b| {
                        let load = |(k, n): (&String, &usize)| busy.get(k).copied().unwrap_or(0) as f64 / *n as f64;
                        load(*a).total_cmp(&load(*b)).then(a.0.cmp(b.0))
                    })
                    .map(|(b, _)| b.clone())
                    .expect("room means a free slot");
                *busy.entry(base.clone()).or_default() += 1;
                seen.insert(c.id.clone());
                pending.insert(c.id.clone());
                let model = models.entry(base.clone()).or_insert_with(|| make(&base)).clone();
                let (tx, retries) = (tx.clone(), s.retries.clone());
                std::thread::spawn(move || {
                    let r = extract_one(model.as_ref(), &c, &retries);
                    let _ = tx.send((c, base, r));
                });
            }
        }
        if pending.is_empty() {
            break;
        }
        let first = rx.recv().map_err(|e| e.to_string())?;
        let mut finished = vec![first];
        finished.extend(rx.try_iter());
        for (c, base, r) in finished {
            pending.remove(&c.id);
            if let Some(n) = busy.get_mut(&base) {
                *n = n.saturating_sub(1);
            }
            let (result, took) = match r {
                Ok(x) => x,
                Err(e) => {
                    log(&format!("{}: failed ({})", c.id, cut(&e, 200)));
                    failed.insert(c.id);
                    continue;
                }
            };
            seconds.push(took);
            done += 1;
            let (ne, nr) = (
                result["entities"].as_array().map_or(0, Vec::len),
                result["relations"].as_array().map_or(0, Vec::len),
            );
            entities += ne;
            relations += nr;
            let at = if several {
                format!(" ({})", place(&models, &base))
            } else {
                String::new()
            };
            if s.dry_run {
                let mut line = json!({"chunk": c.id, "seconds": (took * 10.0).round() / 10.0});
                if several {
                    line["api"] = json!(place(&models, &base));
                }
                for k in ["entities", "relations", "mentions"] {
                    line[k] = result[k].clone();
                }
                println!("{line}");
            } else {
                server.upsert(&s.namespace, &result, &c.id)?;
                log(&format!("{}: {ne} entities, {nr} relations, {took:.1} s{at}", c.id));
            }
        }
    }
    let (_, left) = server.todo(&s.namespace, 1)?;
    let avg = if seconds.is_empty() {
        0.0
    } else {
        seconds.iter().sum::<f64>() / seconds.len() as f64
    };
    log(&format!(
        "done: {done} chunks, {entities} entities, {relations} relations, {} failed, {avg:.1} s per chunk, {:.0} s in all, {left} left{}",
        failed.len(),
        started.elapsed().as_secs_f64(),
        if s.dry_run { " (dry run)" } else { "" }
    ));
    Ok(if failed.is_empty() { 0 } else { 1 })
}

fn place(models: &HashMap<String, Arc<dyn Model>>, base: &str) -> String {
    models.get(base).map_or_else(|| base.to_string(), |m| m.place())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn parse_answers() {
        let answer = r#"```json
{"entities": [{"name": " Kubernetes ", "type": "Software", "description": "an  orchestrator"},
  {"name": "kubernetes", "type": "software"}, {"name": "", "type": "x"}, {"name": "Pod", "type": "thing"}],
 "relations": [{"source": "kubernetes", "target": "Pod", "type": "Runs"}, {"source": "Kubernetes", "target": "Pod", "type": "runs"},
  {"source": "Pod", "target": "Pod", "type": "is"}, {"source": "Nobody", "target": "Pod"}]}
```"#;
        let r = parse(answer, "d:a.md#S#0").unwrap();
        let names: Vec<&str> = r["entities"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, ["Kubernetes", "Pod"]); // duplicates and empty names go
        assert_eq!(r["entities"][0]["type"], "software");
        assert_eq!(r["entities"][0]["description"], "an orchestrator");
        assert_eq!(r["entities"][1]["type"], "other");
        assert_eq!(
            r["relations"],
            json!([{"source": "Kubernetes", "target": "Pod", "type": "runs", "description": "", "chunk": "d:a.md#S#0"}])
        );
        assert_eq!(
            r["mentions"],
            json!([{"entity": "Kubernetes", "chunk": "d:a.md#S#0"}, {"entity": "Pod", "chunk": "d:a.md#S#0"}])
        );
        assert!(parse("[1]", "x").is_err());
        assert!(parse("no json", "x").is_err());
        let r = parse(
            r#"{"entities": [{"name": "A"}, {"name": "B"}], "relations": [{"source": "A", "target": "B"}]}"#,
            "c",
        )
        .unwrap();
        assert_eq!(r["relations"][0]["type"], "related to");
    }

    /// The prompt, schema and limits of txtai-kura-sync 0.4.1 (testdata, dumped from its extract.py): a port that
    /// asks the same questions, so an extraction moved over finds the same entities.
    #[test]
    fn same_as_txtai_kura_sync() {
        let py: Value = serde_json::from_str(include_str!("../testdata/txtai-kura-sync-0.4.1-extract.json")).unwrap();
        assert_eq!(py["prompt"], PROMPT);
        assert_eq!(py["schema"], schema());
        // and in the same key order: the model writes its answer in the schema's order (name, type, description),
        // so sorted keys gave other answers (serde_json's preserve_order)
        let order = |v: &Value| serde_json::to_string(&v["properties"]["entities"]["items"]["properties"]).unwrap();
        assert_eq!(
            order(&schema()),
            r#"{"name":{"type":"string","maxLength":100},"type":{"type":"string","maxLength":20},"description":{"type":"string","maxLength":200}}"#
        );
        assert_eq!(
            (
                py["max_tokens"].as_u64(),
                py["max_text"].as_u64(),
                py["timeout"].as_u64()
            ),
            (Some(MAX_TOKENS as u64), Some(MAX_TEXT as u64), Some(TIMEOUT))
        );
        assert_eq!(py["retries"], json!(RETRIES));
        let mut types = TYPES.to_vec();
        types.sort();
        assert_eq!(py["types"], json!(types));
    }

    #[test]
    fn request_body() {
        let llm = Llm::new("http://h:11434/v1/", "m", "", JsonMode::Schema);
        assert_eq!(llm.url, "http://h:11434/v1/chat/completions");
        assert_eq!(llm.place(), "h:11434");
        let b = llm.body("s", "u");
        assert_eq!(
            (b["max_tokens"].as_u64(), b["temperature"].as_i64()),
            (Some(MAX_TOKENS as u64), Some(0))
        );
        assert_eq!(b["response_format"]["type"], "json_schema");
        assert_eq!(
            b["response_format"]["json_schema"]["schema"]["properties"]["relations"]["maxItems"],
            MAX_RELATIONS
        );
        assert_eq!(
            Llm::new("http://h/v1", "m", "", JsonMode::Object).body("s", "u")["response_format"],
            json!({"type": "json_object"})
        );
        assert!(
            Llm::new("http://h/v1", "m", "", JsonMode::None)
                .body("s", "u")
                .get("response_format")
                .is_none()
        );
        assert_eq!(JsonMode::parse("bad"), None);
    }

    /// A model answering from a script: per chunk text, an error kind, else two entities and a relation.
    struct Fake {
        name: String,
        asked: Mutex<Vec<String>>,
        fail: HashMap<String, &'static str>,
        delay: Duration,
        unreachable_times: Mutex<usize>,
    }

    fn fake(name: &str) -> Arc<Fake> {
        Arc::new(Fake {
            name: name.into(),
            asked: Mutex::new(Vec::new()),
            fail: HashMap::new(),
            delay: Duration::ZERO,
            unreachable_times: Mutex::new(0),
        })
    }

    impl Model for Fake {
        fn chat(&self, _: &str, user: &str) -> Result<String, CallError> {
            self.asked.lock().unwrap().push(user.to_string());
            std::thread::sleep(self.delay);
            {
                let mut n = self.unreachable_times.lock().unwrap();
                if *n > 0 {
                    *n -= 1;
                    return Err(CallError::Unreachable("refused".into()));
                }
            }
            match self.fail.get(user).copied() {
                Some("timeout") => Err(CallError::Timeout("slow".into())),
                Some("bad") => Ok("not json".into()),
                Some(_) => Err(CallError::Failed("HTTP 400".into())),
                None => Ok(r#"{"entities": [{"name": "A", "type": "concept"}, {"name": "B"}], "relations": [{"source": "A", "target": "B", "type": "uses"}]}"#.into()),
            }
        }
        fn place(&self) -> String {
            self.name.clone()
        }
    }

    /// A server with n chunks "text i".
    struct FakeServer {
        chunks: Vec<Chunk>,
        done: Mutex<HashSet<String>>,
        upserts: Mutex<Vec<Value>>,
    }

    fn server(n: usize) -> FakeServer {
        FakeServer {
            chunks: (0..n)
                .map(|i| Chunk {
                    id: format!("d:a.md#S#{i}"),
                    text: format!("text {i}"),
                })
                .collect(),
            done: Mutex::new(HashSet::new()),
            upserts: Mutex::new(Vec::new()),
        }
    }

    impl Server for FakeServer {
        fn todo(&self, _: &str, limit: usize) -> Result<(Vec<Chunk>, i64), String> {
            let done = self.done.lock().unwrap();
            let left: Vec<Chunk> = self.chunks.iter().filter(|c| !done.contains(&c.id)).cloned().collect();
            Ok((left.iter().take(limit).cloned().collect(), left.len() as i64))
        }
        fn upsert(&self, _: &str, result: &Value, done: &str) -> Result<(), String> {
            self.upserts.lock().unwrap().push(result.clone());
            self.done.lock().unwrap().insert(done.to_string());
            Ok(())
        }
    }

    fn settings(parallel: usize) -> Settings {
        Settings {
            namespace: "it".into(),
            parallel,
            minutes: None,
            limit: None,
            dry_run: false,
            retries: vec![0, 0],
        }
    }

    fn by_name(models: Vec<Arc<Fake>>) -> impl Fn(&str) -> Arc<dyn Model> {
        move |b: &str| {
            models
                .iter()
                .find(|m| m.name == b)
                .cloned()
                .map(|m| m as Arc<dyn Model>)
                .unwrap()
        }
    }

    #[test]
    fn retries_only_unreachable() {
        let m = fake("a");
        *m.unreachable_times.lock().unwrap() = 2;
        let c = Chunk {
            id: "c".into(),
            text: "t".into(),
        };
        let (r, _) = extract_one(m.as_ref(), &c, &[0, 0]).unwrap();
        assert_eq!(
            (m.asked.lock().unwrap().len(), r["entities"].as_array().unwrap().len()),
            (3, 2)
        );
        let m = fake("a");
        *m.unreachable_times.lock().unwrap() = 5;
        assert!(extract_one(m.as_ref(), &c, &[0, 0]).is_err());
        assert_eq!(m.asked.lock().unwrap().len(), 3);
        let mut f = Fake {
            name: "a".into(),
            asked: Mutex::new(Vec::new()),
            fail: HashMap::new(),
            delay: Duration::ZERO,
            unreachable_times: Mutex::new(0),
        };
        f.fail.insert("t".into(), "timeout");
        assert!(extract_one(&f, &c, &[0, 0]).is_err());
        assert_eq!(f.asked.lock().unwrap().len(), 1); // a timeout is not tried again
    }

    #[test]
    fn run_failures_limit_and_dry_run() {
        let srv = server(5);
        let mut f = Fake {
            name: "a".into(),
            asked: Mutex::new(Vec::new()),
            fail: HashMap::new(),
            delay: Duration::ZERO,
            unreachable_times: Mutex::new(0),
        };
        f.fail.insert("text 3".into(), "bad");
        let f = Arc::new(f);
        let code = run(
            &srv,
            &mut Bases::Fixed(vec!["a".into()]),
            &by_name(vec![f.clone()]),
            &settings(2),
        )
        .unwrap();
        assert_eq!(code, 1); // one chunk failed
        assert_eq!(srv.done.lock().unwrap().len(), 4);
        assert_eq!(f.asked.lock().unwrap().iter().filter(|t| *t == "text 3").count(), 1); // not asked again in the run
        assert_eq!(srv.upserts.lock().unwrap()[0]["relations"].as_array().unwrap().len(), 1);

        let srv = server(5);
        let a = fake("a");
        let s = Settings {
            limit: Some(3),
            ..settings(2)
        };
        assert_eq!(
            run(&srv, &mut Bases::Fixed(vec!["a".into()]), &by_name(vec![a.clone()]), &s).unwrap(),
            0
        );
        assert_eq!(srv.done.lock().unwrap().len(), 3);
        assert_eq!(
            run(
                &srv,
                &mut Bases::Fixed(vec!["a".into()]),
                &by_name(vec![a]),
                &settings(2)
            )
            .unwrap(),
            0
        );
        assert_eq!(srv.done.lock().unwrap().len(), 5);

        let srv = server(5);
        let a = fake("a");
        let s = Settings {
            limit: Some(3),
            dry_run: true,
            ..settings(2)
        };
        assert_eq!(
            run(&srv, &mut Bases::Fixed(vec!["a".into()]), &by_name(vec![a.clone()]), &s).unwrap(),
            0
        );
        assert_eq!(
            (srv.upserts.lock().unwrap().len(), a.asked.lock().unwrap().len()),
            (0, 3)
        );
    }

    #[test]
    fn several_apis_share_the_work() {
        let srv = server(12);
        let fast = fake("fast");
        let slow = Arc::new(Fake {
            delay: Duration::from_millis(60),
            ..Arc::try_unwrap(fake("slow")).ok().unwrap()
        });
        let bases = vec!["fast".into(), "slow".into()];
        assert_eq!(
            run(
                &srv,
                &mut Bases::Fixed(bases),
                &by_name(vec![fast.clone(), slow.clone()]),
                &settings(1)
            )
            .unwrap(),
            0
        );
        let (f, s) = (fast.asked.lock().unwrap().len(), slow.asked.lock().unwrap().len());
        assert_eq!((srv.done.lock().unwrap().len(), f + s), (12, 12)); // every chunk once
        assert!(f > s);

        // dry run, 2 APIs x 2 slots: two rounds of chunks, none twice
        let srv = server(20);
        let (a, b) = (fake("a"), fake("b"));
        let s = Settings {
            dry_run: true,
            ..settings(2)
        };
        assert_eq!(
            run(
                &srv,
                &mut Bases::Fixed(vec!["a".into(), "b".into()]),
                &by_name(vec![a.clone(), b.clone()]),
                &s
            )
            .unwrap(),
            0
        );
        let all: Vec<String> = a
            .asked
            .lock()
            .unwrap()
            .iter()
            .chain(b.asked.lock().unwrap().iter())
            .cloned()
            .collect();
        assert_eq!((all.len(), all.iter().collect::<HashSet<_>>().len()), (8, 8));
    }

    #[test]
    fn bases_file_is_read_again() {
        let dir = std::env::temp_dir().join(format!("kurag-sync-bases-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bases.txt");
        std::fs::write(&path, "# machines\nhttp://a/v1 2\n\nhttp://b/v1  # second\n").unwrap();
        assert_eq!(
            read_api_bases(&path).unwrap(),
            ["http://a/v1", "http://a/v1", "http://b/v1"]
        );
        let mut bases = Bases::File(path.clone(), vec![]);
        assert_eq!(bases.current().len(), 3);
        std::fs::write(&path, "http://b/v1\n").unwrap();
        assert_eq!(bases.current(), ["http://b/v1"]);
        std::fs::remove_file(&path).unwrap();
        assert_eq!(bases.current(), ["http://b/v1"]); // file being replaced: the last list is kept

        // a machine taken out while running: its running request finishes, no new one comes
        let srv = server(12);
        let a = Arc::new(Fake {
            delay: Duration::from_millis(20),
            ..Arc::try_unwrap(fake("a")).ok().unwrap()
        });
        let b = Arc::new(Fake {
            delay: Duration::from_millis(20),
            ..Arc::try_unwrap(fake("b")).ok().unwrap()
        });
        std::fs::write(&path, "a\nb\n").unwrap();
        let (b2, p2) = (b.clone(), path.clone());
        let watcher = std::thread::spawn(move || {
            // once b has its first chunk, take it out
            while b2.asked.lock().unwrap().is_empty() {
                std::thread::sleep(Duration::from_millis(1));
            }
            std::fs::write(&p2, "a\n").unwrap();
        });
        let code = run(
            &srv,
            &mut Bases::File(path.clone(), vec![]),
            &by_name(vec![a.clone(), b.clone()]),
            &settings(1),
        )
        .unwrap();
        watcher.join().unwrap();
        assert_eq!((code, srv.done.lock().unwrap().len()), (0, 12));
        let nb = b.asked.lock().unwrap().len();
        assert!((1..=2).contains(&nb), "b was asked {nb} times after it was taken out");
        assert_eq!(a.asked.lock().unwrap().len() + nb, 12);

        // every API gone: the run ends
        let srv = server(12);
        std::fs::write(&path, "").unwrap();
        assert_eq!(
            run(
                &srv,
                &mut Bases::File(path.clone(), vec![]),
                &by_name(vec![]),
                &settings(1)
            )
            .unwrap(),
            0
        );
        assert_eq!(srv.done.lock().unwrap().len(), 0);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
