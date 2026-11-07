//! Keeping a kurag server in step with the documents it indexes (`kurag-sync run`), ported from txtai-kura-sync.
//!
//! Reads the sources of the sources file (git repositories, folders, shelves of books), splits each file into parts
//! (split.rs), compares each part with the last run (sha256) and sends only what changed: a changed part's chunks are
//! deleted and sent again, a removed part's chunks are deleted. A file whose size and time are those of the last run
//! is not even opened (files.json). The state files are txtai-kura-sync's, so a setup moves over without sending
//! anything again:
//!   <data>/state/state.json  part key -> {sha, namespace, chunks}
//!   <data>/state/files.json  "<source>:<file>" -> [size, mtime_ns, namespace] when it was last sent
//!   <data>/state/runs.json   source -> start time of its last run without errors (for --due)
//!   <data>/state/lock        one run at a time
//!   <data>/cache/git/<name>  git work trees; <data>/cache/text/ PDF and EPUB text (read.rs)
//!
//! Source types: git (url, branch, token_env, token_auth), dir (path: one folder or a list), shelf ([source.map]:
//! namespace -> folder or folders; sub is appended to the namespace) and shelves (every folder directly under path
//! whose name is a namespace name is one namespace; prefix, only). Inside a folder of a shelf source, a `.rag.toml`
//! (`namespace` or `shelf` = ..., `[[file]]` path / namespace) sets the namespace below it; the nearest setting wins.
//! `walls` (top level) lists first namespace segments kept apart: a source mapped under one may only send there, and
//! a source mapped under none may send to none of them, whatever a `.rag.toml` says.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};
use toml::{Table, Value as Toml};

use crate::extract::log;
use crate::pystr::fnmatch;
use crate::read::{read_text, sha256_hex};
use crate::split::{Words, chunks, parts_of, place};

/// chunks per request
const BATCH: usize = 200;
const FOLDER_CONFIG: &str = ".rag.toml";
const TYPES: [&str; 4] = ["git", "dir", "shelf", "shelves"];

pub struct Paths {
    pub state: PathBuf,
    pub cache: PathBuf,
}

impl Paths {
    pub fn new(data: &Path) -> Paths {
        Paths {
            state: data.join("state"),
            cache: data.join("cache"),
        }
    }
    fn state_file(&self) -> PathBuf {
        self.state.join("state.json")
    }
    fn files_file(&self) -> PathBuf {
        self.state.join("files.json")
    }
    fn runs_file(&self) -> PathBuf {
        self.state.join("runs.json")
    }
    pub fn lock_file(&self) -> PathBuf {
        self.state.join("lock")
    }
}

/// The sources file, loaded (crate::config::load), with what run needs from it.
pub struct Sources {
    pub cfg: Table,
    /// sources with enabled = false: left out of everything, what they sent stays
    pub disabled: Vec<String>,
    pub words: Words,
    pub walls: Vec<String>,
}

impl Sources {
    pub fn new(mut cfg: Table) -> Result<Sources, String> {
        let list = cfg.get("source").and_then(Toml::as_array).cloned().unwrap_or_default();
        let mut keep = Vec::new();
        let mut disabled = Vec::new();
        for s in list {
            let name = s.get("name").and_then(Toml::as_str).unwrap_or("").to_string();
            match s.get("enabled") {
                None | Some(Toml::Boolean(true)) => keep.push(s),
                Some(Toml::Boolean(false)) => disabled.push(name),
                Some(_) => return Err(format!("source {name}: enabled must be true or false")),
            }
        }
        cfg.insert("source".into(), Toml::Array(keep));
        let text = crate::config::table(&cfg, "text");
        let mut words = Words::default();
        if let Some(v) = crate::config::string(text, "intro") {
            words.intro = v;
        }
        if let Some(v) = crate::config::string(text, "source") {
            words.source = v;
        }
        let walls = cfg
            .get("walls")
            .and_then(Toml::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        Ok(Sources {
            cfg,
            disabled,
            words,
            walls,
        })
    }

    pub fn list(&self) -> Vec<&Table> {
        self.cfg
            .get("source")
            .and_then(Toml::as_array)
            .map(|a| a.iter().filter_map(Toml::as_table).collect())
            .unwrap_or_default()
    }
}

fn s<'a>(t: &'a Table, k: &str) -> Option<&'a str> {
    t.get(k).and_then(Toml::as_str)
}

fn strings(t: &Table, k: &str, default: &[&str]) -> Vec<String> {
    match t.get(k) {
        Some(Toml::String(x)) => vec![x.clone()],
        Some(Toml::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => default.iter().map(|x| x.to_string()).collect(),
    }
}

fn is_namespace(n: &str) -> bool {
    // ^[a-z][a-z0-9-]*(\.[a-z0-9][a-z0-9-]*)*$
    let mut first = true;
    for seg in n.split('.') {
        let mut c = seg.chars();
        match c.next() {
            Some(x)
                if (if first {
                    x.is_ascii_lowercase()
                } else {
                    x.is_ascii_lowercase() || x.is_ascii_digit()
                }) => {}
            _ => return false,
        }
        if !c.all(|x| x.is_ascii_lowercase() || x.is_ascii_digit() || x == '-') {
            return false;
        }
        first = false;
    }
    true
}

/// "fnmatch, where a leading "**/" may also match nothing ("**/*.pdf" matches "a.pdf" as well as "x/a.pdf")."
fn matches(rel: &str, patterns: &[String]) -> bool {
    patterns
        .iter()
        .any(|p| fnmatch(rel, p) || (p.starts_with("**/") && fnmatch(rel, &p[3..])))
}

fn posix(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

/// {relative path: path} of the files under root matching include and not exclude (like os.walk: .git folders and
/// symlinked folders are not entered). Folders holding a .rag.toml are added to markers.
fn collect(
    root: &Path,
    include: &[String],
    exclude: &[String],
    mut markers: Option<&mut HashSet<PathBuf>>,
) -> BTreeMap<String, PathBuf> {
    let mut files = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(folder) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&folder) else { continue };
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = folder.join(&name);
            let Ok(ft) = entry.file_type() else { continue };
            let is_dir =
                ft.is_dir() || (ft.is_symlink() && std::fs::metadata(&path).map(|m| m.is_dir()).unwrap_or(false));
            if is_dir {
                if name != ".git" && !ft.is_symlink() {
                    stack.push(path);
                }
                continue;
            }
            if name == FOLDER_CONFIG
                && let Some(m) = markers.as_deref_mut()
            {
                m.insert(folder.clone());
            }
            let rel = posix(path.strip_prefix(root).unwrap_or(&path));
            if matches(&rel, include) && !matches(&rel, exclude) {
                files.insert(rel, path);
            }
        }
    }
    files
}

/// A folder's .rag.toml: its namespace and its [[file]] entries (path pattern, namespace).
struct FolderCfg {
    namespace: Option<String>,
    files: Vec<(String, String)>,
}

/// Which namespace a file of a shelf source goes to, from the nearest setting above it: a [[file]] entry of a
/// .rag.toml, a .rag.toml's namespace, a folder of [source.map]. In one folder the .rag.toml wins over the map. A
/// .rag.toml that cannot be read, or names a namespace on the wrong side of a wall, stops the files under it.
struct Folders<'a> {
    name: String,
    wall: Option<String>,
    walls: &'a [String],
    /// mapped folder -> namespace (before sub), in the order of the map
    mapped: Vec<(PathBuf, String)>,
    configs: HashMap<PathBuf, Option<Result<FolderCfg, ()>>>,
    markers: HashSet<PathBuf>,
    bad: usize,
}

impl<'a> Folders<'a> {
    fn new(src: &Table, map: &[(String, Vec<String>)], walls: &'a [String]) -> Folders<'a> {
        let mut mapped: Vec<(PathBuf, String)> = Vec::new();
        for (ns, folders) in map {
            for f in folders {
                let p = PathBuf::from(f);
                if !mapped.iter().any(|(q, _)| *q == p) {
                    mapped.push((p, ns.clone()));
                }
            }
        }
        let wall = map.iter().find_map(|(ns, _)| wall_of(ns, walls));
        Folders {
            name: s(src, "name").unwrap_or("").into(),
            wall,
            walls,
            mapped,
            configs: HashMap::new(),
            markers: HashSet::new(),
            bad: 0,
        }
    }

    fn config(&mut self, folder: &Path) -> Option<&Result<FolderCfg, ()>> {
        if !self.configs.contains_key(folder) {
            let mut got = None;
            if self.markers.contains(folder) {
                let path = folder.join(FOLDER_CONFIG);
                let parsed = std::fs::read_to_string(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|t| t.parse::<Table>().map_err(|e| e.to_string()))
                    .and_then(|t| self.check(&t));
                got = Some(match parsed {
                    Ok(c) => Ok(c),
                    Err(problem) => {
                        self.bad += 1;
                        log(&format!(
                            "{}: {}: {problem}; the files under it are not read",
                            self.name,
                            path.display()
                        ));
                        Err(())
                    }
                });
            }
            self.configs.insert(folder.to_path_buf(), got);
        }
        self.configs[folder].as_ref()
    }

    /// The settings of a .rag.toml, or what is wrong with it. Only namespace (or shelf) and [[file]] (path,
    /// namespace or shelf) may be set.
    fn check(&self, t: &Table) -> Result<FolderCfg, String> {
        let unknown: Vec<&String> = t
            .keys()
            .filter(|k| !["namespace", "shelf", "file"].contains(&k.as_str()))
            .collect();
        if !unknown.is_empty() {
            return Err(format!("unknown keys {unknown:?} (only namespace and [[file]])"));
        }
        let ns_of = |x: &Table| -> Result<Option<String>, String> {
            match (x.get("namespace"), x.get("shelf")) {
                (Some(_), Some(_)) => Err("namespace and shelf are the same setting; give one".into()),
                (Some(Toml::String(v)), None) | (None, Some(Toml::String(v))) => Ok(Some(v.clone())),
                (None, None) => Ok(None),
                _ => Err("a namespace is a string".into()),
            }
        };
        let namespace = ns_of(t)?;
        let mut files = Vec::new();
        for f in t.get("file").and_then(Toml::as_array).map(Vec::as_slice).unwrap_or(&[]) {
            let Some(f) = f.as_table() else {
                return Err("each [[file]] needs path and namespace, nothing else".into());
            };
            let extra = f.keys().any(|k| !["path", "namespace", "shelf"].contains(&k.as_str()));
            match (s(f, "path"), ns_of(f)?) {
                (Some(p), Some(ns)) if !extra => files.push((p.to_string(), ns)),
                _ => return Err("each [[file]] needs path and namespace, nothing else".into()),
            }
        }
        for ns in namespace.iter().chain(files.iter().map(|(_, n)| n)) {
            if !is_namespace(ns) {
                return Err(format!("bad namespace name {ns:?}"));
            }
            let w = wall_of(ns, self.walls);
            if w != self.wall {
                return Err(match (&w, &self.wall) {
                    (Some(w), _) => format!("namespace {ns} is behind the wall {w}; this source is not"),
                    (None, Some(mine)) => format!("namespace {ns} is outside the wall {mine} of this source"),
                    _ => unreachable!(),
                });
            }
        }
        Ok(FolderCfg { namespace, files })
    }

    /// (namespace before sub, or None; which setting decided it) for a file under top (a mapped folder).
    fn subject(&mut self, path: &Path, top: &Path) -> (Option<String>, String) {
        let mut folder = path.parent().unwrap_or(Path::new("/")).to_path_buf();
        loop {
            match self.config(&folder) {
                Some(Err(())) => return (None, format!("{} (broken)", folder.join(FOLDER_CONFIG).display())),
                Some(Ok(c)) => {
                    let rel = posix(path.strip_prefix(&folder).unwrap_or(path));
                    for (pat, ns) in &c.files {
                        if matches(&rel, std::slice::from_ref(pat)) {
                            return (
                                Some(ns.clone()),
                                format!("{} [[file]] {pat}", folder.join(FOLDER_CONFIG).display()),
                            );
                        }
                    }
                    if let Some(ns) = &c.namespace {
                        return (Some(ns.clone()), folder.join(FOLDER_CONFIG).display().to_string());
                    }
                }
                None => {}
            }
            if let Some((_, ns)) = self.mapped.iter().find(|(p, _)| *p == folder) {
                return (Some(ns.clone()), format!("map ({})", folder.display()));
            }
            match folder.parent() {
                Some(up) if folder != top => folder = up.to_path_buf(),
                _ => return (None, String::new()),
            }
        }
    }
}

fn wall_of(ns: &str, walls: &[String]) -> Option<String> {
    let first = ns.split('.').next().unwrap_or("");
    walls.iter().find(|w| *w == first).cloned()
}

/// The map of a shelf source ([source.map], in its order), or of a shelves source (its folders).
fn shelf_map(src: &Table) -> Result<Vec<(String, Vec<String>)>, String> {
    if s(src, "type") == Some("shelves") {
        let Some(path) = s(src, "path") else {
            return Err("a shelves source needs path".into());
        };
        let root = Path::new(path);
        let mut names: Vec<String> = std::fs::read_dir(root)
            .map(|rd| {
                rd.flatten()
                    .filter(|e| e.path().is_dir())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .filter(|n| is_namespace(n))
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        if let Some(only) = s(src, "only") {
            let (keep, skipped): (Vec<String>, Vec<String>) = names
                .into_iter()
                .partition(|n| format!(".{n}.").contains(&format!(".{only}.")));
            if !skipped.is_empty() {
                log(&format!(
                    "{}: folders without .{only}. in the name, skipped: {}",
                    s(src, "name").unwrap_or(""),
                    skipped.join(", ")
                ));
            }
            names = keep;
        }
        let pre = s(src, "prefix").map(|p| format!("{p}.")).unwrap_or_default();
        return Ok(names
            .into_iter()
            .map(|n| (format!("{pre}{n}"), vec![posix(&root.join(&n))]))
            .collect());
    }
    let Some(map) = src.get("map").and_then(Toml::as_table) else {
        return Err("a shelf source needs [source.map]".into());
    };
    Ok(map
        .iter()
        .map(|(ns, v)| {
            (
                ns.clone(),
                match v {
                    Toml::String(x) => vec![x.clone()],
                    Toml::Array(a) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
                    _ => vec![],
                },
            )
        })
        .collect())
}

type Origins = BTreeMap<String, (Option<String>, String)>;

/// A shelf source's files: ({key: file}, key -> namespace, errors). Keys are the files' absolute paths without the
/// leading "/", so files of different folders never collide. origins, if given, gets {key: (namespace or None,
/// which setting decided it)} for `kurag-sync shelves`.
fn shelf_files(
    src: &Table,
    include: &[String],
    exclude: &[String],
    walls: &[String],
    mut origins: Option<&mut Origins>,
) -> (BTreeMap<String, PathBuf>, HashMap<String, String>, usize) {
    let (mut files, mut namespace, mut bad) = (BTreeMap::new(), HashMap::new(), 0);
    let map = match shelf_map(src) {
        Ok(m) => m,
        Err(e) => {
            log(&format!("{}: {e}", s(src, "name").unwrap_or("")));
            return (files, namespace, 1);
        }
    };
    let sub = s(src, "sub").unwrap_or("");
    let mut folders = Folders::new(src, &map, walls);
    let tops: Vec<PathBuf> = folders.mapped.iter().map(|(p, _)| p.clone()).collect();
    for top in tops {
        if !top.is_dir() {
            bad += 1;
            log(&format!("{}: {} is not a directory", folders.name, top.display()));
            continue;
        }
        let found = collect(&top, include, exclude, Some(&mut folders.markers));
        // files under two mapped folders (one inside the other) are the same file with the same answer
        let mut paths: Vec<PathBuf> = found.into_values().collect();
        paths.sort();
        for path in paths {
            let key = posix(&path).trim_start_matches('/').to_string();
            if files.contains_key(&key) || origins.as_ref().is_some_and(|o| o.contains_key(&key)) {
                continue;
            }
            let (subject, from) = folders.subject(&path, &top);
            let ns = subject.map(|x| if sub.is_empty() { x } else { format!("{x}.{sub}") });
            if let Some(o) = origins.as_deref_mut() {
                o.insert(key.clone(), (ns.clone(), from));
            }
            if let Some(ns) = ns {
                files.insert(key.clone(), path);
                namespace.insert(key, ns);
            }
        }
    }
    (files, namespace, bad + folders.bad)
}

fn in_namespace(ns: &str, under: Option<&str>) -> bool {
    match under {
        None => true,
        Some(u) => ns == u || ns.starts_with(&format!("{u}.")),
    }
}

/// Of files with the same folder and name but different extensions in prefer (book.epub and book.pdf), only the one
/// whose extension comes first; files of other extensions are kept as they are.
fn one_format(files: BTreeMap<String, PathBuf>, prefer: &[String]) -> BTreeMap<String, PathBuf> {
    let ext_of = |rel: &str| {
        rel.rsplit_once('.')
            .map(|(_, e)| e.to_lowercase())
            .unwrap_or_else(|| rel.to_lowercase())
    };
    let mut best: HashMap<String, String> = HashMap::new();
    for rel in files.keys() {
        let Some((stem, _)) = rel.rsplit_once('.') else {
            continue;
        };
        let ext = ext_of(rel);
        if stem.is_empty() || !prefer.contains(&ext) {
            continue;
        }
        let rank = |e: &str| prefer.iter().position(|p| p == e).unwrap_or(usize::MAX);
        match best.get(stem) {
            Some(b) if rank(&ext_of(b)) <= rank(&ext) => {}
            _ => {
                best.insert(stem.to_string(), rel.clone());
            }
        }
    }
    let keep: HashSet<&String> = best.values().collect();
    let left: Vec<String> = files
        .keys()
        .filter(|r| prefer.contains(&ext_of(r)) && !keep.contains(r))
        .cloned()
        .collect();
    if let Some(first) = left.first() {
        log(&format!(
            "{} file(s) also in a preferred format, skipped (e.g. {first})",
            left.len()
        ));
    }
    files.into_iter().filter(|(r, _)| !left.contains(r)).collect()
}

/// The server: /upsert, /delete (by id prefixes), /namespaces.
pub struct Rag {
    url: String,
    key: String,
    http: reqwest::blocking::Client,
}

/// A request that failed: the server answered with an error (counted, the run goes on) or could not be reached
/// (the run stops).
pub enum Fail {
    Http(String),
    Net(String),
}

impl Rag {
    pub fn new(url: &str, key: &str) -> Rag {
        let http = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(600))
            .build()
            .expect("http client");
        Rag {
            url: url.trim_end_matches('/').into(),
            key: key.into(),
            http,
        }
    }

    fn call(&self, path: &str, body: Option<&Value>) -> Result<Value, Fail> {
        let url = format!("{}{path}", self.url);
        let req = match body {
            Some(b) => self.http.post(&url).json(b),
            None => self.http.get(&url),
        };
        let r = req
            .header("X-API-Key", &self.key)
            .send()
            .map_err(|e| Fail::Net(format!("{url}: {e}")))?;
        let status = r.status();
        let text = r.text().map_err(|e| Fail::Net(format!("{url}: {e}")))?;
        if !status.is_success() {
            return Err(Fail::Http(format!(
                "HTTP {} {}",
                status.as_u16(),
                text.chars().take(200).collect::<String>()
            )));
        }
        Ok(serde_json::from_str(&text).unwrap_or(Value::Null))
    }

    fn upsert(&self, ns: &str, docs: &[Value]) -> Result<(), Fail> {
        for batch in docs.chunks(BATCH) {
            self.call("/upsert", Some(&json!({"namespace": ns, "documents": batch})))?;
        }
        Ok(())
    }

    fn delete(&self, ns: &str, prefixes: &[String]) -> Result<(), Fail> {
        if !prefixes.is_empty() {
            self.call("/delete", Some(&json!({"namespace": ns, "prefixes": prefixes})))?;
        }
        Ok(())
    }

    pub fn namespaces(&self) -> Result<Value, Fail> {
        self.call("/namespaces", None)
    }
}

pub fn load_json(path: &Path) -> Map<String, Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|v| v.as_object().cloned())
        .unwrap_or_default()
}

/// Written whole and renamed into place, indented like txtai-kura-sync's files.
fn save_json(path: &Path, data: &Map<String, Value>) {
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, serde_json::ser::PrettyFormatter::with_indent(b" "));
    serde::Serialize::serialize(data, &mut ser).expect("json");
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, &buf)
        .and_then(|_| std::fs::rename(&tmp, path))
        .is_err()
    {
        log(&format!("could not write {}", path.display()));
    }
}

fn git_auth(src: &Table, env: &HashMap<String, String>) -> Vec<String> {
    let token = s(src, "token_env")
        .and_then(|k| env.get(k))
        .cloned()
        .unwrap_or_default();
    if token.is_empty() {
        return vec![];
    }
    let value = match s(src, "token_auth").unwrap_or("token") {
        "bearer" => format!("Bearer {token}"),
        "basic" => {
            use base64_lite::encode;
            format!("Basic {}", encode(format!("x-access-token:{token}").as_bytes()))
        }
        _ => format!("token {token}"),
    };
    vec!["-c".into(), format!("http.extraHeader=Authorization: {value}")]
}

mod base64_lite {
    pub fn encode(b: &[u8]) -> String {
        const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for c in b.chunks(3) {
            let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
            for k in 0..4 {
                out.push(if k <= c.len() {
                    T[(n >> (18 - 6 * k) & 63) as usize] as char
                } else {
                    '='
                });
            }
        }
        out
    }
}

/// Clone or update the repository (shallow) into cache/git/<name> and return the work tree.
fn fetch_git(src: &Table, env: &HashMap<String, String>, paths: &Paths) -> Result<PathBuf, String> {
    let name = s(src, "name").unwrap_or("");
    let dest = paths.cache.join("git").join(name);
    let auth = git_auth(src, env);
    let branch = s(src, "branch").unwrap_or("main");
    let run = |args: Vec<String>| -> Result<(), String> {
        let st = std::process::Command::new("git")
            .args(&args)
            .status()
            .map_err(|e| format!("git: {e}"))?;
        if st.success() {
            Ok(())
        } else {
            Err(format!(
                "{name}: git {} failed ({st})",
                args.iter()
                    .find(|a| ["fetch", "reset", "clone"].contains(&a.as_str()))
                    .map(String::as_str)
                    .unwrap_or("")
            ))
        }
    };
    let d = dest.to_string_lossy().into_owned();
    if dest.join(".git").exists() {
        run([
            auth.clone(),
            vec![
                "-C".into(),
                d.clone(),
                "fetch".into(),
                "-q".into(),
                "--depth".into(),
                "1".into(),
                "origin".into(),
                branch.into(),
            ],
        ]
        .concat())?;
        run(vec![
            "-C".into(),
            d,
            "reset".into(),
            "-q".into(),
            "--hard".into(),
            format!("origin/{branch}"),
        ])?;
    } else {
        let _ = std::fs::create_dir_all(dest.parent().unwrap_or(Path::new(".")));
        let url = s(src, "url").ok_or(format!("{name}: a git source needs url"))?;
        run([
            auth,
            vec![
                "clone".into(),
                "-q".into(),
                "--depth".into(),
                "1".into(),
                "--branch".into(),
                branch.into(),
                url.into(),
                d,
            ],
        ]
        .concat())?;
    }
    Ok(dest)
}

fn label_of(src: &Table) -> String {
    match s(src, "label") {
        Some(l) => l.to_string(),
        None if matches!(s(src, "type"), Some("shelf" | "shelves")) => String::new(),
        None => s(src, "name").unwrap_or("").to_string(),
    }
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Which namespace a file goes to (None: none).
type NamespaceOf = Box<dyn Fn(&str) -> Option<String>>;
/// A part waiting to be sent: key, sha256, namespace, the namespace it was in before (if sent before), its chunks.
type Pending = (String, String, String, Option<String>, Vec<Value>);

pub struct RunArgs<'a> {
    pub only: Option<&'a str>,
    pub kind: Option<&'a str>,
    pub names: Option<Vec<String>>,
    pub under: Option<&'a str>,
    pub dry: bool,
}

/// Send what changed. under: only the files going to that namespace or below it; what is not found any more is then
/// deleted only there. A source's max_minutes stops it after that long; what was sent is kept (state is saved every
/// batch), the rest goes in on the next run, and nothing is deleted as gone until a run sees every file.
/// Ok(1) when something failed (counted), Err when the server or git could not be reached.
pub fn run(
    src_cfg: &Sources,
    env: &HashMap<String, String>,
    paths: &Paths,
    rag: &Rag,
    a: &RunArgs,
) -> Result<i32, String> {
    let mut state = load_json(&paths.state_file());
    let mut file_stamps = load_json(&paths.files_file());
    let (mut changed, mut errors) = (0usize, 0usize);
    let words = &src_cfg.words;
    let other: Vec<String> = src_cfg
        .list()
        .iter()
        .filter(|t| !TYPES.contains(&s(t, "type").unwrap_or("")))
        .map(|t| format!("{} ({})", s(t, "name").unwrap_or(""), s(t, "type").unwrap_or("no type")))
        .collect();
    if !other.is_empty() && a.names.is_none() {
        log(&format!(
            "left out, types kurag-sync does not handle: {}",
            other.join(", ")
        ));
    }
    for src in src_cfg.list() {
        let name = s(src, "name").unwrap_or("").to_string();
        let typ = s(src, "type").unwrap_or("");
        if a.only.is_some_and(|o| o != name)
            || a.names.as_ref().is_some_and(|n| !n.contains(&name))
            || a.kind.is_some_and(|k| s(src, "kind") != Some(k))
            || !TYPES.contains(&typ)
        {
            continue;
        }
        let include = strings(src, "include", &["**/*.md"]);
        let exclude = strings(src, "exclude", &[]);
        let (mut files, ns_of): (BTreeMap<String, PathBuf>, NamespaceOf) = match typ {
            "shelf" | "shelves" => {
                let (f, map, bad) = shelf_files(src, &include, &exclude, &src_cfg.walls, None);
                errors += bad;
                (f, Box::new(move |rel: &str| map.get(rel).cloned()))
            }
            _ => {
                let Some(ns) = s(src, "namespace").map(String::from) else {
                    errors += 1;
                    log(&format!("{name}: no namespace, skipped"));
                    continue;
                };
                let roots: Vec<PathBuf> = if typ == "git" {
                    vec![fetch_git(src, env, paths)?]
                } else {
                    let roots: Vec<PathBuf> = strings(src, "path", &[]).iter().map(PathBuf::from).collect();
                    let names: Vec<String> = roots
                        .iter()
                        .map(|r| {
                            r.file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_default()
                        })
                        .collect();
                    let unique: HashSet<&String> = names.iter().collect();
                    if roots.is_empty() || roots.iter().any(|r| !r.is_dir()) || unique.len() < names.len() {
                        errors += 1;
                        let msg = match roots.iter().find(|r| !r.is_dir()) {
                            Some(m) => format!("{} is not a directory", m.display()),
                            None if roots.is_empty() => "a dir source needs path".into(),
                            None => format!("two paths end in the same folder name ({})", names.join(", ")),
                        };
                        log(&format!("{name}: {msg}"));
                        continue;
                    }
                    roots
                };
                let files = if roots.len() > 1 {
                    // several paths: ids start with the folder name
                    let mut f = BTreeMap::new();
                    for r in &roots {
                        let top = r
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        for (rel, p) in collect(r, &include, &exclude, None) {
                            f.insert(format!("{top}/{rel}"), p);
                        }
                    }
                    f
                } else {
                    collect(&roots[0], &include, &exclude, None)
                };
                (files, Box::new(move |_: &str| Some(ns.clone())))
            }
        };
        let unmapped: Vec<&String> = files.keys().filter(|r| ns_of(r).is_none()).collect();
        if let Some(first) = unmapped.first() {
            log(&format!(
                "{name}: {} file(s) outside every namespace folder, skipped (e.g. {first})",
                unmapped.len()
            ));
        }
        files.retain(|rel, _| ns_of(rel).is_some_and(|ns| in_namespace(&ns, a.under)));
        if src.contains_key("prefer") {
            let prefer: Vec<String> = strings(src, "prefer", &[])
                .iter()
                .map(|e| e.to_lowercase().trim_start_matches('.').to_string())
                .collect();
            files = one_format(files, &prefer);
        }
        let prefix = format!("{name}:");
        let label = label_of(src);
        let mut seen: HashSet<String> = HashSet::new();
        let mut pending: Vec<Pending> = Vec::new();
        let before = errors;
        let mut stopped = false;
        let max_minutes = src
            .get("max_minutes")
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)));
        let deadline = max_minutes.map(|m| now() + m * 60.0);
        let mut stamps: Map<String, Value> = Map::new();
        let mut parts_by_file: HashMap<String, Vec<String>> = HashMap::new();
        for k in state.keys() {
            if let Some(rest) = k.strip_prefix(&prefix) {
                parts_by_file
                    .entry(rest.split('#').next().unwrap_or("").to_string())
                    .or_default()
                    .push(k.clone());
            }
        }

        let flush =
            |pending: &mut Vec<Pending>, state: &mut Map<String, Value>, errors: &mut usize| -> Result<(), String> {
                if pending.is_empty() {
                    return Ok(());
                }
                let result = (|| -> Result<(), Fail> {
                    let olds: HashSet<&String> = pending.iter().filter_map(|p| p.3.as_ref()).collect();
                    for ns in olds {
                        let prefixes: Vec<String> = pending
                            .iter()
                            .filter(|p| p.3.as_ref() == Some(ns))
                            .map(|p| format!("{}#", p.0))
                            .collect();
                        rag.delete(ns, &prefixes)?;
                    }
                    let news: HashSet<&String> = pending.iter().map(|p| &p.2).collect();
                    for ns in news {
                        let docs: Vec<Value> = pending
                            .iter()
                            .filter(|p| &p.2 == ns)
                            .flat_map(|p| p.4.iter().cloned())
                            .collect();
                        rag.upsert(ns, &docs)?;
                    }
                    Ok(())
                })();
                match result {
                    Ok(()) => {
                        for (key, sha, ns, _, docs) in pending.iter() {
                            state.insert(key.clone(), json!({"sha": sha, "namespace": ns, "chunks": docs.len()}));
                        }
                        save_json(&paths.state_file(), state);
                    }
                    Err(Fail::Http(e)) => {
                        *errors += 1;
                        log(&format!("  failed ({} part(s)): {e}", pending.len()));
                    }
                    Err(Fail::Net(e)) => return Err(e),
                }
                pending.clear();
                Ok(())
            };

        for (rel, path) in &files {
            if deadline.is_some_and(|d| now() > d) {
                stopped = true;
                log(&format!(
                    "{name}: {} minutes up; the rest goes in on the next run",
                    max_minutes.unwrap_or(0.0)
                ));
                break;
            }
            let ns = ns_of(rel).unwrap_or_default();
            let Ok(meta) = std::fs::metadata(path) else {
                errors += 1;
                log(&format!("{}: cannot be read", path.display()));
                continue;
            };
            let mtime_ns = {
                use std::os::unix::fs::MetadataExt;
                meta.mtime() as i128 * 1_000_000_000 + meta.mtime_nsec() as i128
            };
            let stamp = json!([meta.len(), mtime_ns as i64, ns]);
            let fkey = format!("{prefix}{rel}");
            if file_stamps.get(&fkey) == Some(&stamp) {
                // unchanged since it was last sent: not even opened
                seen.extend(parts_by_file.get(rel).cloned().unwrap_or_default());
                stamps.insert(fkey, stamp);
                continue;
            }
            let Some(text) = read_text(path, &paths.cache) else {
                errors += 1;
                seen.extend(parts_by_file.get(rel).cloned().unwrap_or_default());
                continue;
            };
            stamps.insert(fkey, stamp);
            for (part, ptext) in parts_of(rel, &text, words, &label) {
                let key = format!("{prefix}{part}");
                seen.insert(key.clone());
                let sha = sha256_hex(&ptext);
                let old = state.get(&key);
                if old.is_some_and(|o| o["sha"] == json!(sha) && o["namespace"] == json!(ns)) {
                    continue;
                }
                log(&format!("{} {key}", if old.is_some() { "update" } else { "add" }));
                changed += 1;
                if a.dry {
                    continue;
                }
                let docs: Vec<Value> = chunks(&ptext)
                    .into_iter()
                    .enumerate()
                    .map(|(i, c)| json!({"id": format!("{key}#{i}"), "text": c, "source": place(&label, rel)}))
                    .collect();
                let old_ns = old.map(|o| o["namespace"].as_str().unwrap_or(&ns).to_string());
                pending.push((key, sha, ns.clone(), old_ns, docs));
                if pending.iter().map(|p| p.4.len()).sum::<usize>() >= BATCH {
                    flush(&mut pending, &mut state, &mut errors)?;
                }
            }
        }
        flush(&mut pending, &mut state, &mut errors)?;
        // files not reached before the time limit are not gone: delete only after a run that saw every file
        let gone: Vec<String> = if stopped {
            vec![]
        } else {
            state
                .iter()
                .filter(|(k, v)| {
                    k.starts_with(&prefix)
                        && !seen.contains(*k)
                        && in_namespace(v["namespace"].as_str().unwrap_or(""), a.under)
                })
                .map(|(k, _)| k.clone())
                .collect()
        };
        for key in &gone {
            log(&format!("delete {key}"));
        }
        changed += gone.len();
        if !gone.is_empty() && !a.dry {
            let spaces: HashSet<String> = gone
                .iter()
                .filter_map(|k| state[k]["namespace"].as_str().map(String::from))
                .collect();
            let mut result = Ok(());
            for ns in &spaces {
                let prefixes: Vec<String> = gone
                    .iter()
                    .filter(|k| state[*k]["namespace"].as_str() == Some(ns))
                    .map(|k| format!("{k}#"))
                    .collect();
                if let Err(e) = rag.delete(ns, &prefixes) {
                    result = Err(e);
                    break;
                }
            }
            match result {
                Ok(()) => {
                    for k in &gone {
                        state.remove(k);
                    }
                    save_json(&paths.state_file(), &state);
                }
                Err(Fail::Http(e)) => {
                    errors += 1;
                    log(&format!("  failed: {e}"));
                }
                Err(Fail::Net(e)) => return Err(e),
            }
        }
        if !a.dry && errors == before {
            // with errors, every changed file is read again next run
            if !stopped {
                // every file was seen: forget the ones that are gone
                file_stamps
                    .retain(|k, v| !(k.starts_with(&prefix) && in_namespace(v[2].as_str().unwrap_or(""), a.under)));
            }
            file_stamps.extend(stamps);
            save_json(&paths.files_file(), &file_stamps);
        }
    }
    log(&format!(
        "done: {changed} change(s), {errors} error(s){}",
        if a.dry { " (dry run)" } else { "" }
    ));
    Ok(if errors > 0 { 1 } else { 0 })
}

// ---- schedule (--due) ----

fn interval(text: &str) -> Option<f64> {
    let (n, unit) = text.split_at(text.len().checked_sub(1)?);
    let n: f64 = n.parse().ok()?;
    Some(
        n * match unit {
            "m" => 60.0,
            "h" => 3600.0,
            "d" => 86400.0,
            _ => return None,
        },
    )
}

/// One field of a cron expression: *, 5, 1-5, 1,3,5, */2, 1-10/3.
fn cron_field(field: &str, value: i64, low: i64, high: i64) -> bool {
    field.split(',').any(|part| {
        let (rng, step) = part.split_once('/').unwrap_or((part, ""));
        let (a, b) = if rng == "*" {
            (low, high)
        } else if let Some((x, y)) = rng.split_once('-') {
            (x.parse().unwrap_or(i64::MAX), y.parse().unwrap_or(i64::MIN))
        } else {
            let v = rng.parse().unwrap_or(i64::MAX);
            (v, v)
        };
        let step: i64 = if step.is_empty() {
            1
        } else {
            step.parse().unwrap_or(1).max(1)
        };
        a <= value && value <= b && (value - a) % step == 0
    })
}

/// Whether the hour of t matches "min hour day month weekday" (the minute is not used: --due runs once an hour).
/// Weekday 0 or 7 = Sunday. Day and weekday both restricted: either may match.
fn cron_hour(expr: &str, t: &chrono::DateTime<chrono::Local>) -> bool {
    use chrono::{Datelike, Timelike};
    let f: Vec<&str> = expr.split_whitespace().collect();
    if f.len() != 5 {
        return false;
    }
    let (hour, day, month, wday) = (f[1], f[2], f[3], f[4]);
    let wd = t.weekday().num_days_from_sunday() as i64;
    let days = cron_field(day, t.day() as i64, 1, 31);
    let wdays = cron_field(wday, wd, 0, 7) || (wd == 0 && cron_field(wday, 7, 0, 7));
    let dayok = if day != "*" && wday != "*" {
        days || wdays
    } else {
        days && wdays
    };
    cron_field(hour, t.hour() as i64, 0, 23) && cron_field(month, t.month() as i64, 1, 12) && dayok
}

/// Names of the sources whose time has come. Schedule: the source's own (cron / every / at), else its kind's
/// ([kinds.<kind>]); a source without one is never due. Meant to be called once an hour.
pub fn due(src_cfg: &Sources, paths: &Paths, now_s: f64) -> Vec<String> {
    let runs = load_json(&paths.runs_file());
    let t = chrono::DateTime::from_timestamp(now_s as i64, 0)
        .map(|d| d.with_timezone(&chrono::Local))
        .unwrap_or_else(chrono::Local::now);
    let kinds = crate::config::table(&src_cfg.cfg, "kinds");
    let mut names = Vec::new();
    for src in src_cfg.list() {
        if !TYPES.contains(&s(src, "type").unwrap_or("")) {
            continue;
        }
        let own: Table = ["cron", "every", "at"]
            .iter()
            .filter_map(|k| src.get(*k).map(|v| (k.to_string(), v.clone())))
            .collect();
        let sched = if !own.is_empty() {
            own
        } else {
            s(src, "kind")
                .and_then(|k| kinds.and_then(|t| t.get(k)).and_then(Toml::as_table))
                .cloned()
                .unwrap_or_default()
        };
        let name = s(src, "name").unwrap_or("").to_string();
        let last = runs.get(&name).and_then(Value::as_f64).unwrap_or(0.0);
        if let Some(cron) = sched.get("cron").and_then(Toml::as_str) {
            if cron_hour(cron, &t) && now_s - last >= 3000.0 {
                names.push(name);
            }
        } else if let Some(every) = sched.get("every").and_then(Toml::as_str) {
            use chrono::Timelike;
            if let Some(at) = sched.get("at") {
                let at = at
                    .as_str()
                    .and_then(|x| x.parse::<u32>().ok())
                    .or_else(|| at.as_integer().map(|i| i as u32));
                if at != Some(t.hour()) {
                    continue;
                }
            }
            // 10 minutes of slack: an hourly source run at 10:15:40 is due again at 11:15:05
            if interval(every).is_some_and(|iv| now_s - last >= iv - 600.0) {
                names.push(name);
            }
        }
    }
    names
}

/// Record when the run started, not when it ended: a run from 05:15 that ends at 06:06 must not count as the 06
/// o'clock run of a cron source ("0 5-7 * * *"), or the 06:15 slot is skipped.
pub fn mark_runs(paths: &Paths, names: &[String], started: f64) {
    let mut runs = load_json(&paths.runs_file());
    for n in names {
        runs.insert(n.clone(), json!(started));
    }
    save_json(&paths.runs_file(), &runs);
}

/// The sources a plain run (not --due) covers, for mark_runs.
pub fn ran(src_cfg: &Sources, only: Option<&str>, kind: Option<&str>) -> Vec<String> {
    src_cfg
        .list()
        .iter()
        .filter(|t| {
            TYPES.contains(&s(t, "type").unwrap_or(""))
                && only.is_none_or(|o| s(t, "name") == Some(o))
                && kind.is_none_or(|k| s(t, "kind") == Some(k))
        })
        .filter_map(|t| s(t, "name").map(String::from))
        .collect()
}

// ---- status and shelves ----

pub fn status(src_cfg: &Sources, paths: &Paths, rag: &Rag) -> i32 {
    let state = load_json(&paths.state_file());
    let chunks: u64 = state.values().filter_map(|v| v["chunks"].as_u64()).sum();
    println!("tracked parts: {}, chunks: {chunks}", state.len());
    if !src_cfg.disabled.is_empty() {
        println!(
            "disabled sources (enabled = false; what they sent stays): {}",
            src_cfg.disabled.join(", ")
        );
    }
    match rag.namespaces() {
        Ok(v) => {
            let list: Vec<String> = v
                .as_object()
                .map(|m| m.iter().map(|(k, n)| format!("{k} {n}")).collect())
                .unwrap_or_default();
            println!("namespaces (chunks): {}", list.join(", "));
            0
        }
        Err(Fail::Http(e) | Fail::Net(e)) => {
            log(&format!("error: {e}"));
            1
        }
    }
}

/// Per folder (or per file), the namespace its files go to and which setting decided it (the map or a .rag.toml);
/// files under a broken .rag.toml are listed as "not read". Reads only the folders, sends nothing.
pub fn shelves(src_cfg: &Sources, only: Option<&str>, per_file: bool) -> i32 {
    let mut bad = 0;
    for name in &src_cfg.disabled {
        println!("== {name}: enabled = false, not read");
    }
    for src in src_cfg.list() {
        if !matches!(s(src, "type"), Some("shelf" | "shelves")) || only.is_some_and(|o| s(src, "name") != Some(o)) {
            continue;
        }
        let mut origins = Origins::new();
        let (_, _, n) = shelf_files(
            src,
            &strings(src, "include", &["**/*.md"]),
            &strings(src, "exclude", &[]),
            &src_cfg.walls,
            Some(&mut origins),
        );
        bad += n;
        let mut rows: BTreeMap<(String, String, String), usize> = BTreeMap::new();
        for (key, (ns, from)) in origins {
            let file = format!("/{key}");
            let folder = Path::new(&file)
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default();
            let shown = if per_file { file.clone() } else { folder.clone() };
            let from = from
                .replace(&format!(" ({folder})"), "")
                .replace(&format!("{folder}/"), "");
            *rows
                .entry((shown, ns.unwrap_or_else(|| "(not read)".into()), from))
                .or_insert(0) += 1;
        }
        println!("== {}", s(src, "name").unwrap_or(""));
        for ((place, ns, from), count) in rows {
            println!(
                "{count:5}  {place}  ->  {ns}  [{}]",
                if from.is_empty() { "no setting" } else { &from }
            );
        }
    }
    if bad > 0 { 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("kurag-sync-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn names_and_globs() {
        assert!(is_namespace("academic.math.algebra.ag.book"));
        assert!(is_namespace("b2.x1"));
        assert!(!is_namespace("Academic") && !is_namespace("a..b") && !is_namespace("1a"));
        assert!(matches("a.pdf", &["**/*.pdf".into()]));
        assert!(matches("x/a.pdf", &["**/*.pdf".into()]));
        assert!(!matches("a.PDF", &["**/*.pdf".into()]));
        assert_eq!(base64_lite::encode(b"x-access-token:t"), "eC1hY2Nlc3MtdG9rZW46dA==");
    }

    #[test]
    fn cron_and_intervals() {
        assert!(cron_field("5-7", 6, 0, 23) && !cron_field("5-7", 8, 0, 23));
        assert!(cron_field("*/2", 4, 0, 23) && !cron_field("*/2", 5, 0, 23));
        assert!(cron_field("1,3,5", 3, 0, 23));
        assert_eq!(interval("48h"), Some(172800.0));
        assert_eq!(interval("1d"), Some(86400.0));
    }

    #[test]
    fn folder_settings_and_walls() {
        let d = tmp("folders");
        let top = d.join("books/math");
        std::fs::create_dir_all(top.join("alg/sub")).unwrap();
        std::fs::create_dir_all(top.join("bad")).unwrap();
        std::fs::write(top.join("a.pdf"), "x").unwrap();
        std::fs::write(top.join("alg/b.pdf"), "x").unwrap();
        std::fs::write(top.join("alg/sub/c.pdf"), "x").unwrap();
        std::fs::write(top.join("alg/sub/d.pdf"), "x").unwrap();
        std::fs::write(top.join("bad/e.pdf"), "x").unwrap();
        std::fs::write(top.join("alg/.rag.toml"), "shelf = \"academic.math.algebra\"\n").unwrap();
        std::fs::write(
            top.join("alg/sub/.rag.toml"),
            "[[file]]\npath = \"d.pdf\"\nnamespace = \"academic.math.ag\"\n",
        )
        .unwrap();
        std::fs::write(top.join("bad/.rag.toml"), "shelf = \"private.x\"\n").unwrap();
        let src: Table = format!(
            "name = \"b\"\ntype = \"shelf\"\nsub = \"book\"\n[map]\n\"academic.math\" = \"{}\"\n",
            top.display()
        )
        .parse()
        .unwrap();
        let walls = vec!["private".to_string()];
        let mut origins = Origins::new();
        let (files, ns, bad) = shelf_files(&src, &["**/*.pdf".into()], &[], &walls, Some(&mut origins));
        let key = |p: &str| posix(&top.join(p)).trim_start_matches('/').to_string();
        assert_eq!(ns[&key("a.pdf")], "academic.math.book");
        assert_eq!(ns[&key("alg/b.pdf")], "academic.math.algebra.book");
        assert_eq!(ns[&key("alg/sub/c.pdf")], "academic.math.algebra.book");
        assert_eq!(ns[&key("alg/sub/d.pdf")], "academic.math.ag.book");
        assert!(!files.contains_key(&key("bad/e.pdf")));
        assert_eq!(bad, 1);
        assert_eq!(origins[&key("bad/e.pdf")].0, None);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn one_format_prefers() {
        let f: BTreeMap<String, PathBuf> = ["a/b.pdf", "a/b.epub", "a/c.pdf", "a/d.md"]
            .iter()
            .map(|k| (k.to_string(), PathBuf::from(k)))
            .collect();
        let kept: Vec<String> = one_format(f, &["epub".into(), "pdf".into()]).into_keys().collect();
        assert_eq!(kept, ["a/b.epub", "a/c.pdf", "a/d.md"]);
    }
}
