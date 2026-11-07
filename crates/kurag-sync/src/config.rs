//! Settings: the sources file (TOML) with `<name>.local.toml` next to it, and KEY=value lines from an env file.
//! The same files and keys as txtai-kura-sync, so an existing setup moves over as it is.

use std::collections::HashMap;
use std::path::Path;

use toml::{Table, Value};

/// The config file, then `<name>.local.toml` next to it (not in git, like .env: places that should not be on
/// record). A local `[[source]]` replaces the shared one of the same name, new names are added after the shared
/// ones; local tables `[rag]`, `[text]`, `[extract]` and `[kinds]` replace the shared one's keys; a local `walls`
/// replaces the shared list.
pub fn load(path: &Path) -> Result<Table, String> {
    let read = |p: &Path| -> Result<Table, String> {
        let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
        text.parse::<Table>().map_err(|e| format!("{}: {e}", p.display()))
    };
    let mut cfg = read(path)?;
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("sources");
    let local = path.with_file_name(format!("{stem}.local.toml"));
    if local.exists() {
        let extra = read(&local)?;
        let name_of = |s: &Value| s.get("name").and_then(Value::as_str).map(str::to_string);
        if let Some(Value::Array(more)) = extra.get("source") {
            let names: Vec<String> = more.iter().filter_map(name_of).collect();
            let mut list: Vec<Value> = match cfg.get("source") {
                Some(Value::Array(a)) => a.iter().filter(|s| !name_of(s).is_some_and(|n| names.contains(&n))).cloned().collect(),
                _ => Vec::new(),
            };
            list.extend(more.iter().cloned());
            cfg.insert("source".into(), Value::Array(list));
        }
        for name in ["rag", "text", "extract", "kinds"] {
            if let Some(Value::Table(more)) = extra.get(name) {
                let table = cfg.entry(name).or_insert_with(|| Value::Table(Table::new()));
                if let Value::Table(t) = table {
                    t.extend(more.clone());
                }
            }
        }
        if let Some(w) = extra.get("walls") {
            cfg.insert("walls".into(), w.clone());
        }
    }
    Ok(cfg)
}

/// A table of the config (empty when missing).
pub fn table<'a>(cfg: &'a Table, name: &str) -> Option<&'a Table> {
    cfg.get(name).and_then(Value::as_table)
}

pub fn string(t: Option<&Table>, key: &str) -> Option<String> {
    t?.get(key)?.as_str().map(str::to_string)
}

/// KEY=value lines; blank lines and # comments left out, quotes around values removed.
pub fn load_env(path: &Path) -> HashMap<String, String> {
    let mut env = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(path) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                env.insert(k.trim().to_string(), v.trim().trim_matches(|c| c == '\'' || c == '"').to_string());
            }
        }
    }
    env
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_file_and_env() {
        let dir = std::env::temp_dir().join(format!("kurag-sync-config-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("kura.toml"), "[rag]\nurl = \"http://a\"\napi_key_env = \"K\"\n[extract]\nmodel = \"m\"\n").unwrap();
        std::fs::write(dir.join("kura.local.toml"), "[rag]\nurl = \"http://b\"\n[[source]]\nname = \"a\"\nkind = \"local\"\n\
                                                    [[source]]\nname = \"c\"\n").unwrap();
        std::fs::write(dir.join(".env"), "# c\nK='secret'\n\nX = 1\n").unwrap();
        let mut shared = std::fs::read_to_string(dir.join("kura.toml")).unwrap();
        shared.push_str("[[source]]\nname = \"a\"\n[[source]]\nname = \"b\"\n");
        std::fs::write(dir.join("kura.toml"), shared).unwrap();
        let cfg = load(&dir.join("kura.toml")).unwrap();
        let names: Vec<&str> = cfg["source"].as_array().unwrap().iter().map(|s| s["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["b", "a", "c"]);
        assert_eq!(cfg["source"][1]["kind"].as_str(), Some("local"));
        assert_eq!(string(table(&cfg, "rag"), "url").as_deref(), Some("http://b"));
        assert_eq!(string(table(&cfg, "rag"), "api_key_env").as_deref(), Some("K"));
        assert_eq!(string(table(&cfg, "extract"), "model").as_deref(), Some("m"));
        let env = load_env(&dir.join(".env"));
        assert_eq!((env["K"].as_str(), env["X"].as_str()), ("secret", "1"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
