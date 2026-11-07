//! kurag-sync: the client that fills a kurag server (docs/design.md, "Why the sync client is here"), with the
//! command line and config of txtai-kura-sync:
//!
//!   kurag-sync [--config sources.toml] [--data DIR] [--env-file .env] run [--source NAME] [--kind KIND]
//!              [--namespace NAMESPACE] [--due] [--dry-run]       send what changed (sync.rs)
//!   kurag-sync ... status                                        parts tracked here, chunks per namespace there
//!   kurag-sync ... shelves [--source NAME] [--files]              which namespace each folder goes to, and why
//!   kurag-sync text --file PATH                                   the text run reads from a file (no cache)
//!   kurag-sync ... extract --namespace NAMESPACE
//!              [--model M] [--api-base URL ... | --api-bases-file F] [--json-mode schema|object|none]
//!              [--api-key-env VAR] [--parallel N] [--minutes M] [--limit N] [--dry-run]   knowledge graph
//!
//! Config: `[rag] url`, `api_key_env` (default RAG_API_KEY); `[text] intro`, `source`; `walls`; `[kinds.<kind>]`
//! and `[[source]]` (sync.rs); `[extract] model`, `api_base` (a string or a list), `api_key_env`, `json_mode`,
//! `parallel`, `minutes`. Options win over the config. `--data` holds state and cache (default
//! $XDG_STATE_HOME/kurag-sync or ~/.local/state/kurag-sync); extraction keeps no state of its own.

mod config;
mod entities;
mod extract;
mod html;
mod pystr;
mod read;
mod split;
mod sync;

use std::path::PathBuf;
use std::sync::Arc;

use extract::{Bases, JsonMode, Llm, Model, Rag, Settings, log};

const USAGE: &str = "usage: kurag-sync [--config FILE] [--data DIR] [--env-file FILE] COMMAND
  run [--source NAME] [--kind KIND] [--namespace NAMESPACE] [--due] [--dry-run]
  status
  shelves [--source NAME] [--files]
  text --file PATH
  extract --namespace NAMESPACE [--model M] [--api-base URL ... | --api-bases-file FILE] [--json-mode schema|object|none]
          [--api-key-env VAR] [--parallel N] [--minutes M] [--limit N] [--dry-run]";

#[derive(Default)]
struct Args {
    config: Option<PathBuf>,
    data: Option<PathBuf>,
    env_file: Option<PathBuf>,
    source: Option<String>,
    kind: Option<String>,
    due: bool,
    files: bool,
    file: Option<PathBuf>,
    command: Option<String>,
    namespace: Option<String>,
    model: Option<String>,
    api_base: Vec<String>,
    api_bases_file: Option<PathBuf>,
    json_mode: Option<String>,
    api_key_env: Option<String>,
    parallel: Option<usize>,
    minutes: Option<f64>,
    limit: Option<usize>,
    dry_run: bool,
}

fn parse_args(argv: &[String]) -> Result<Args, String> {
    let mut a = Args::default();
    let mut it = argv.iter();
    while let Some(arg) = it.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n, Some(v.to_string())),
            _ => (arg.as_str(), None),
        };
        let mut value = || {
            inline
                .clone()
                .or_else(|| it.next().cloned())
                .ok_or(format!("{name} needs a value"))
        };
        let number = |v: String| v.parse::<f64>().map_err(|_| format!("{name}: not a number: {v}"));
        match name {
            "--config" => a.config = Some(value()?.into()),
            "--data" => a.data = Some(value()?.into()),
            "--source" => a.source = Some(value()?),
            "--kind" => a.kind = Some(value()?),
            "--due" => a.due = true,
            "--files" => a.files = true,
            "--file" => a.file = Some(value()?.into()),
            "--env-file" => a.env_file = Some(value()?.into()),
            "--namespace" => a.namespace = Some(value()?),
            "--model" => a.model = Some(value()?),
            "--api-base" => a.api_base.push(value()?),
            "--api-bases-file" => a.api_bases_file = Some(value()?.into()),
            "--json-mode" => a.json_mode = Some(value()?),
            "--api-key-env" => a.api_key_env = Some(value()?),
            "--parallel" => a.parallel = Some(number(value()?)? as usize),
            "--minutes" => a.minutes = Some(number(value()?)?),
            "--limit" => a.limit = Some(number(value()?)? as usize),
            "--dry-run" => a.dry_run = true,
            "-h" | "--help" => return Err(String::new()),
            "-V" | "--version" => {
                println!("kurag-sync {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            cmd if !cmd.starts_with('-') && a.command.is_none() => a.command = Some(cmd.to_string()),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(a)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let a = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("kurag-sync: {e}");
            }
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    };
    std::process::exit(match a.command.as_deref() {
        Some("extract") => extract_command(a),
        Some("text") => match a.file.as_deref().map(read::fresh_text) {
            Some(Some(t)) => {
                print!("{t}");
                0
            }
            Some(None) => 1,
            None => {
                eprintln!("{USAGE}");
                2
            }
        },
        Some(cmd @ ("run" | "status" | "shelves")) => {
            let cmd = cmd.to_string();
            sync_command(&cmd, a)
        }
        _ => {
            eprintln!("{USAGE}");
            2
        }
    });
}

struct Setup {
    sources: sync::Sources,
    env: std::collections::HashMap<String, String>,
    paths: sync::Paths,
    rag: sync::Rag,
}

fn setup(a: &Args) -> Result<Setup, String> {
    let config_path = a.config.clone().unwrap_or_else(|| PathBuf::from("sources.toml"));
    let cfg = config::load(&config_path)?;
    let env_file = a.env_file.clone().unwrap_or_else(|| config_path.with_file_name(".env"));
    let mut env = config::load_env(&env_file);
    env.extend(std::env::vars());
    let rag_cfg = config::table(&cfg, "rag");
    let url = config::string(rag_cfg, "url").ok_or(format!("{}: [rag] url is missing", config_path.display()))?;
    let key_env = config::string(rag_cfg, "api_key_env").unwrap_or_else(|| "RAG_API_KEY".into());
    let rag = sync::Rag::new(&url, env.get(&key_env).map_or("", String::as_str));
    let data = a.data.clone().unwrap_or_else(|| {
        let base = std::env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".local/state"));
        base.join("kurag-sync")
    });
    Ok(Setup {
        sources: sync::Sources::new(cfg)?,
        env,
        paths: sync::Paths::new(&data),
        rag,
    })
}

fn sync_command(cmd: &str, a: Args) -> i32 {
    let s = match setup(&a) {
        Ok(s) => s,
        Err(e) => {
            log(&format!("error: {e}"));
            return 2;
        }
    };
    if let Some(src) = a.source.as_deref()
        && s.sources.disabled.iter().any(|d| d == src)
    {
        log(&format!("{src}: enabled = false; nothing to do"));
        return 0;
    }
    if let Err(e) = std::fs::create_dir_all(&s.paths.state) {
        log(&format!("error: {}: {e}", s.paths.state.display()));
        return 2;
    }
    match cmd {
        "status" => return sync::status(&s.sources, &s.paths, &s.rag), // read-only: never waits for a running sync
        "shelves" => return sync::shelves(&s.sources, a.source.as_deref(), a.files),
        _ => {}
    }
    // one run at a time: wait (up to 30 minutes) for a run that is already going, instead of failing
    let lock = match std::fs::File::create(s.paths.lock_file()) {
        Ok(f) => f,
        Err(e) => {
            log(&format!("error: {}: {e}", s.paths.lock_file().display()));
            return 2;
        }
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1800);
    loop {
        match lock.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_secs(10))
            }
            Err(std::fs::TryLockError::WouldBlock) => {
                log("another run has been going for 30 minutes; exit");
                return 1;
            }
            Err(std::fs::TryLockError::Error(e)) => {
                log(&format!("error: lock: {e}"));
                return 2;
            }
        }
    }
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);
    let mut args = sync::RunArgs {
        only: a.source.as_deref(),
        kind: a.kind.as_deref(),
        names: None,
        under: a.namespace.as_deref(),
        dry: a.dry_run,
    };
    if a.due {
        let names = sync::due(&s.sources, &s.paths, started);
        log(&format!(
            "due: {}",
            if names.is_empty() {
                "nothing".to_string()
            } else {
                names.join(", ")
            }
        ));
        if names.is_empty() {
            return 0;
        }
        args = sync::RunArgs {
            only: None,
            kind: None,
            names: Some(names.clone()),
            under: None,
            dry: a.dry_run,
        };
        return match sync::run(&s.sources, &s.env, &s.paths, &s.rag, &args) {
            Ok(rc) => {
                if rc == 0 && !a.dry_run {
                    sync::mark_runs(&s.paths, &names, started); // with errors, they are due again next hour
                }
                rc
            }
            Err(e) => {
                log(&format!("error: {e}"));
                1
            }
        };
    }
    match sync::run(&s.sources, &s.env, &s.paths, &s.rag, &args) {
        Ok(rc) => {
            // a full or a kind run also counts for --due (not a part of one namespace)
            if rc == 0 && !a.dry_run && a.namespace.is_none() {
                sync::mark_runs(
                    &s.paths,
                    &sync::ran(&s.sources, a.source.as_deref(), a.kind.as_deref()),
                    started,
                );
            }
            rc
        }
        Err(e) => {
            log(&format!("error: {e}"));
            1
        }
    }
}

fn extract_command(a: Args) -> i32 {
    let config_path = a.config.clone().unwrap_or_else(|| PathBuf::from("sources.toml"));
    let cfg = match config::load(&config_path) {
        Ok(c) => c,
        Err(e) => {
            log(&format!("error: {e}"));
            return 2;
        }
    };
    let env_file = a.env_file.clone().unwrap_or_else(|| config_path.with_file_name(".env"));
    let mut env = config::load_env(&env_file);
    env.extend(std::env::vars());
    let x = config::table(&cfg, "extract");
    let rag_cfg = config::table(&cfg, "rag");
    let Some(namespace) = a.namespace.clone() else {
        log("error: extract needs --namespace");
        return 2;
    };
    let model = a.model.clone().or_else(|| config::string(x, "model"));
    let key_env = a
        .api_key_env
        .clone()
        .or_else(|| config::string(x, "api_key_env"))
        .unwrap_or_default();
    let key = env.get(&key_env).cloned().unwrap_or_default();
    let mode_name = a
        .json_mode
        .clone()
        .or_else(|| config::string(x, "json_mode"))
        .unwrap_or_else(|| "schema".into());
    let Some(mode) = JsonMode::parse(&mode_name) else {
        log("error: json_mode is one of schema, object, none");
        return 2;
    };
    let Some(model) = model else {
        log("error: extract needs --model (or model in [extract])");
        return 2;
    };
    let mut bases = if let Some(file) = a.api_bases_file.clone() {
        match extract::read_api_bases(&file) {
            Ok(first) => Bases::File(file, first),
            Err(e) => {
                log(&format!("error: {e}"));
                return 2;
            }
        }
    } else {
        let from_cfg: Vec<String> = match x.and_then(|t| t.get("api_base")) {
            Some(toml::Value::String(s)) => vec![s.clone()],
            Some(toml::Value::Array(v)) => v.iter().filter_map(|b| b.as_str().map(str::to_string)).collect(),
            _ => Vec::new(),
        };
        let list = if a.api_base.is_empty() {
            from_cfg
        } else {
            a.api_base.clone()
        };
        if list.is_empty() {
            log("error: extract needs --api-base and --model (or api_base / model in [extract])");
            return 2;
        }
        Bases::Fixed(list)
    };
    let Some(url) = config::string(rag_cfg, "url") else {
        log(&format!("error: {}: [rag] url is missing", config_path.display()));
        return 2;
    };
    let rag_key_env = config::string(rag_cfg, "api_key_env").unwrap_or_else(|| "RAG_API_KEY".into());
    let rag = Rag::new(&url, env.get(&rag_key_env).map_or("", String::as_str));
    let int = |k: &str| {
        x.and_then(|t| t.get(k))
            .and_then(|v| v.as_integer().map(|i| i as f64).or_else(|| v.as_float()))
    };
    let settings = Settings {
        namespace,
        parallel: a
            .parallel
            .or_else(|| int("parallel").map(|p| p as usize))
            .unwrap_or(1)
            .max(1),
        minutes: a.minutes.or_else(|| int("minutes")),
        limit: a.limit,
        dry_run: a.dry_run,
        retries: extract::RETRIES.to_vec(),
    };
    let make = |base: &str| -> Arc<dyn Model> { Arc::new(Llm::new(base, &model, &key, mode)) };
    match extract::run(&rag, &mut bases, &make, &settings) {
        Ok(code) => code,
        Err(e) => {
            log(&format!("error: {e}"));
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments() {
        let v: Vec<String> =
            "--config k.toml --data d --env-file e extract --namespace it --model m --api-bases-file b \
                              --parallel 2 --minutes=1.5 --dry-run"
                .split_whitespace()
                .map(String::from)
                .collect();
        let a = parse_args(&v).unwrap();
        assert_eq!(a.command.as_deref(), Some("extract"));
        assert_eq!(
            (a.namespace.as_deref(), a.parallel, a.minutes, a.dry_run),
            (Some("it"), Some(2), Some(1.5), true)
        );
        assert_eq!(a.api_bases_file, Some(PathBuf::from("b")));
        assert!(parse_args(&["--bogus".to_string()]).is_err());
        assert!(parse_args(&["--model".to_string()]).is_err());
    }
}
