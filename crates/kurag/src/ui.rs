//! The graph view (`ui/`, built into the binary with the feature "ui"): the sections of a namespace and the
//! similarity links between them, drawn around one section at a time, each labelled with its keywords
//! (`sections.rs`). Served under `/ui` behind a Basic login (`UI_USER`, `UI_PASSWORD`); there is no `/ui` without
//! a password. The routes are those of txtai-kura's graph view.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, Query, Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use base64::Engine;
use kurag_core::ids;
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::http::{ApiError, App, blocking, namespace_counts, same};
use crate::sections;
use crate::store::Namespace;

type S = State<Arc<App>>;

const INDEX: &str = include_str!("../../../ui/index.html");
const APP_JS: &str = include_str!("../../../ui/app.js");
const STYLE: &str = include_str!("../../../ui/style.css");
const CYTOSCAPE: &str = include_str!("../../../ui/vendor/cytoscape.min.js");

/// The `/ui` routes, None when no password is set.
pub fn router(app: Arc<App>) -> Option<Router<Arc<App>>> {
    app.cfg.ui_password.as_ref()?;
    Some(
        Router::new()
            .route("/ui", get(|| file(INDEX, "text/html; charset=utf-8")))
            .route("/ui/app.js", get(|| file(APP_JS, "text/javascript; charset=utf-8")))
            .route("/ui/style.css", get(|| file(STYLE, "text/css; charset=utf-8")))
            .route(
                "/ui/vendor/cytoscape.min.js",
                get(|| file(CYTOSCAPE, "text/javascript; charset=utf-8")),
            )
            .route("/ui/namespaces", get(namespaces))
            .route("/ui/search/{namespace}", get(search))
            .route("/ui/hubs/{namespace}", get(hubs))
            .route("/ui/around/{namespace}", get(around))
            .route("/ui/section/{namespace}", get(section))
            .route_layer(middleware::from_fn_with_state(app, login)),
    )
}

async fn file(body: &'static str, kind: &'static str) -> Response {
    ([(header::CONTENT_TYPE, kind)], body).into_response()
}

async fn login(State(app): S, req: Request, next: Next) -> Response {
    let password = app.cfg.ui_password.as_deref().unwrap_or_default();
    let want = format!(
        "Basic {}",
        base64::engine::general_purpose::STANDARD.encode(format!("{}:{password}", app.cfg.ui_user))
    );
    let got = req
        .headers()
        .get(header::AUTHORIZATION)
        .map(|v| v.as_bytes())
        .unwrap_or_default();
    if password.is_empty() || !same(got, want.as_bytes()) {
        return (
            StatusCode::UNAUTHORIZED,
            [(header::WWW_AUTHENTICATE, "Basic realm=\"kurag\"")],
            "login required",
        )
            .into_response();
    }
    next.run(req).await
}

fn open(app: &App, name: &str) -> Result<Arc<Namespace>, ApiError> {
    app.store
        .get(name, false)?
        .ok_or_else(|| ApiError::Status(StatusCode::NOT_FOUND, format!("no namespace {name}")))
}

/// A section as the view shows it: its keywords as label (else its heading or page), its file, its heading or page.
fn describe(db: &Connection, section: &str) -> Result<Value, ApiError> {
    let (source, rest) = section.split_once(':').unwrap_or(("", section));
    let (file, name) = rest.split_once('#').unwrap_or((rest, ""));
    let (keywords, from) = sections::label(db, section)?;
    let label = if !keywords.is_empty() {
        keywords
    } else if !name.is_empty() {
        name.to_string()
    } else {
        file.to_string()
    };
    Ok(json!({"id": section, "label": label, "keywords_from": from, "name": name, "file": format!("{source}:{file}")}))
}

async fn namespaces(State(app): S) -> Result<Json<serde_json::Map<String, Value>>, ApiError> {
    blocking(move || namespace_counts(&app).map(Json)).await
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    #[serde(default = "twenty")]
    limit: usize,
}

fn twenty() -> usize {
    20
}

/// The sections of the namespace whose chunks are closest to the text q, best first.
async fn search(
    State(app): S,
    Path(name): Path<String>,
    Query(p): Query<SearchQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let limit = p.limit.clamp(1, 100);
    let ns = {
        let (app, name) = (app.clone(), name.clone());
        blocking(move || open(&app, &name)).await?
    };
    let q = app
        .embed
        .embed(std::slice::from_ref(&p.q))
        .await
        .map_err(|e| ApiError::Status(StatusCode::BAD_GATEWAY, e))?
        .pop()
        .unwrap_or_default();
    blocking(move || {
        let inner = ns.lock();
        let hits = inner.vector_search(&q, limit * 3)?;
        let rows = inner.chunks(&hits.iter().map(|h| h.0).collect::<Vec<_>>())?;
        // by score: the first chunk of a section is its best
        let mut best: Vec<(String, f64)> = Vec::new();
        for (vid, score) in hits {
            if let Some((id, _, _)) = rows.get(&vid) {
                let sec = ids::section_of(id);
                if !best.iter().any(|(s, _)| s == sec) {
                    best.push((sec.to_string(), score));
                }
            }
        }
        best.truncate(limit);
        let mut out = Vec::new();
        for (sec, score) in best {
            let mut d = describe(&inner.db, &sec)?;
            d["score"] = json!((score * 1000.0).round() / 1000.0);
            out.push(d);
        }
        Ok(Json(out))
    })
    .await
}

#[derive(Deserialize)]
struct HubsQuery {
    #[serde(default = "strong")]
    min_weight: f64,
    #[serde(default = "twenty")]
    limit: usize,
}

fn strong() -> f64 {
    0.75
}

/// The sections with the most links at least min_weight similar: places to start without a search.
async fn hubs(
    State(app): S,
    Path(name): Path<String>,
    Query(p): Query<HubsQuery>,
) -> Result<Json<Vec<Value>>, ApiError> {
    blocking(move || {
        let ns = open(&app, &name)?;
        let inner = ns.lock();
        let mut out = Vec::new();
        for (sec, n) in sections::hubs(&inner.db, p.min_weight, p.limit.clamp(1, 100))? {
            let mut d = describe(&inner.db, &sec)?;
            d["links"] = json!(n);
            out.push(d);
        }
        Ok(Json(out))
    })
    .await
}

#[derive(Deserialize)]
struct AroundQuery {
    key: String,
    #[serde(default = "one")]
    depth: usize,
    #[serde(default = "strong")]
    min_weight: f64,
    #[serde(default = "sixty")]
    limit: usize,
}

fn one() -> usize {
    1
}

fn sixty() -> usize {
    60
}

/// The section `key` and the sections linked to it at least `min_weight` similar, up to `depth` steps away and
/// `limit` more sections (the strongest links first), with the edges between them. A whole big namespace
/// (thousands of sections) is too many to lay out in a browser and too dense to read.
async fn around(
    State(app): S,
    Path(name): Path<String>,
    Query(p): Query<AroundQuery>,
) -> Result<Json<Value>, ApiError> {
    let (depth, limit) = (p.depth.clamp(1, 3), p.limit.clamp(1, 300));
    blocking(move || {
        let ns = open(&app, &name)?;
        let inner = ns.lock();
        let db = &inner.db;
        if sections::chunk_counts(db, std::slice::from_ref(&p.key))?.is_empty() {
            return Err(ApiError::Status(StatusCode::NOT_FOUND, format!("no section {}", p.key)));
        }
        // breadth first, the strongest links first; `order` keeps the sections in the order they were reached
        let mut steps: HashMap<String, usize> = HashMap::from([(p.key.clone(), 0)]);
        let mut order = vec![p.key.clone()];
        let mut frontier = vec![p.key.clone()];
        for step in 1..=depth {
            let adj = sections::links(db, &frontier, p.min_weight)?;
            let mut found: HashMap<String, f64> = HashMap::new();
            for s in &frontier {
                for (t, &w) in adj.get(s).into_iter().flatten() {
                    if !steps.contains_key(t) && w > found.get(t).copied().unwrap_or(0.0) {
                        found.insert(t.clone(), w);
                    }
                }
            }
            let mut next: Vec<(String, f64)> = found.into_iter().collect();
            next.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            next.truncate((limit + 1).saturating_sub(steps.len()));
            frontier = next.into_iter().map(|(s, _)| s).collect();
            for s in &frontier {
                steps.insert(s.clone(), step);
                order.push(s.clone());
            }
            if frontier.is_empty() {
                break;
            }
        }
        let adj = sections::links(db, &order, p.min_weight)?;
        let counts = sections::chunk_counts(db, &order)?;
        let mut edges = Vec::new();
        for a in &order {
            for (b, w) in adj.get(a).into_iter().flatten() {
                if a < b && steps.contains_key(b) {
                    edges.push(json!({"source": a, "target": b, "weight": (w * 1000.0).round() / 1000.0}));
                }
            }
        }
        let mut nodes = Vec::new();
        for s in &order {
            let mut d = describe(db, s)?;
            d["chunks"] = json!(counts.get(s).copied().unwrap_or(0));
            d["step"] = json!(steps[s]);
            nodes.push(d);
        }
        Ok(Json(json!({"key": p.key, "nodes": nodes, "edges": edges})))
    })
    .await
}

#[derive(Deserialize)]
struct SectionQuery {
    key: String,
}

/// The text of a section: its chunks in order.
async fn section(
    State(app): S,
    Path(name): Path<String>,
    Query(p): Query<SectionQuery>,
) -> Result<Json<Value>, ApiError> {
    blocking(move || {
        let ns = open(&app, &name)?;
        let inner = ns.lock();
        let (source, text) = sections::text(&inner.db, &p.key)?.unwrap_or_default();
        Ok(Json(json!({"key": p.key, "source": source, "text": text})))
    })
    .await
}
