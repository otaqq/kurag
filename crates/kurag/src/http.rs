//! The HTTP API (version 1, see docs/design.md "API"). Every route but `/health` needs the `X-API-Key` header.
//! Database work runs on the blocking thread pool; embedding calls run before a namespace is locked.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use kurag_core::{ids, types::*};
use serde_json::{Map, Value, json};

use crate::config::Config;
use crate::embed::Embedder;
use crate::store::{self, Namespace, Store};
use crate::{kg, search};

pub struct App {
    pub cfg: Config,
    pub store: Store,
    pub embed: Embedder,
}

type S = State<Arc<App>>;

pub enum ApiError {
    Status(StatusCode, String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let ApiError::Status(code, detail) = self;
        (code, Json(json!({"detail": detail}))).into_response()
    }
}

impl From<store::Error> for ApiError {
    fn from(e: store::Error) -> Self {
        match e {
            store::Error::Bad(m) => ApiError::Status(StatusCode::BAD_REQUEST, m),
            store::Error::Db(e) => ApiError::Status(StatusCode::INTERNAL_SERVER_ERROR, format!("database: {e}")),
            store::Error::Io(e) => ApiError::Status(StatusCode::INTERNAL_SERVER_ERROR, format!("files: {e}")),
        }
    }
}

impl From<rusqlite::Error> for ApiError {
    fn from(e: rusqlite::Error) -> Self {
        store::Error::Db(e).into()
    }
}

type Answer<T> = Result<Json<T>, ApiError>;

fn not_found(what: String) -> ApiError {
    ApiError::Status(StatusCode::NOT_FOUND, what)
}

fn checked<T: Check>(req: T) -> Result<T, ApiError> {
    req.check().map_err(|m| ApiError::Status(StatusCode::UNPROCESSABLE_ENTITY, m))?;
    Ok(req)
}

/// Runs database work off the async threads.
async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, ApiError> + Send + 'static) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ApiError::Status(StatusCode::INTERNAL_SERVER_ERROR, format!("task: {e}")))?
}

fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

async fn auth(State(app): S, req: Request, next: Next) -> Response {
    let key = req.headers().get("x-api-key").map(|v| v.as_bytes()).unwrap_or_default();
    if !same(key, app.cfg.api_key.as_bytes()) {
        return ApiError::Status(StatusCode::UNAUTHORIZED, "bad api key".into()).into_response();
    }
    next.run(req).await
}

pub fn router(app: Arc<App>) -> Router {
    let api = Router::new()
        .route("/namespaces", get(namespaces))
        .route("/upsert", post(upsert))
        .route("/delete", post(delete))
        .route("/search", post(search_route))
        .route("/kg/upsert", post(kg_upsert))
        .route("/kg/todo", post(kg_todo))
        .route("/kg/delete", post(kg_delete))
        .route("/kg/search", post(kg_search))
        .route("/rebuild", post(rebuild))
        .route_layer(middleware::from_fn_with_state(app.clone(), auth));
    Router::new().route("/health", get(health)).merge(api).with_state(app)
}

async fn health(State(app): S) -> Json<Value> {
    Json(json!({"ok": true, "namespaces": app.store.names()}))
}

async fn namespaces(State(app): S) -> Answer<Map<String, Value>> {
    blocking(move || {
        let mut out = Map::new();
        for name in app.store.names() {
            if let Some(ns) = app.store.get(&name, false)? {
                out.insert(name, json!(ns.lock().count()?));
            }
        }
        Ok(Json(out))
    })
    .await
}

async fn upsert(State(app): S, Json(req): Json<Upsert>) -> Answer<Value> {
    Store::check(&req.namespace)?;
    let name = req.namespace.clone();
    if req.documents.is_empty() {
        let app = app.clone();
        return blocking(move || {
            let count = match app.store.get(&name, false)? {
                Some(ns) => ns.lock().count()?,
                None => 0,
            };
            Ok(Json(json!({"namespace": name, "count": count})))
        })
        .await;
    }
    // chunks sent again with the same text keep their vectors: only new or changed text is embedded
    let docs = Arc::new(req.documents);
    let known = {
        let (app, docs, name) = (app.clone(), docs.clone(), name.clone());
        blocking(move || match app.store.get(&name, false)? {
            Some(ns) => Ok(ns.lock().unchanged(&docs)?),
            None => Ok(Default::default()),
        })
        .await?
    };
    let texts: Vec<String> = docs.iter().filter(|d| !known.contains_key(&d.id)).map(|d| d.text.clone()).collect();
    let mut fresh = app
        .embed
        .embed(&texts)
        .await
        .map_err(|e| ApiError::Status(StatusCode::BAD_GATEWAY, e))?
        .into_iter();
    let vectors: Vec<Vec<f32>> =
        docs.iter().map(|d| known.get(&d.id).cloned().unwrap_or_else(|| fresh.next().unwrap_or_default())).collect();
    blocking(move || {
        let ns = app.store.get(&name, true)?.expect("created");
        let count = ns.lock().upsert(&docs, &vectors, app.cfg.edge_neighbours, app.cfg.edge_min)?;
        Ok(Json(json!({"namespace": name, "count": count})))
    })
    .await
}

async fn delete(State(app): S, Json(req): Json<Delete>) -> Answer<Value> {
    blocking(move || {
        let prefixes = req.all_prefixes();
        let name = req.namespace;
        let ns = app.store.get(&name, false)?;
        let (Some(ns), false) = (ns, prefixes.is_empty()) else {
            return Ok(Json(json!({"namespace": name, "deleted": 0})));
        };
        let (deleted, empty) = {
            let mut inner = ns.lock();
            let n = inner.delete(&prefixes)?;
            (n, inner.empty()?)
        };
        if empty {
            // nothing left (e.g. its documents moved to another namespace): drop the namespace itself
            drop(ns);
            app.store.drop_namespace(&name)?;
            return Ok(Json(json!({"namespace": name, "deleted": deleted, "removed": true})));
        }
        Ok(Json(json!({"namespace": name, "deleted": deleted})))
    })
    .await
}

/// The namespaces to search that hold chunks.
fn spaces(app: &App, asked: Option<&[String]>, exclude: Option<&[String]>, with_chunks: bool) -> Result<Vec<Arc<Namespace>>, ApiError> {
    let mut out = Vec::new();
    for name in app.store.expand(asked, exclude) {
        if let Some(ns) = app.store.get(&name, false)? {
            if !with_chunks || ns.lock().count()? > 0 {
                out.push(ns);
            }
        }
    }
    Ok(out)
}

async fn search_route(State(app): S, Json(req): Json<Search>) -> Answer<Vec<Hit>> {
    let req = Arc::new(checked(req)?);
    let found = {
        let (app, req) = (app.clone(), req.clone());
        blocking(move || spaces(&app, req.namespaces.as_deref(), req.exclude.as_deref(), true)).await?
    };
    if found.is_empty() {
        return Ok(Json(Vec::new()));
    }
    let q = if req.mode == Mode::Text {
        None
    } else {
        let mut v = app.embed.embed(std::slice::from_ref(&req.query)).await.map_err(|e| ApiError::Status(StatusCode::BAD_GATEWAY, e))?;
        v.pop()
    };
    blocking(move || Ok(Json(search::run(&found, &req, q.as_deref())?))).await
}

async fn kg_upsert(State(app): S, Json(req): Json<KgUpsert>) -> Answer<Value> {
    blocking(move || {
        let ns = app.store.get(&req.namespace, true)?.expect("created");
        let inner = ns.lock();
        let tx = inner.db.unchecked_transaction()?;
        let c = kg::upsert(&tx, &req)?;
        tx.commit()?;
        Ok(Json(json!({"namespace": req.namespace, "entities": c.entities, "relations": c.relations,
                       "mentions": c.mentions, "done": c.done})))
    })
    .await
}

async fn kg_todo(State(app): S, Json(req): Json<KgTodo>) -> Answer<Value> {
    let req = checked(req)?;
    blocking(move || {
        let ns = app.store.get(&req.namespace, false)?.ok_or_else(|| not_found(format!("no namespace {}", req.namespace)))?;
        let (chunks, remaining) = kg::todo(&ns.lock().db, req.limit)?;
        Ok(Json(json!({"namespace": req.namespace, "chunks": chunks, "remaining": remaining})))
    })
    .await
}

async fn kg_delete(State(app): S, Json(req): Json<Delete>) -> Answer<Value> {
    blocking(move || {
        let prefixes = req.all_prefixes();
        let ns = app.store.get(&req.namespace, false)?;
        let (Some(ns), false) = (ns, prefixes.is_empty()) else {
            return Ok(Json(json!({"namespace": req.namespace, "relations": 0, "mentions": 0, "entities": 0})));
        };
        let inner = ns.lock();
        let tx = inner.db.unchecked_transaction()?;
        let (relations, mentions, entities) = kg::delete(&tx, &prefixes.iter().map(|p| ids::prefix_range(p)).collect::<Vec<_>>())?;
        tx.commit()?;
        Ok(Json(json!({"namespace": req.namespace, "relations": relations, "mentions": mentions, "entities": entities})))
    })
    .await
}

async fn kg_search(State(app): S, Json(req): Json<KgSearch>) -> Answer<Value> {
    let req = checked(req)?;
    blocking(move || {
        let (mut entities, mut relations) = (Vec::new(), Vec::new());
        for ns in spaces(&app, req.namespaces.as_deref(), req.exclude.as_deref(), false)? {
            let inner = ns.lock();
            let found = kg::find(&inner.db, &req.query, req.limit)?;
            if found.is_empty() {
                continue;
            }
            let around = kg::around(&inner.db, &found, req.depth, req.limit * 5)?;
            entities.extend(around.entities.into_iter().map(|(name, kind, description, chunks)| KgEntity {
                namespace: ns.name.clone(),
                name,
                kind,
                description,
                chunks,
            }));
            relations.extend(around.relations.into_iter().map(|(source, target, kind, description, weight, chunks)| {
                KgRelation { namespace: ns.name.clone(), source, target, kind, description, weight, chunks }
            }));
        }
        Ok(Json(json!({"entities": entities, "relations": relations})))
    })
    .await
}

async fn rebuild(State(app): S, Json(req): Json<Rebuild>) -> Answer<Map<String, Value>> {
    blocking(move || {
        let names = match req.namespace {
            Some(n) => vec![n],
            None => app.store.names(),
        };
        let mut out = Map::new();
        for name in names {
            let ns = app.store.get(&name, false)?.ok_or_else(|| not_found(format!("no namespace {name}")))?;
            let n = ns.lock().rebuild()?;
            out.insert(name, json!(n));
        }
        Ok(Json(out))
    })
    .await
}
