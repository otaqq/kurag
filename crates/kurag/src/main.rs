//! kurag: a retrieval server for RAG. Settings come from the environment (see config.rs).

mod backend;
mod config;
mod embed;
mod fts;
mod http;
mod import;
mod kg;
mod recall;
mod search;
// without the graph view only the label forgetting is used
#[cfg_attr(not(feature = "ui"), allow(dead_code))]
mod sections;
mod store;
#[cfg(feature = "ui")]
mod ui;
mod vectors;

use std::sync::Arc;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.iter().map(String::as_str).collect::<Vec<_>>()[..] {
        [] => {}
        ["import", src] => {
            let data = std::env::var("DATA_DIR")
                .ok()
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "data".into());
            if let Err(e) = import::run(std::path::Path::new(src), std::path::Path::new(&data)) {
                eprintln!("kurag import: {e}");
                std::process::exit(1);
            }
            return;
        }
        ["recall", questions] => {
            let data = std::env::var("DATA_DIR")
                .ok()
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| "data".into());
            if let Err(e) = recall::run(std::path::Path::new(questions), std::path::Path::new(&data)).await {
                eprintln!("kurag recall: {e}");
                std::process::exit(1);
            }
            return;
        }
        _ => {
            eprintln!("usage: kurag                 run the server (settings from the environment)");
            eprintln!(
                "       kurag import <dir>    copy txtai-kura namespaces (<dir>/<namespace>/kura.db) into DATA_DIR"
            );
            eprintln!(
                "       kurag recall <tsv>    recall@5 of servers A_URL / B_URL against exact search over DATA_DIR"
            );
            std::process::exit(2);
        }
    }
    let cfg = match config::Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("kurag: {e}");
            std::process::exit(2);
        }
    };
    // the external index's HTTP client is a blocking one (used from the blocking pool): made outside the runtime
    let (kind, url) = (cfg.vector_backend.clone(), cfg.vector_url.clone());
    let (gkind, gurl) = (cfg.graph_backend.clone(), cfg.graph_url.clone());
    let (made, graph) = std::thread::spawn(move || {
        (
            backend::from_settings(&kind, url.as_deref()),
            backend::graph_from_settings(&gkind, gurl.as_deref()),
        )
    })
    .join()
    .expect("backend thread");
    let graph: Option<Arc<dyn backend::GraphStore>> = match graph {
        Ok(g) => g.map(Arc::from),
        Err(e) => {
            eprintln!("kurag: {e}");
            std::process::exit(2);
        }
    };
    let graph_name = graph.as_ref().map_or("sqlite", |g| g.name()).to_string();
    let remote = match made {
        Ok(r) => r.map(Arc::from),
        Err(e) => {
            eprintln!("kurag: {e}");
            std::process::exit(2);
        }
    };
    let index = remote
        .as_ref()
        .map_or("memory".to_string(), |r: &Arc<dyn backend::RemoteIndex>| {
            r.name().to_string()
        });
    let app = Arc::new(http::App {
        store: store::Store::new(cfg.data_dir.clone(), remote).with_graph(graph),
        embed: embed::Embedder::new(&cfg),
        cfg,
    });
    let listener = tokio::net::TcpListener::bind(&app.cfg.listen)
        .await
        .unwrap_or_else(|e| {
            eprintln!("kurag: cannot listen on {}: {e}", app.cfg.listen);
            std::process::exit(2);
        });
    eprintln!(
        "kurag {}: listening on {}, data in {}, vector index {index}, graph {graph_name}",
        env!("CARGO_PKG_VERSION"),
        app.cfg.listen,
        app.cfg.data_dir.display()
    );
    axum::serve(listener, http::router(app))
        .with_graceful_shutdown(async {
            // SIGINT, or SIGTERM as sent by Kubernetes and systemd
            let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()).expect("signal");
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
        })
        .await
        .expect("serve");
}
