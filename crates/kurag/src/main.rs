//! kurag: a retrieval server for RAG. Settings come from the environment (see config.rs).

mod config;
mod embed;
mod fts;
mod http;
mod kg;
mod search;
mod store;
mod vectors;

use std::sync::Arc;

#[tokio::main]
async fn main() {
    let cfg = match config::Config::from_env() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("kurag: {e}");
            std::process::exit(2);
        }
    };
    let app = Arc::new(http::App {
        store: store::Store::new(cfg.data_dir.clone()),
        embed: embed::Embedder::new(&cfg),
        cfg,
    });
    let listener = tokio::net::TcpListener::bind(&app.cfg.listen).await.unwrap_or_else(|e| {
        eprintln!("kurag: cannot listen on {}: {e}", app.cfg.listen);
        std::process::exit(2);
    });
    eprintln!("kurag {}: listening on {}, data in {}", env!("CARGO_PKG_VERSION"), app.cfg.listen, app.cfg.data_dir.display());
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
