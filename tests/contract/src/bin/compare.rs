//! Compares two servers with the same API and data: the same queries to both, per search mode.
//!
//!   compare <questions.tsv> [modes]      modes default: vector,text,hybrid
//!
//! Servers: A_URL / A_KEY and B_URL / B_KEY. Questions: one per line, `<namespace>\t<query>`. Prints, per mode, how
//! many of A's top 5 B also returns (overlap@5), how often the first hit is the same, and the latency of each
//! server (median, 95th percentile, includes embedding the query). Only numbers and ids are printed, no text.

use std::time::Instant;

use serde_json::{Value, json};

struct Server {
    url: String,
    key: String,
}

impl Server {
    fn from_env(p: &str) -> Server {
        let url = std::env::var(format!("{p}_URL")).unwrap_or_else(|_| panic!("set {p}_URL"));
        Server { url: url.trim_end_matches('/').into(), key: std::env::var(format!("{p}_KEY")).unwrap_or_default() }
    }

    async fn search(&self, http: &reqwest::Client, body: &Value) -> (Vec<String>, f64) {
        let t = Instant::now();
        let r = http.post(format!("{}/search", self.url)).header("X-API-Key", &self.key).json(body).send().await;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let hits: Vec<Value> = match r {
            Ok(r) if r.status().is_success() => r.json().await.unwrap_or_default(),
            Ok(r) => panic!("{}: {}", self.url, r.status()),
            Err(e) => panic!("{}: {e}", self.url),
        };
        (hits.iter().filter_map(|h| h["id"].as_str().map(String::from)).collect(), ms)
    }
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(((v.len() as f64 - 1.0) * p).round() as usize).copied().unwrap_or(0.0)
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(path) = args.first() else {
        eprintln!("usage: compare <questions.tsv> [vector,text,hybrid]   (A_URL A_KEY B_URL B_KEY)");
        std::process::exit(2);
    };
    let modes: Vec<String> = args.get(1).map_or("vector,text,hybrid", |s| s.as_str()).split(',').map(String::from).collect();
    let questions: Vec<(String, String)> = std::fs::read_to_string(path)
        .expect("questions")
        .lines()
        .filter_map(|l| l.split_once('\t').map(|(n, q)| (n.to_string(), q.to_string())))
        .collect();
    let (a, b) = (Server::from_env("A"), Server::from_env("B"));
    let http = reqwest::Client::new();
    println!("{} questions; A = {}, B = {}", questions.len(), a.url, b.url);
    println!("mode     overlap@5  top1-same  A ms (p50 / p95)  B ms (p50 / p95)  differing");
    for mode in &modes {
        let (mut overlap, mut top1, mut n) = (0usize, 0usize, 0usize);
        let (mut ta, mut tb) = (Vec::new(), Vec::new());
        let mut differing = Vec::new();
        for (i, (ns, q)) in questions.iter().enumerate() {
            let body = json!({"query": q, "namespaces": [ns], "mode": mode, "limit": 5});
            // alternate the order, so neither server always gets the cache warmed by the other
            let ((ha, ma), (hb, mb)) = if i % 2 == 0 {
                let x = a.search(&http, &body).await;
                (x, b.search(&http, &body).await)
            } else {
                let y = b.search(&http, &body).await;
                (a.search(&http, &body).await, y)
            };
            ta.push(ma);
            tb.push(mb);
            if ha.is_empty() && hb.is_empty() {
                continue;
            }
            n += 1;
            overlap += ha.iter().filter(|id| hb.contains(id)).count();
            if ha.first() == hb.first() {
                top1 += 1;
            } else {
                differing.push(i + 1);
            }
        }
        let max = (n * 5).max(1);
        println!(
            "{mode:<8} {:>8.1} % {:>8.1} %  {:>7.1} / {:>6.1}  {:>7.1} / {:>6.1}  {:?}",
            100.0 * overlap as f64 / max as f64,
            100.0 * top1 as f64 / n.max(1) as f64,
            pct(&mut ta, 0.5),
            pct(&mut ta, 0.95),
            pct(&mut tb, 0.5),
            pct(&mut tb, 0.95),
            &differing[..differing.len().min(15)]
        );
    }
}
