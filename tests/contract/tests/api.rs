//! The API v1 contract. Exact score values depend on the engine and are not checked; ordering, shapes and what
//! deletes reach are.

use kurag_contract::{Api, BOOK, docs, ids};
use serde_json::{Value, json};

macro_rules! api {
    ($test:literal) => {
        match Api::from_env($test) {
            Some(api) => api,
            None => return,
        }
    };
}

fn map(pairs: &[(&str, i64)]) -> std::collections::BTreeMap<String, i64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

#[tokio::test]
async fn upsert_search_replace_delete() {
    let api = api!("upsert");
    let r = api.upsert("k8s", docs("book:k8s.md#Pod", &BOOK)).await;
    assert_eq!(r, json!({"namespace": api.ns("k8s"), "count": 5}));
    assert_eq!(api.namespaces().await, map(&[("k8s", 5)]));

    let hits = api.search(json!({"query": "PostgreSQL のバックアップ", "limit": 2})).await;
    assert_eq!(hits.len(), 2);
    let h = &hits[0];
    assert_eq!(h["id"], "book:k8s.md#Pod#3");
    assert_eq!(h["namespace"], api.ns("k8s"));
    assert_eq!(h["text"], BOOK[3]);
    assert_eq!(h["source"], "book:k8s.md#Pod");
    let score = h["score"].as_f64().unwrap();
    assert!(0.0 < score && score <= 1.01, "cosine: {score}");
    assert!(hits[1]["score"].as_f64().unwrap() <= score);

    // same id again: replaced, not added
    api.upsert("k8s", json!([{"id": "book:k8s.md#Pod#3", "text": "MySQL のレプリケーション", "source": "book"}])).await;
    assert_eq!(api.namespaces().await, map(&[("k8s", 5)]));
    let hits = api.search(json!({"query": "MySQL のレプリケーション", "limit": 1})).await;
    assert_eq!(hits[0]["text"], "MySQL のレプリケーション");

    let r = api.post("/delete", json!({"namespace": api.ns("k8s"), "prefix": "book:k8s.md#Pod#1"})).await;
    assert_eq!(r["deleted"], 1);
    // an empty namespace goes away
    let r = api.delete("k8s", &["book:k8s.md#"]).await;
    assert_eq!(r, json!({"namespace": api.ns("k8s"), "deleted": 4, "removed": true}));
    assert_eq!(api.namespaces().await, map(&[]));
    assert_eq!(api.search(json!({"query": "Pod"})).await, Vec::<Value>::new());
}

#[tokio::test]
async fn empty_upsert_and_unknown_delete() {
    let api = api!("empty");
    assert_eq!(api.upsert("n", json!([])).await, json!({"namespace": api.ns("n"), "count": 0}));
    assert_eq!(api.namespaces().await, map(&[]));
    assert_eq!(api.delete("n", &["a:"]).await, json!({"namespace": api.ns("n"), "deleted": 0}));
}

#[tokio::test]
async fn prefix_is_literal() {
    let api = api!("prefix");
    api.upsert("n", json!([
        {"id": "a:x_1.md#S#0", "text": "one"}, {"id": "a:xa1.md#S#0", "text": "two"}, {"id": "a:X_1.md#S#0", "text": "three"},
        {"id": "a:x%1.md#S#0", "text": "four"}
    ])).await;
    assert_eq!(api.delete("n", &["a:x_1.md#"]).await["deleted"], 1); // _ and % are not wildcards, case counts
    assert_eq!(api.namespaces().await, map(&[("n", 3)]));
}

#[tokio::test]
async fn namespaces_tree_and_exclude() {
    let api = api!("tree");
    api.upsert("academic.math", docs("m:a#S", &["algebra group ring"])).await;
    api.upsert("academic.cs", docs("c:a#S", &["algebra of programs"])).await;
    api.upsert("notes", docs("n:a#S", &["algebra notes"])).await;
    let found = |hits: Vec<Value>| {
        hits.iter().map(|h| h["namespace"].as_str().unwrap().to_string()).collect::<std::collections::BTreeSet<_>>()
    };
    let set = |names: &[&str]| names.iter().map(|n| api.ns(n)).collect::<std::collections::BTreeSet<_>>();
    let q = |extra: Value| {
        let mut body = json!({"query": "algebra", "limit": 10});
        body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        body
    };
    assert_eq!(found(api.search(q(json!({}))).await), set(&["academic.math", "academic.cs", "notes"]));
    assert_eq!(found(api.search(q(json!({"namespaces": ["academic"]}))).await), set(&["academic.math", "academic.cs"]));
    assert_eq!(found(api.search(q(json!({"exclude": ["academic.cs"]}))).await), set(&["academic.math", "notes"]));
    // a dotted name is a tree, not a string prefix
    api.upsert("academics", docs("x:a#S", &["algebra elsewhere"])).await;
    assert_eq!(found(api.search(q(json!({"namespaces": ["academic"]}))).await), set(&["academic.math", "academic.cs"]));

    // bad names: no hits when searching, 400 when writing
    let bad = api.post("/search", json!({"query": "x", "namespaces": ["Bad Name"]})).await;
    assert_eq!(bad, json!([]));
    let (status, _) = api.post_status("/upsert", json!({"namespace": "Bad Name", "documents": []})).await;
    assert_eq!(status, 400);
}

#[tokio::test]
async fn full_text_japanese_and_hybrid() {
    let api = api!("fulltext");
    api.upsert("it", docs("book:k8s.md#Pod", &BOOK)).await;
    let text = async |q: &str| ids(&api.search(json!({"query": q, "mode": "text", "limit": 3})).await);
    assert_eq!(text("ConfigMap").await[0], "book:k8s.md#Pod#2");
    assert_eq!(text("認証の仕組み").await[0], "book:k8s.md#Pod#4"); // 認証 (two kanji) is found before に
    assert_eq!(text("ネットワーク共有").await[0], "book:k8s.md#Pod#1"); // 共有 in 共有します
    let mut top2 = text("コンテナ").await[..2].to_vec();
    top2.sort();
    assert_eq!(top2, ["book:k8s.md#Pod#0", "book:k8s.md#Pod#1"]);
    assert_eq!(text("とは").await, Vec::<String>::new()); // hiragana only: no term
    assert_eq!(text("configmap").await[0], "book:k8s.md#Pod#2"); // Latin words: any case

    let hits = api.search(json!({"query": "OpenID Connect の認証", "mode": "hybrid", "limit": 2})).await;
    assert_eq!(hits[0]["id"], "book:k8s.md#Pod#4");
    assert!(hits[0]["vector_score"].is_f64() && hits[0]["text_score"].is_f64(), "{}", hits[0]);
}

#[tokio::test]
async fn text_ranks_coverage_across_namespaces() {
    let api = api!("coverage");
    // 設計 is rare in "stats" and PostgreSQL common in "db": BM25 alone would put the stats chunk first
    api.upsert("stats", docs("s:a#S", &["実験の設計 設計 設計"])).await;
    api.upsert("db", docs("d:a#S", &["PostgreSQL のインデックス設計", "PostgreSQL の設定", "PostgreSQL の起動", "PostgreSQL の停止"])).await;
    let q = "PostgreSQL のインデックス設計";
    let hits = api.search(json!({"query": q, "mode": "text", "limit": 3})).await;
    assert_eq!(hits[0]["id"], "d:a#S#0");
    assert!(hits[0]["score"].as_f64().unwrap() >= 1.0, "all terms: coverage 1 + BM25 part: {}", hits[0]["score"]);

    let hy = api.search(json!({"query": q, "mode": "hybrid", "limit": 5})).await;
    assert_eq!(hy[0]["id"], "d:a#S#0");
    assert!(hy[0].get("text_score").is_some());
    assert!(hy.iter().filter(|h| h["id"] == "s:a#S#0").all(|h| h.get("text_score").is_none()), "few terms: not a text hit");
    // a question as a sentence has more words than the chunk; most of them (by length) is enough for hybrid
    let long = api.search(json!({"query": "PostgreSQL のインデックス設計はどう考える方針がよいですか", "mode": "hybrid", "limit": 5})).await;
    assert!(long.iter().any(|h| h["id"] == "d:a#S#0" && h.get("text_score").is_some()));
    assert!(long.iter().filter(|h| h["id"] == "s:a#S#0").all(|h| h.get("text_score").is_none()));
    // text-only hits get a cosine similarity too
    assert!(hy.iter().all(|h| h["vector_score"].is_f64()));
    let exact = api.search(json!({"query": q, "namespaces": ["db"], "limit": 4})).await;
    for h in &hy {
        if let Some(e) = exact.iter().find(|e| e["id"] == h["id"]) {
            let d = (h["vector_score"].as_f64().unwrap() - e["score"].as_f64().unwrap()).abs();
            assert!(d < 0.01, "vector_score of {} differs from its vector search score by {d}", h["id"]);
        }
    }
}

#[tokio::test]
async fn graph_expansion() {
    let api = api!("graph");
    let mut all = docs("book:k8s.md#Pod", &BOOK[..2]).as_array().unwrap().clone();
    all.extend(docs("book:k8s.md#Pods", &[&format!("{} 補足", BOOK[1])]).as_array().unwrap().clone());
    all.extend(docs("book:db.md#DB", &BOOK[3..4]).as_array().unwrap().clone());
    api.upsert("it", json!(all)).await;
    let hits = api.search(json!({"query": BOOK[1], "limit": 1, "graph": 3, "min_weight": 0.3})).await;
    let mut top2 = ids(&hits[..2]);
    top2.sort();
    assert_eq!(top2, ["book:k8s.md#Pod#1", "book:k8s.md#Pods#0"]); // near copies, linked
    assert_eq!(hits[1]["via"], hits[0]["id"]);
    assert!(hits[1]["score"].as_f64() <= hits[0]["score"].as_f64());
    assert!(hits[0].get("via").is_none());
    // no expansion when asked for none
    assert_eq!(api.search(json!({"query": BOOK[1], "limit": 1})).await.len(), 1);
}

#[tokio::test]
async fn many_upserts_delete_and_rebuild() {
    let api = api!("rebuild");
    let many: Vec<String> = (0..300).map(|i| format!("文書 {i} 番目 topic{} {}", i % 7, "x".repeat(i % 13))).collect();
    let many: Vec<&str> = many.iter().map(String::as_str).collect();
    let all = docs("s:f#S", &many);
    for chunk in all.as_array().unwrap().chunks(30) {
        api.upsert("big", json!(chunk)).await; // many small upserts, as sync tools send them
    }
    let top = async |i: usize| api.search(json!({"query": many[i], "namespaces": ["big"], "limit": 1})).await[0]["id"].clone();
    assert_eq!(top(123).await, "s:f#S#123");
    api.delete("big", &["s:f#S#123"]).await;
    assert_ne!(top(123).await, "s:f#S#123");
    assert_eq!(api.namespaces().await, map(&[("big", 299)]));

    let r = api.post("/rebuild", json!({"namespace": api.ns("big")})).await;
    assert!(r.get(api.ns("big")).is_some(), "rebuild answers per namespace: {r}");
    assert_eq!(top(200).await, "s:f#S#200");
    let (status, _) = api.post_status("/rebuild", json!({"namespace": api.ns("missing")})).await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn knowledge_graph() {
    let api = api!("kg");
    let it = api.ns("it");
    api.upsert("it", docs("book:k8s.md#Pod", &BOOK)).await;
    let r = api.post("/kg/upsert", json!({"namespace": it,
        "entities": [{"name": "Kubernetes", "type": "software", "description": "コンテナ管理"}, {"name": "Pod", "type": "concept"}],
        "relations": [{"source": "Kubernetes", "target": "Pod", "type": "runs", "chunk": "book:k8s.md#Pod#0"},
                      {"source": "Pod", "target": "コンテナ", "type": "contains", "chunk": "book:k8s.md#Pod#1"}],
        "mentions": [{"entity": "Pod", "chunk": "book:k8s.md#Pod#0"}, {"entity": "Pod", "chunk": "book:k8s.md#Pod#1"}]})).await;
    assert_eq!(r, json!({"namespace": it, "entities": 3, "relations": 2, "mentions": 2, "done": 0}));

    let kg = api.kg_search(json!({"query": "pod", "depth": 1})).await;
    let first = &kg["entities"][0];
    assert_eq!(first["name"], "Pod");
    assert_eq!(first["namespace"], it);
    assert_eq!(first["chunks"].as_array().unwrap().len(), 2);
    let rels = |kg: &Value| {
        kg["relations"].as_array().unwrap().iter()
            .map(|x| (x["source"].as_str().unwrap().to_string(), x["type"].as_str().unwrap().to_string(), x["target"].as_str().unwrap().to_string()))
            .collect::<std::collections::BTreeSet<_>>()
    };
    let want: std::collections::BTreeSet<_> = [("Kubernetes", "runs", "Pod"), ("Pod", "contains", "コンテナ")]
        .iter().map(|(a, b, c)| (a.to_string(), b.to_string(), c.to_string())).collect();
    assert_eq!(rels(&kg), want);
    let kg = api.kg_search(json!({"query": "コンテナ管理"})).await; // by description
    assert_eq!(kg["entities"][0]["name"], "Kubernetes");

    // deleting the chunks removes what was learned from them, and entities left with nothing
    api.delete("it", &["book:k8s.md#Pod#1"]).await;
    let kg = api.kg_search(json!({"query": "Pod"})).await;
    assert_eq!(rels(&kg).iter().map(|r| r.1.as_str()).collect::<Vec<_>>(), ["runs"]);
    assert!(kg["entities"].as_array().unwrap().iter().all(|e| e["name"] != "コンテナ"));

    // a relation not learned from a chunk stays when every chunk goes, and keeps the namespace
    api.post("/kg/upsert", json!({"namespace": it, "relations": [{"source": "Kubernetes", "target": "CNCF", "type": "project of"}]})).await;
    api.delete("it", &["book:"]).await;
    assert_eq!(api.namespaces().await, map(&[("it", 0)]));
    let kg = api.kg_search(json!({"query": "Kubernetes"})).await;
    assert_eq!(rels(&kg).iter().map(|r| (r.0.as_str(), r.2.as_str())).collect::<Vec<_>>(), [("Kubernetes", "CNCF")]);

    // /kg/delete by prefix
    let r = api.post("/kg/delete", json!({"namespace": it, "prefix": "book:"})).await;
    assert!(r["relations"].is_i64() && r["mentions"].is_i64() && r["entities"].is_i64(), "{r}");
}

#[tokio::test]
async fn kg_todo_and_done() {
    let api = api!("todo");
    let it = api.ns("it");
    api.upsert("it", docs("book:k8s.md#Pod", &BOOK[..3])).await;
    let todo = api.post("/kg/todo", json!({"namespace": it, "limit": 2})).await;
    assert_eq!(ids(todo["chunks"].as_array().unwrap()), ["book:k8s.md#Pod#0", "book:k8s.md#Pod#1"]);
    assert_eq!(todo["remaining"], 3);
    assert_eq!(todo["chunks"][0]["text"], BOOK[0]);

    // extraction of chunk 0: two relations; chunk 1: nothing found, still done
    let r = api.post("/kg/upsert", json!({"namespace": it, "done": ["book:k8s.md#Pod#0", "book:k8s.md#Pod#1"],
        "relations": [{"source": "Kubernetes", "target": "Pod", "type": "runs", "chunk": "book:k8s.md#Pod#0"},
                      {"source": "Pod", "target": "Container", "type": "contains", "chunk": "book:k8s.md#Pod#0"}],
        "mentions": [{"entity": "Pod", "chunk": "book:k8s.md#Pod#0"}]})).await;
    assert_eq!((r["relations"].as_i64(), r["done"].as_i64()), (Some(2), Some(2)));
    let todo = api.post("/kg/todo", json!({"namespace": it})).await;
    assert_eq!(ids(todo["chunks"].as_array().unwrap()), ["book:k8s.md#Pod#2"]);
    assert_eq!(todo["remaining"], 1);

    // chunk 0 sent again (changed): to do again; its new extraction replaces the old relations
    api.upsert("it", json!([{"id": "book:k8s.md#Pod#0", "text": "Pod の説明を改訂", "source": "book"}])).await;
    let mut again = ids(api.post("/kg/todo", json!({"namespace": it})).await["chunks"].as_array().unwrap());
    again.sort();
    assert_eq!(again, ["book:k8s.md#Pod#0", "book:k8s.md#Pod#2"]);
    let r = api.post("/kg/upsert", json!({"namespace": it, "done": ["book:k8s.md#Pod#0"],
        "relations": [{"source": "Pod", "target": "Node", "type": "runs on", "chunk": "book:k8s.md#Pod#0"}]})).await;
    assert_eq!((r["relations"].as_i64(), r["entities"].as_i64()), (Some(1), Some(2))); // Kubernetes and Container went

    // deleting the chunks forgets that they were done
    api.delete("it", &["book:k8s.md#Pod#0"]).await;
    assert_eq!(api.post("/kg/upsert", json!({"namespace": it})).await["done"], 1);

    let (status, _) = api.post_status("/kg/todo", json!({"namespace": api.ns("missing")})).await;
    assert_eq!(status, 404);
}

#[tokio::test]
async fn api_key_and_health() {
    let api = api!("auth");
    assert_eq!(api.get_with_key("/namespaces", None).await.0, 401);
    assert_eq!(api.get_with_key("/namespaces", Some("nope")).await.0, 401);
    let (status, _) = api.post_status("/search", json!({"query": "x"})).await;
    assert_eq!(status, 200);
    let (status, health) = api.get_with_key("/health", None).await; // no key needed
    assert_eq!(status, 200);
    assert_eq!(health["ok"], true);
}
