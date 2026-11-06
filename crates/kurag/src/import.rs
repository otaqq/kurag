//! `kurag import <dir>`: copies txtai-kura namespaces (`<dir>/<namespace>/kura.db`, v0.3) into the data folder,
//! without calling the embedding API. Chunks keep their row ids, so the similarity graph and the knowledge graph's
//! done marks carry over as they are; the full-text indexes are made again with kurag's tokenizer. A namespace that
//! already exists in the data folder is skipped. Read a copy of a running server's data (SQLite backup), not the
//! live files.

use std::path::Path;

use crate::{fts, store};

const COPY: &[(&str, &str)] = &[
    ("chunks", "insert into chunks(vid, id, section, source, text) select vid, id, section, source, text from src.chunks"),
    ("vectors", "insert into vectors(vid, v) select vid, v from src.vectors"),
    ("edges", "insert into edges(a, b, w) select a, b, w from src.edges"),
    ("entities", "insert into entities(eid, name, type, description) select eid, name, type, description from src.entities"),
    (
        "relations",
        "insert into relations(source, target, type, description, weight, chunk)
         select source, target, type, description, weight, chunk from src.relations",
    ),
    ("mentions", "insert into mentions(eid, chunk) select eid, chunk from src.mentions"),
    ("kg_done", "insert into kg_done(chunk, vid) select chunk, vid from src.kg_done"),
];

pub fn run(src: &Path, data: &Path) -> Result<(), String> {
    // the in-memory index: an external one is filled from the copies when the server opens the namespace
    let st = store::Store::new(data.to_path_buf(), None);
    let have = st.names();
    let mut names: Vec<String> = std::fs::read_dir(src)
        .map_err(|e| format!("{}: {e}", src.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("kura.db").exists())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| kurag_core::ids::valid_namespace(n))
        .collect();
    names.sort();
    for name in names {
        if have.contains(&name) {
            eprintln!("{name}: already there, skipped");
            continue;
        }
        let started = std::time::Instant::now();
        let ns = st.get(&name, true).map_err(|e| format!("{name}: {e:?}"))?.expect("created");
        let result = (|| -> rusqlite::Result<(Vec<(String, i64)>, usize)> {
            let mut inner = ns.lock();
            let db = &inner.db;
            db.execute("attach database ?1 as src", [src.join(&name).join("kura.db").to_string_lossy()])?;
            let tx = db.unchecked_transaction()?;
            let mut counts = Vec::new();
            for (table, sql) in COPY {
                counts.push((table.to_string(), tx.execute(sql, [])? as i64));
            }
            {
                let mut rows = tx.prepare("select vid, text from chunks")?;
                let rows = rows.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?.collect::<rusqlite::Result<Vec<_>>>()?;
                fts::add(&tx, "fts", &rows.iter().map(|(v, t)| (*v, t.as_str())).collect::<Vec<_>>())?;
                let mut ents = tx.prepare("select eid, name || ' ' || description from entities")?;
                let ents = ents.query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?.collect::<rusqlite::Result<Vec<_>>>()?;
                fts::add(&tx, "entities_fts", &ents.iter().map(|(e, t)| (*e, t.as_str())).collect::<Vec<_>>())?;
            }
            tx.commit()?;
            db.execute("detach database src", [])?;
            let n = inner.rebuild().map_err(|e| match e {
                store::Error::Db(e) => e,
                other => rusqlite::Error::InvalidParameterName(format!("{other:?}")),
            })?;
            Ok((counts, n))
        })();
        match result {
            Ok((counts, vectors)) => {
                let list: Vec<String> = counts.iter().map(|(t, n)| format!("{t} {n}")).collect();
                eprintln!("{name}: {}; {vectors} vectors in memory; {:.1} s", list.join(", "), started.elapsed().as_secs_f64());
            }
            Err(e) => {
                drop(ns);
                let _ = st.drop_namespace(&name);
                return Err(format!("{name}: {e} (nothing kept for this namespace)"));
            }
        }
    }
    Ok(())
}
