//! Full text in SQLite FTS5. The tables are contentless (the text is in `chunks` / `entities`): rows are added and
//! removed here, with the terms made by `kurag_core::text`, so CJK is searched the same whatever the engine knows.

use kurag_core::text;
use rusqlite::{Connection, params};

pub const SCHEMA: &str = "
create virtual table if not exists fts using fts5(text, content='', contentless_delete=1,
  tokenize='unicode61 remove_diacritics 2');
";

pub fn add(db: &Connection, table: &str, rows: &[(i64, &str)]) -> rusqlite::Result<()> {
    let mut st = db.prepare_cached(&format!("insert into {table}(rowid, text) values (?1, ?2)"))?;
    for (id, t) in rows {
        st.execute(params![id, text::tokens(t)])?;
    }
    Ok(())
}

pub fn remove(db: &Connection, table: &str, ids: &[i64]) -> rusqlite::Result<()> {
    let mut st = db.prepare_cached(&format!("delete from {table} where rowid = ?1"))?;
    for id in ids {
        st.execute([id])?;
    }
    Ok(())
}

/// (rowid, BM25: higher is better), best first: rows with every term, then rows with some.
pub fn search(db: &Connection, table: &str, query: &str, limit: usize) -> rusqlite::Result<Vec<(i64, f64)>> {
    let mut out: Vec<(i64, f64)> = Vec::new();
    for op in ["AND", "OR"] {
        let expr = text::match_expression(query, op);
        if expr.is_empty() || out.len() >= limit {
            break;
        }
        let mut st = db.prepare_cached(&format!("select rowid, rank from {table} where {table} match ?1 order by rank limit ?2"))?;
        let rows = st.query_map(params![expr, limit as i64], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)))?;
        for row in rows {
            let (id, rank) = row?;
            if out.len() < limit && !out.iter().any(|(i, _)| *i == id) {
                out.push((id, -rank));
            }
        }
    }
    Ok(out)
}
