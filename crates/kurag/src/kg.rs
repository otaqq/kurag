//! The knowledge graph of a namespace, kept in its database: entities, relations between them and the chunks that
//! mention them, sent by clients (for example extracted by an LLM). Relations and mentions remember the chunk they
//! came from, so deleting chunks by prefix also removes what was learned from them; an entity left with no mention
//! and no relation goes too. `kg_done` remembers which chunks a client has extracted, with the chunk's row id (vid):
//! a chunk sent again gets a new vid, so it is to do again, and extracting it again replaces what was learned before.

use std::collections::HashMap;

use kurag_core::types::{KgCounts, KgUpsert, TodoChunk};
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};

use crate::fts;

pub const SCHEMA: &str = "
create table if not exists entities(eid integer primary key, name text unique not null, type text not null default '',
  description text not null default '');
create table if not exists relations(source integer not null, target integer not null, type text not null default '',
  description text not null default '', weight real not null default 1, chunk text not null default '',
  primary key (source, target, type, chunk)) without rowid;
create index if not exists relations_target on relations(target);
create index if not exists relations_chunk on relations(chunk);
create table if not exists mentions(eid integer not null, chunk text not null, primary key (eid, chunk)) without rowid;
create index if not exists mentions_chunk on mentions(chunk);
create table if not exists kg_done(chunk text primary key, vid integer not null) without rowid;
create virtual table if not exists entities_fts using fts5(text, content='', contentless_delete=1,
  tokenize='unicode61 remove_diacritics 2');
";

pub fn marks(n: usize) -> String {
    vec!["?"; n].join(",")
}

/// The entity's id; created if new, its type and description filled in when given.
fn entity_id(db: &Connection, name: &str, kind: &str, description: &str) -> rusqlite::Result<i64> {
    let row = db
        .query_row("select eid, type, description from entities where name = ?1", [name], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
        })
        .optional()?;
    match row {
        None => {
            db.execute("insert into entities(name, type, description) values (?1, ?2, ?3)", params![name, kind, description])?;
            let eid = db.last_insert_rowid();
            fts::add(db, "entities_fts", &[(eid, &format!("{name} {description}"))])?;
            Ok(eid)
        }
        Some((eid, t, d)) => {
            if (!kind.is_empty() && kind != t) || (!description.is_empty() && description != d) {
                let kind = if kind.is_empty() { &t } else { kind };
                let description = if description.is_empty() { &d } else { description };
                db.execute("update entities set type = ?1, description = ?2 where eid = ?3", params![kind, description, eid])?;
                fts::remove(db, "entities_fts", &[eid])?;
                fts::add(db, "entities_fts", &[(eid, &format!("{name} {description}"))])?;
            }
            Ok(eid)
        }
    }
}

/// What an upsert changed, for a copy of the graph elsewhere (`backend::GraphStore`).
#[derive(Default)]
pub struct Change {
    /// chunks whose extraction was replaced
    pub done: Vec<String>,
    /// entities sent (made if new)
    pub entities: Vec<i64>,
    /// (source, target, type, weight, chunk)
    pub relations: Vec<(i64, i64, String, f64, String)>,
    /// (entity, chunk id)
    pub mentions: Vec<(i64, String)>,
    /// entities left with nothing, removed
    pub removed: Vec<i64>,
}

pub fn upsert(db: &Connection, req: &KgUpsert) -> rusqlite::Result<(KgCounts, Change)> {
    let mut change = Change { done: req.done.clone(), ..Default::default() };
    for chunk in &req.done {
        db.execute("delete from relations where chunk = ?1", [chunk])?;
        db.execute("delete from mentions where chunk = ?1", [chunk])?;
    }
    for e in &req.entities {
        change.entities.push(entity_id(db, &e.name, &e.kind, &e.description)?);
    }
    for r in &req.relations {
        let (s, t) = (entity_id(db, &r.source, "", "")?, entity_id(db, &r.target, "", "")?);
        db.execute(
            "insert or replace into relations(source, target, type, description, weight, chunk) values (?1, ?2, ?3, ?4, ?5, ?6)",
            params![s, t, r.kind, r.description, r.weight, r.chunk],
        )?;
        change.entities.extend([s, t]);
        change.relations.push((s, t, r.kind.clone(), r.weight, r.chunk.clone()));
    }
    for m in &req.mentions {
        let eid = entity_id(db, &m.entity, "", "")?;
        db.execute("insert or ignore into mentions(eid, chunk) values (?1, ?2)", params![eid, m.chunk])?;
        change.entities.push(eid);
        change.mentions.push((eid, m.chunk.clone()));
    }
    change.entities.sort_unstable();
    change.entities.dedup();
    for chunk in &req.done {
        let vid: Option<i64> = db.query_row("select vid from chunks where id = ?1", [chunk], |r| r.get(0)).optional()?;
        if let Some(vid) = vid {
            db.execute("insert or replace into kg_done(chunk, vid) values (?1, ?2)", params![chunk, vid])?;
        }
    }
    if !req.done.is_empty() {
        change.removed = remove_orphans(db)?;
    }
    Ok((counts(db)?, change))
}

/// Chunks not extracted yet (or sent again since), oldest first, and how many there are in all.
pub fn todo(db: &Connection, limit: usize) -> rusqlite::Result<(Vec<TodoChunk>, i64)> {
    let cond = "from chunks c left join kg_done d on d.chunk = c.id where d.vid is null or d.vid != c.vid";
    let mut st = db.prepare(&format!("select c.id, c.source, c.text {cond} order by c.vid limit ?1"))?;
    let chunks = st
        .query_map([limit as i64], |r| Ok(TodoChunk { id: r.get(0)?, source: r.get(1)?, text: r.get(2)? }))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let total = db.query_row(&format!("select count(*) {cond}"), [], |r| r.get(0))?;
    Ok((chunks, total))
}

pub fn counts(db: &Connection) -> rusqlite::Result<KgCounts> {
    let n = |t: &str| db.query_row(&format!("select count(*) from {t}"), [], |r| r.get::<_, i64>(0));
    Ok(KgCounts { entities: n("entities")?, relations: n("relations")?, mentions: n("mentions")?, done: n("kg_done")? })
}

/// Removes relations and mentions learned from chunks in the id ranges, then entities left with nothing:
/// (relations, mentions) removed and the ids of the entities removed.
pub fn delete(db: &Connection, ranges: &[(String, Option<String>)]) -> rusqlite::Result<(i64, i64, Vec<i64>)> {
    let (mut relations, mut mentions) = (0, 0);
    for (lo, hi) in ranges {
        let (cond, args): (&str, Vec<&str>) = match hi {
            None => ("chunk >= ?", vec![lo.as_str()]),
            Some(hi) => ("chunk >= ? and chunk < ?", vec![lo.as_str(), hi.as_str()]),
        };
        relations += db.execute(&format!("delete from relations where chunk != '' and {cond}"), params_from_iter(&args))? as i64;
        mentions += db.execute(&format!("delete from mentions where {cond}"), params_from_iter(&args))? as i64;
        db.execute(&format!("delete from kg_done where {cond}"), params_from_iter(&args))?;
    }
    Ok((relations, mentions, remove_orphans(db)?))
}

/// Removes entities with no mention and no relation; their ids.
fn remove_orphans(db: &Connection) -> rusqlite::Result<Vec<i64>> {
    let orphans: Vec<i64> = db
        .prepare(
            "select eid from entities where eid not in (select eid from mentions)
             and eid not in (select source from relations) and eid not in (select target from relations)",
        )?
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for e in &orphans {
        db.execute("delete from entities where eid = ?1", [e])?;
    }
    fts::remove(db, "entities_fts", &orphans)?;
    Ok(orphans)
}

/// Entities named exactly like the query first (case ignored), then by full-text match on name and description.
pub fn find(db: &Connection, query: &str, limit: usize) -> rusqlite::Result<Vec<i64>> {
    let mut found: Vec<i64> = db
        .prepare("select eid from entities where name = ?1 collate nocase")?
        .query_map([query.trim()], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for (eid, _) in fts::search(db, "entities_fts", query, limit)? {
        if !found.contains(&eid) {
            found.push(eid);
        }
    }
    found.truncate(limit);
    Ok(found)
}

pub struct Around {
    /// (name, type, description, chunks mentioning it)
    pub entities: Vec<(String, String, String, Vec<String>)>,
    /// (source, target, type, description, weight, chunks)
    pub relations: Vec<(String, String, String, String, f64, Vec<String>)>,
}

/// One step of a walk over the relations: (source, target, type, highest weight) touching these entities, the
/// heaviest first (then by source, target, type), at most `limit`. Given by the namespace's tables
/// (`relations_around`) or by a graph backend.
pub type Step<'a> = &'a dyn Fn(&[i64], usize) -> Result<Vec<(i64, i64, String, f64)>, String>;

/// `Step` over the namespace's own tables.
pub fn relations_around(db: &Connection, eids: &[i64], limit: usize) -> rusqlite::Result<Vec<(i64, i64, String, f64)>> {
    if eids.is_empty() {
        return Ok(Vec::new());
    }
    let q = marks(eids.len());
    let sql = format!(
        "select source, target, type, max(weight) w from relations where source in ({q}) or target in ({q})
         group by source, target, type order by w desc, source, target, type limit ?"
    );
    let mut args: Vec<i64> = eids.to_vec();
    args.extend(eids);
    args.push(limit as i64);
    db.prepare(&sql)?
        .query_map(params_from_iter(&args), |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect()
}

/// Entities reached from `eids` within `depth` steps (the start ones first), at most `limit`; and the relations
/// walked, in the order they were reached.
pub fn walk(eids: &[i64], depth: usize, limit: usize, step: Step) -> Result<(Vec<i64>, Vec<(i64, i64, String, f64)>), String> {
    let mut seen: Vec<i64> = eids.to_vec();
    let mut frontier: Vec<i64> = eids.to_vec();
    let mut rels: Vec<(i64, i64, String, f64)> = Vec::new();
    let mut index: HashMap<(i64, i64, String), usize> = HashMap::new();
    for _ in 0..depth {
        if frontier.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for (s, t, ty, w) in step(&frontier, limit)? {
            match index.get(&(s, t, ty.clone())) {
                Some(&i) => rels[i].3 = w,
                None => {
                    index.insert((s, t, ty.clone()), rels.len());
                    rels.push((s, t, ty, w));
                }
            }
            for x in [s, t] {
                if !seen.contains(&x) && seen.len() < limit {
                    seen.push(x);
                    next.push(x);
                }
            }
        }
        frontier = next;
    }
    Ok((seen, rels))
}

/// The entities, the relations around them up to `depth` steps, and the chunks that mention each entity. The walk
/// goes through `step`; names, descriptions and chunks come from the namespace's tables.
pub fn around(db: &Connection, eids: &[i64], depth: usize, limit: usize, step: Step) -> Result<Around, String> {
    let (seen, rels) = walk(eids, depth, limit, step)?;
    let fail = |e: rusqlite::Error| e.to_string();
    let mut names: HashMap<i64, String> = HashMap::new();
    let mut entities = Vec::new();
    for eid in seen {
        let row = db
            .query_row("select name, type, description from entities where eid = ?1", [eid], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))
            })
            .optional()
            .map_err(fail)?;
        let Some((name, kind, description)) = row else { continue };
        let chunks: Vec<String> = db
            .prepare_cached("select chunk from mentions where eid = ?1 limit 20")
            .map_err(fail)?
            .query_map([eid], |r| r.get(0))
            .map_err(fail)?
            .collect::<rusqlite::Result<_>>()
            .map_err(fail)?;
        names.insert(eid, name.clone());
        entities.push((name, kind, description, chunks));
    }
    let mut relations = Vec::new();
    for (s, t, ty, w) in rels {
        let (Some(sn), Some(tn)) = (names.get(&s), names.get(&t)) else { continue };
        let (chunks, description): (Option<String>, Option<String>) = db
            .query_row(
                "select group_concat(nullif(chunk, ''), char(31)), max(description) from relations
                 where source = ?1 and target = ?2 and type = ?3",
                params![s, t, ty],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .map_err(fail)?;
        let chunks: Vec<String> = chunks.map(|c| c.split('\u{1f}').map(String::from).collect()).unwrap_or_default();
        relations.push((sn.clone(), tn.clone(), ty, description.unwrap_or_default(), w, chunks));
    }
    Ok(Around { entities, relations })
}

/// `GraphStore::chunks_mentioning` over the namespace's own tables: (chunk row id, how many of the entities).
pub fn chunks_mentioning(db: &Connection, eids: &[i64]) -> rusqlite::Result<Vec<(i64, usize)>> {
    if eids.is_empty() {
        return Ok(Vec::new());
    }
    db.prepare(&format!(
        "select c.vid, count(*) from mentions m join chunks c on c.id = m.chunk where m.eid in ({}) group by c.vid",
        marks(eids.len())
    ))?
    .query_map(params_from_iter(eids), |r| Ok((r.get(0)?, r.get::<_, i64>(1)? as usize)))?
    .collect()
}
