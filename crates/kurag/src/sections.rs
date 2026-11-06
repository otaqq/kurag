//! Sections, for the graph view: the chunks of one heading or page (`<source>:<path>#<section>`), the links
//! between sections (the closest pair of their chunks) and a short label of keywords for each section.
//!
//! A label is made the first time a section is shown and kept in `section_labels`; writing to the section's chunks
//! or to what the knowledge graph learned from them forgets it. Keywords are the entities the knowledge graph
//! found in the section, filled up with the section's terms (runs of one script: a word, a katakana or kanji
//! compound). Measured on books (docs/search.md, "Section labels"):
//!
//! - Entities by TF-IDF times ln(1 + df), favouring the middle: by frequency alone every page of a Kubernetes book
//!   reads "Kubernetes", by TF-IDF alone the rarest extractions win (file names, version numbers).
//! - Terms by TF-IDF, leaving out terms in fewer than two chunks (mostly text-extraction noise) and terms in more
//!   than 30 % of the chunks of their file (running heads: the book's title or author on every page).

use std::collections::HashMap;

use kurag_core::{ids, text};
use rusqlite::{Connection, OptionalExtension, params, params_from_iter};

use crate::kg::marks;

pub const SCHEMA: &str = "
create table if not exists section_labels(section text primary key, label text not null, source text not null)
  without rowid;
";

/// Keywords in a label.
const KEYWORDS: usize = 3;
/// Terms of a section looked up in the full-text index (the most frequent ones).
const CANDIDATES: usize = 20;
/// A term in more than this share of the chunks of its file is a running head (files of 10 chunks or more).
const RUNNING_HEAD: f64 = 0.3;

/// Forgets the labels of the sections of these chunks (before the chunks are removed).
pub fn forget_vids(db: &Connection, vids: &[i64]) -> rusqlite::Result<()> {
    for part in vids.chunks(500) {
        let sql = format!(
            "delete from section_labels where section in (select section from chunks where vid in ({}))",
            marks(part.len())
        );
        db.execute(&sql, params_from_iter(part))?;
    }
    Ok(())
}

/// Forgets the labels of the sections of the chunks with these ids.
pub fn forget_chunks(db: &Connection, ids: &[&str]) -> rusqlite::Result<()> {
    for part in ids.chunks(500) {
        let sql = format!(
            "delete from section_labels where section in (select section from chunks where id in ({}))",
            marks(part.len())
        );
        db.execute(&sql, params_from_iter(part))?;
    }
    Ok(())
}

/// Forgets the labels of the sections of the chunks in the id ranges.
pub fn forget_ranges(db: &Connection, ranges: &[(String, Option<String>)]) -> rusqlite::Result<()> {
    for (lo, hi) in ranges {
        match hi {
            None => db.execute(
                "delete from section_labels where section in (select section from chunks where id >= ?1)",
                [lo],
            )?,
            Some(hi) => db.execute(
                "delete from section_labels where section in (select section from chunks where id >= ?1 and id < ?2)",
                [lo, hi],
            )?,
        };
    }
    Ok(())
}

pub fn forget_all(db: &Connection) -> rusqlite::Result<()> {
    db.execute("delete from section_labels", [])?;
    Ok(())
}

/// (label, where its keywords came from: "kg", "terms" or "" when none was found), made and kept if new.
pub fn label(db: &Connection, section: &str) -> rusqlite::Result<(String, String)> {
    let kept = db
        .query_row("select label, source from section_labels where section = ?1", [section], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .optional()?;
    if let Some(l) = kept {
        return Ok(l);
    }
    let mut words = kg_keywords(db, section)?;
    let source = if words.is_empty() { "terms" } else { "kg" };
    if words.len() < KEYWORDS {
        for t in term_keywords(db, section, KEYWORDS * 2)? {
            if words.len() == KEYWORDS {
                break;
            }
            if !words.iter().any(|w| overlaps(w, &t)) {
                words.push(t);
            }
        }
    }
    let source = if words.is_empty() { "" } else { source };
    let label = words.join(" · ");
    db.execute(
        "insert or replace into section_labels(section, label, source) values (?1, ?2, ?3)",
        params![section, label, source],
    )?;
    Ok((label, source.to_string()))
}

/// One keyword holds the other (case ignored): 再起動 and 再起動方法, Pod and pods.
fn overlaps(a: &str, b: &str) -> bool {
    let (a, b) = (a.to_lowercase(), b.to_lowercase());
    a.contains(&b) || b.contains(&a)
}

/// TF-IDF: (1 + ln tf) * ln(N / df), with tf the times in the section, df the chunks of the namespace that have it
/// and N all chunks.
fn tf_idf(tf: usize, df: f64, all: f64) -> f64 {
    (1.0 + (tf as f64).ln()) * (all / df.max(1.0)).ln().max(0.0)
}

fn all_chunks(db: &Connection) -> rusqlite::Result<f64> {
    Ok(db.query_row("select count(*) from chunks", [], |r| r.get::<_, i64>(0))?.max(1) as f64)
}

/// The best keywords of the scored ones, none holding another.
fn best(mut scored: Vec<(String, f64)>, keep: usize) -> Vec<String> {
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut out: Vec<String> = Vec::new();
    for (t, s) in scored {
        if out.len() == keep {
            break;
        }
        if s > 0.0 && !out.iter().any(|o| overlaps(o, &t)) {
            out.push(t);
        }
    }
    out
}

/// Entities mentioned in the section, by TF-IDF times ln(1 + df) over the chunks that mention them; only
/// entities mentioned in two chunks or more, with a letter in their name and at most 24 characters.
fn kg_keywords(db: &Connection, section: &str) -> rusqlite::Result<Vec<String>> {
    let all = all_chunks(db)?;
    let mut st = db.prepare_cached(
        "select e.name, count(*), (select count(*) from mentions m2 where m2.eid = e.eid)
         from chunks c join mentions m on m.chunk = c.id join entities e on e.eid = m.eid
         where c.section = ?1 group by e.eid",
    )?;
    let rows = st
        .query_map([section], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?)))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let scored = rows
        .into_iter()
        .filter(|(name, _, df)| *df >= 2 && name.chars().count() <= 24 && name.chars().any(char::is_alphabetic))
        .map(|(name, tf, df)| (name, tf_idf(tf as usize, df as f64, all) * (1.0 + df as f64).ln()))
        .collect();
    Ok(best(scored, KEYWORDS))
}

/// A run worth showing as a keyword: two characters or more (three for Latin words, but AI and OS count), not
/// only digits, not too long.
fn keyword_like(run: &str) -> bool {
    let n = run.chars().count();
    let short_word = run.is_ascii() && n < 3 && !run.chars().all(|c| c.is_ascii_uppercase());
    (2..=16).contains(&n) && !short_word && !run.chars().all(|c| c.is_numeric() || c == '_')
}

/// The section's terms by TF-IDF (df from the full-text index), best first.
fn term_keywords(db: &Connection, section: &str, keep: usize) -> rusqlite::Result<Vec<String>> {
    let texts: Vec<String> = db
        .prepare_cached("select text from chunks where section = ?1")?
        .query_map([section], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    // the same word in any case counts once, shown as first written
    let mut tf: HashMap<String, (String, usize)> = HashMap::new();
    for t in &texts {
        for run in text::runs(t).into_iter().filter(|r| keyword_like(r)) {
            tf.entry(run.to_lowercase()).or_insert_with(|| (run.to_string(), 0)).1 += 1;
        }
    }
    let mut terms: Vec<(String, usize)> = tf.into_values().collect();
    terms.sort_by(|a, b| b.1.cmp(&a.1).then(b.0.chars().count().cmp(&a.0.chars().count())).then(a.0.cmp(&b.0)));
    terms.truncate(CANDIDATES);
    let all = all_chunks(db)?;
    // the chunks of the section's file: `<source>:<path>#...`
    let file = section.split_once('#').map_or(section, |(f, _)| f);
    let (lo, hi) = ids::prefix_range(&format!("{file}#"));
    let hi = hi.unwrap_or_else(|| format!("{lo}\u{10FFFF}"));
    let (in_file, first, last): (i64, i64, i64) = db.query_row(
        "select count(*), coalesce(min(vid), 0), coalesce(max(vid), 0) from chunks where id >= ?1 and id < ?2",
        [&lo, &hi],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    // a file's chunks are written together, so their row ids are mostly one run: counting matches in that run is
    // what the full-text index does fast; when they are spread out, join on the ids instead (slower)
    let in_one_run = last - first + 1 <= 2 * in_file;
    let mut count = db.prepare_cached("select count(*) from fts where fts match ?1")?;
    let mut count_in_file = if in_one_run {
        db.prepare_cached(
            "select count(*) from fts join chunks c on c.vid = fts.rowid
             where fts match ?1 and fts.rowid between ?4 and ?5 and c.id >= ?2 and c.id < ?3",
        )?
    } else {
        db.prepare_cached(
            "select count(*) from fts join chunks c on c.vid = fts.rowid
             where fts match ?1 and c.id >= ?2 and c.id < ?3 and ?4 <= ?5",
        )?
    };
    let mut scored: Vec<(String, f64, String)> = Vec::new();
    for (term, n) in terms {
        let expr = format!("\"{}\"", text::phrase(&term).replace('"', "\"\""));
        let df = count.query_row([&expr], |r| r.get::<_, i64>(0)).unwrap_or(0);
        if df >= 2 {
            scored.push((term, tf_idf(n, df as f64, all), expr));
        }
    }
    scored.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    // best first, so the running-head check (a costlier query) runs only until enough are found
    let mut out: Vec<String> = Vec::new();
    for (term, score, expr) in scored {
        if out.len() == keep {
            break;
        }
        if score <= 0.0 || out.iter().any(|o| overlaps(o, &term)) {
            continue;
        }
        if in_file >= 10 {
            let here = count_in_file.query_row(params![expr, lo, hi, first, last], |r| r.get::<_, i64>(0)).unwrap_or(0);
            if here as f64 > RUNNING_HEAD * in_file as f64 {
                continue;
            }
        }
        out.push(term);
    }
    Ok(out)
}

/// Sections with links to the most other sections: (section, linked sections).
pub fn hubs(db: &Connection, min_weight: f64, limit: usize) -> rusqlite::Result<Vec<(String, i64)>> {
    db.prepare(
        "select ca.section, count(distinct cb.section) n from edges e
         join chunks ca on ca.vid = e.a join chunks cb on cb.vid = e.b
         where e.w >= ?1 and ca.section != cb.section
         group by ca.section order by n desc, ca.section limit ?2",
    )?
    .query_map(params![min_weight, limit as i64], |r| Ok((r.get(0)?, r.get(1)?)))?
    .collect()
}

/// {section: {linked section: weight of the closest pair of their chunks}} for the given sections.
pub fn links(db: &Connection, sections: &[String], min_weight: f64) -> rusqlite::Result<HashMap<String, HashMap<String, f64>>> {
    let mut adj: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for part in sections.chunks(500) {
        let sql = format!(
            "select ca.section, cb.section, max(e.w) from chunks ca
             join edges e on e.a = ca.vid join chunks cb on cb.vid = e.b
             where ca.section in ({}) and e.w >= ? and cb.section != ca.section
             group by ca.section, cb.section",
            marks(part.len())
        );
        let mut args: Vec<rusqlite::types::Value> = part.iter().map(|s| s.clone().into()).collect();
        args.push(min_weight.into());
        let mut st = db.prepare(&sql)?;
        let rows = st.query_map(params_from_iter(args), |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get(2)?)))?;
        for row in rows {
            let (a, b, w) = row?;
            adj.entry(a).or_default().insert(b, w);
        }
    }
    Ok(adj)
}

/// Chunks in each of the sections.
pub fn chunk_counts(db: &Connection, sections: &[String]) -> rusqlite::Result<HashMap<String, i64>> {
    let mut out = HashMap::new();
    for part in sections.chunks(500) {
        let sql = format!("select section, count(*) from chunks where section in ({}) group by section", marks(part.len()));
        let mut st = db.prepare(&sql)?;
        for row in st.query_map(params_from_iter(part), |r| Ok((r.get::<_, String>(0)?, r.get(1)?)))? {
            let (s, n) = row?;
            out.insert(s, n);
        }
    }
    Ok(out)
}

/// The section's source and its text (its chunks in order), None when it has no chunk.
pub fn text(db: &Connection, section: &str) -> rusqlite::Result<Option<(String, String)>> {
    let mut rows: Vec<(String, String, String)> = db
        .prepare_cached("select id, source, text from chunks where section = ?1")?
        .query_map([section], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<rusqlite::Result<_>>()?;
    if rows.is_empty() {
        return Ok(None);
    }
    let n = |id: &str| id.rsplit('#').next().and_then(|n| n.parse::<i64>().ok()).unwrap_or(0);
    rows.sort_by_key(|r| n(&r.0));
    let source = rows[0].1.clone();
    Ok(Some((source, rows.into_iter().map(|r| r.2).collect::<Vec<_>>().join("\n\n"))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keywords_like() {
        assert!(keyword_like("再起動"));
        assert!(keyword_like("kubelet"));
        assert!(!keyword_like("p1"));
        assert!(!keyword_like("2024"));
        assert!(!keyword_like("１１２３"));
        assert!(!keyword_like("Go"));
        assert!(keyword_like("AI"));
        assert!(keyword_like("認証"));
        assert!(!keyword_like("的"));
        assert!(overlaps("再起動", "再起動方法"));
        assert!(overlaps("Pod", "pods"));
        assert!(!overlaps("Pod", "kubelet"));
    }
}
