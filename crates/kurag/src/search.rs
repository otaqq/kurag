//! Search over several namespaces: by vector, by full text, or both (hybrid), then along the similarity graph.
//!
//! - vector: hits ranked by cosine similarity; the query is embedded once for all namespaces.
//! - text: hits ranked by text score (`kurag_core::score`).
//! - hybrid: the vector hits and the text hits with most of the query's terms (coverage >= HYBRID_MIN_COVERAGE),
//!   merged by RRF. A hit in both lists comes first; on a tie the vector hit wins. Every hybrid hit carries
//!   vector_score (computed for hits only the text search found), so a client can keep a similarity threshold.
//! - graph: adds up to n chunks linked to the hits; they score hit score x edge weight and carry `via`.

use std::collections::HashMap;
use std::sync::Arc;

use kurag_core::score::{self, HYBRID_MIN_COVERAGE};
use kurag_core::types::{Hit, Mode, Search};

use crate::store::{Namespace, Result};

#[derive(Default)]
struct Merged {
    space: usize,
    vid: i64,
    vector: Option<f64>,
    text: Option<f64>,
    rrf: f64,
}

/// Runs the search over the given namespaces (`q`: the embedded query, for vector and hybrid).
pub fn run(spaces: &[Arc<Namespace>], req: &Search, q: Option<&[f32]>) -> Result<Vec<Hit>> {
    let want = if req.mode == Mode::Hybrid { (req.limit * 2).max(10) } else { req.limit };
    let mut vec_hits: Vec<(f64, usize, i64)> = Vec::new();
    let mut text_hits: Vec<(f64, usize, i64)> = Vec::new();
    for (i, ns) in spaces.iter().enumerate() {
        let inner = ns.lock();
        if let Some(q) = q {
            vec_hits.extend(inner.vector_search(q, want)?.into_iter().map(|(vid, s)| (s, i, vid)));
        }
        if req.mode != Mode::Vector {
            let hits = inner.text_search(&req.query, want)?;
            text_hits.extend(
                hits.into_iter()
                    .filter(|(_, s)| req.mode != Mode::Hybrid || *s >= HYBRID_MIN_COVERAGE) // the score is the coverage (+ < 0.01)
                    .map(|(vid, s)| (s, i, vid)),
            );
        }
    }
    for hits in [&mut vec_hits, &mut text_hits] {
        hits.sort_by(|a, b| b.0.total_cmp(&a.0));
        hits.truncate(want);
    }

    let mut merged: Vec<Merged> = Vec::new();
    let mut at: HashMap<(usize, i64), usize> = HashMap::new();
    for (is_vector, hits) in [(true, &vec_hits), (false, &text_hits)] {
        for (rank, &(s, space, vid)) in hits.iter().enumerate() {
            let k = *at.entry((space, vid)).or_insert_with(|| {
                merged.push(Merged { space, vid, ..Default::default() });
                merged.len() - 1
            });
            let m = &mut merged[k];
            if is_vector { m.vector = Some(s) } else { m.text = Some(s) }
            m.rrf += score::rrf(rank);
        }
    }
    let key = |m: &Merged| match req.mode {
        Mode::Vector => m.vector.unwrap_or(0.0),
        Mode::Text => m.text.unwrap_or(0.0),
        Mode::Hybrid => m.rrf,
    };
    merged.sort_by(|a, b| key(b).total_cmp(&key(a)).then(b.vector.is_some().cmp(&a.vector.is_some())));
    merged.truncate(req.limit);

    // text-only hybrid hits get their cosine similarity too
    if let (Mode::Hybrid, Some(q)) = (req.mode, q) {
        for m in merged.iter_mut().filter(|m| m.vector.is_none()) {
            m.vector = spaces[m.space].lock().similarity(q, m.vid);
        }
    }

    let mut out: Vec<(Hit, usize, i64)> = Vec::new();
    for (i, ns) in spaces.iter().enumerate() {
        let mine: Vec<&Merged> = merged.iter().filter(|m| m.space == i).collect();
        if mine.is_empty() {
            continue;
        }
        let rows = ns.lock().chunks(&mine.iter().map(|m| m.vid).collect::<Vec<_>>())?;
        for m in mine {
            let Some((id, source, text)) = rows.get(&m.vid) else { continue };
            let hybrid = req.mode == Mode::Hybrid;
            out.push((
                Hit {
                    namespace: ns.name.clone(),
                    id: id.clone(),
                    source: source.clone(),
                    text: text.clone(),
                    score: key(m),
                    vector_score: if hybrid { m.vector } else { None },
                    text_score: if hybrid { m.text } else { None },
                    via: None,
                },
                i,
                m.vid,
            ));
        }
    }
    out.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));

    if req.graph > 0 {
        let extra = expand_graph(spaces, &out, req.graph, req.min_weight)?;
        out.extend(extra);
    }
    Ok(out.into_iter().map(|(h, _, _)| h).collect())
}

/// Chunks linked to the hits; a neighbour scores hit score x edge weight.
fn expand_graph(spaces: &[Arc<Namespace>], hits: &[(Hit, usize, i64)], n: usize, min_weight: f64) -> Result<Vec<(Hit, usize, i64)>> {
    let by_vid: HashMap<(usize, i64), &Hit> = hits.iter().map(|(h, s, v)| ((*s, *v), h)).collect();
    // (space, vid) -> (score, via); insertion order kept for ties
    let mut extra: Vec<((usize, i64), f64, String)> = Vec::new();
    let mut at: HashMap<(usize, i64), usize> = HashMap::new();
    let mut used: Vec<usize> = hits.iter().map(|h| h.1).collect();
    used.sort_unstable();
    used.dedup();
    for space in used {
        let vids: Vec<i64> = hits.iter().filter(|h| h.1 == space).map(|h| h.2).collect();
        let near = spaces[space].lock().neighbours(&vids, min_weight)?;
        for (vid, w, via) in near {
            let k = (space, vid);
            if by_vid.contains_key(&k) {
                continue;
            }
            let hit = by_vid[&(space, via)];
            let s = hit.score * w;
            match at.get(&k) {
                Some(&i) if extra[i].1 >= s => {}
                Some(&i) => extra[i] = (k, s, hit.id.clone()),
                None => {
                    at.insert(k, extra.len());
                    extra.push((k, s, hit.id.clone()));
                }
            }
        }
    }
    extra.sort_by(|a, b| b.1.total_cmp(&a.1));
    extra.truncate(n);
    let mut out = Vec::new();
    for ((space, vid), s, via) in extra {
        let ns = &spaces[space];
        let rows = ns.lock().chunks(&[vid])?;
        if let Some((id, source, text)) = rows.get(&vid) {
            out.push((
                Hit {
                    namespace: ns.name.clone(),
                    id: id.clone(),
                    source: source.clone(),
                    text: text.clone(),
                    score: s,
                    vector_score: None,
                    text_score: None,
                    via: Some(via),
                },
                space,
                vid,
            ));
        }
    }
    Ok(out)
}
