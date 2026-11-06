//! Scores.
//!
//! - vector: cosine similarity. Every namespace uses the same embedding model, so scores compare across namespaces.
//! - text: the coverage of the query's terms (0..1, comparable across namespaces; see `text::coverage`) plus
//!   BM25 / 10000 to order hits with the same coverage. BM25 alone does not compare across namespaces (a word common
//!   in one is rare in another).
//! - hybrid: reciprocal rank fusion of the vector hits and the text hits with most of the terms.
//! - graph: a neighbour scores hit score x edge weight.

/// RRF constant: a hit at rank r (0-based) in a list adds 1 / (RRF_K + r + 1).
pub const RRF_K: f64 = 60.0;

/// Share of the query's terms (by length) a text hit needs to join hybrid search. Measured on txtai-kura: 0.5 and
/// 1.0 ranked worse; with 1.0, questions written as sentences got no text hits at all.
pub const HYBRID_MIN_COVERAGE: f64 = 0.6;

/// The text score of a hit.
pub fn text_score(coverage: f64, bm25: f64) -> f64 {
    coverage + bm25.min(99.0) / 10000.0
}

/// What a hit at `rank` (0-based) in one list adds to its RRF score.
pub fn rrf(rank: usize) -> f64 {
    1.0 / (RRF_K + rank as f64 + 1.0)
}
