//! Terms for full-text search: words, and CJK cut into bigrams.
//!
//! Japanese has no spaces between words, so text is cut into runs by script (Latin letters and digits, katakana,
//! kanji) and every CJK run is indexed as overlapping two-character pieces (再起動 "restart" -> 再起 起動; a
//! one-character run is kept as it is). A query is cut the same way and each run becomes a phrase (再起動 must match
//! 再起 followed by 起動), so a two-character word such as 認証 is found whatever follows it. Hiragana runs (mostly
//! particles and endings: の, とは, する) are left out of both. Case and accents of Latin words are folded by the
//! full-text engine.

use std::sync::LazyLock;

use regex::Regex;

const HIRAGANA: &str = r"\u{3040}-\u{309F}";
const KATAKANA: &str = r"\u{30A0}-\u{30FF}\u{31F0}-\u{31FF}\u{FF66}-\u{FF9F}";
const KANJI: &str = r"\u{3400}-\u{4DBF}\u{4E00}-\u{9FFF}\u{F900}-\u{FAFF}\u{3005}\u{3006}\u{30F5}\u{30F6}";

static RUN: LazyLock<Regex> = LazyLock::new(|| {
    let (h, k, j) = (HIRAGANA, KATAKANA, KANJI);
    Regex::new(&format!(r"[{h}]+|[{k}]+|[{j}]+|[\w&&[^{h}{k}{j}]]+")).unwrap()
});
static ONLY_HIRAGANA: LazyLock<Regex> = LazyLock::new(|| Regex::new(&format!(r"^[{HIRAGANA}]+$")).unwrap());
static ONLY_CJK: LazyLock<Regex> = LazyLock::new(|| Regex::new(&format!(r"^[{KATAKANA}{KANJI}]+$")).unwrap());

/// Runs of one script, in order; hiragana runs left out.
pub fn runs(text: &str) -> Vec<&str> {
    RUN.find_iter(text).map(|m| m.as_str()).filter(|r| !ONLY_HIRAGANA.is_match(r)).collect()
}

/// A run as the full-text index sees it: a CJK run becomes its bigrams, joined by spaces.
pub fn phrase(run: &str) -> String {
    let chars: Vec<char> = run.chars().collect();
    if chars.len() > 1 && ONLY_CJK.is_match(run) {
        chars.windows(2).map(|w| w.iter().collect::<String>()).collect::<Vec<_>>().join(" ")
    } else {
        run.to_string()
    }
}

/// The text as it is indexed.
pub fn tokens(text: &str) -> String {
    runs(text).into_iter().map(phrase).collect::<Vec<_>>().join(" ")
}

/// The query's terms (runs), each once (case ignored).
pub fn terms(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for t in runs(query) {
        let lower = t.to_lowercase();
        if !out.iter().any(|o| o.to_lowercase() == lower) {
            out.push(t.to_string());
        }
    }
    out
}

/// The FTS5 MATCH expression for a query, one phrase per term joined by `op` ("AND" / "OR"); empty when the query
/// has no usable term.
pub fn match_expression(query: &str, op: &str) -> String {
    terms(query)
        .iter()
        .map(|t| format!("\"{}\"", phrase(t).replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(&format!(" {op} "))
}

/// Share of the terms found in the text, each weighing its length (in characters).
pub fn coverage(terms: &[String], text: &str) -> f64 {
    let total: usize = terms.iter().map(|t| t.chars().count()).sum();
    if total == 0 {
        return 0.0;
    }
    let folded = tokens(text).to_lowercase();
    let found: usize = terms.iter().filter(|t| folded.contains(&phrase(t).to_lowercase())).map(|t| t.chars().count()).sum();
    found as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_and_expressions() {
        assert_eq!(terms("Kubernetes の Pod とは"), ["Kubernetes", "Pod"]);
        assert_eq!(terms("設定の再起動方法"), ["設定", "再起動方法"]);
        assert_eq!(match_expression("設定の再起動", "AND"), "\"設定\" AND \"再起 起動\"");
        assert_eq!(terms("a \"quote\" b A"), ["a", "quote", "b"]);
        assert_eq!(match_expression("とは", "OR"), "");
        assert_eq!(match_expression("say \"hi\"", "OR"), "\"say\" OR \"hi\"");
        assert_eq!(tokens("コンテナを動かす"), "コン ンテ テナ 動");
        assert_eq!(tokens("々"), "々");
    }

    #[test]
    fn coverage_by_length() {
        let t = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(coverage(&t(&["PostgreSQL", "設計"]), "PostgreSQL の設計"), 1.0);
        assert!((coverage(&t(&["PostgreSQL", "設計"]), "テーブル設計") - 2.0 / 12.0).abs() < 1e-9);
        assert_eq!(coverage(&[], "x"), 0.0);
    }
}
