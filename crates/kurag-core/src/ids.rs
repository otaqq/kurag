//! Namespaces and chunk ids.
//!
//! Namespaces are dotted names (`dev`, `academic.math`); asking for one includes everything under it. Chunk ids are
//! `<source>:<path>#<section>#<n>` (n: the chunk's number in its section), e.g. `notes:a/b.md#Intro#0`; deletes go
//! by id prefix (a file, a folder, a source), taken literally.

use std::sync::LazyLock;

use regex::Regex;

static NAME: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[a-z][a-z0-9-]*(\.[a-z0-9][a-z0-9-]*)*$").unwrap());

/// Whether `name` is a valid namespace name.
pub fn valid_namespace(name: &str) -> bool {
    NAME.is_match(name)
}

/// `notes:docs/design.md#Setup#2` -> `notes:docs/design.md#Setup`
pub fn section_of(chunk_id: &str) -> &str {
    chunk_id.rsplit_once('#').map_or(chunk_id, |(s, _)| s)
}

/// Ids starting with `prefix` are those `>= lo` and `< hi` (no `hi`: no upper bound), comparing UTF-8 bytes (the
/// same order as code points), so a unique index on the id serves a prefix delete.
pub fn prefix_range(prefix: &str) -> (String, Option<String>) {
    let chars: Vec<char> = prefix.chars().collect();
    for i in (0..chars.len()).rev() {
        let next = match chars[i] as u32 + 1 {
            0xD800 => Some('\u{E000}'), // past the surrogates
            n => char::from_u32(n),
        };
        if let Some(c) = next {
            let hi: String = chars[..i].iter().chain(std::iter::once(&c)).collect();
            return (prefix.to_string(), Some(hi));
        }
    }
    (prefix.to_string(), None)
}

/// Whether `name` is one of `names` or under one of them.
pub fn under(name: &str, names: &[String]) -> bool {
    names.iter().any(|n| name == n || name.strip_prefix(n.as_str()).is_some_and(|rest| rest.starts_with('.')))
}

/// The namespaces to search among `all`: every one, or each asked namespace with everything under it, minus the
/// excluded ones and everything under them.
pub fn expand(all: &[String], asked: Option<&[String]>, exclude: Option<&[String]>) -> Vec<String> {
    all.iter()
        .filter(|a| asked.is_none_or(|asked| asked.is_empty() || under(a, asked)))
        .filter(|a| !exclude.is_some_and(|ex| under(a, ex)))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        assert!(valid_namespace("academic.math"));
        assert!(valid_namespace("ct1a-x.it"));
        assert!(!valid_namespace("Bad Name"));
        assert!(!valid_namespace("a..b"));
        assert!(!valid_namespace("1a"));
    }

    #[test]
    fn prefixes() {
        assert_eq!(prefix_range("a:x"), ("a:x".into(), Some("a:y".into())));
        assert_eq!(prefix_range(""), ("".into(), None));
        assert_eq!(prefix_range("a\u{10FFFF}"), ("a\u{10FFFF}".into(), Some("b".into())));
        assert_eq!(section_of("n:a.md#S#2"), "n:a.md#S");
        assert_eq!(section_of("plain"), "plain");
    }

    #[test]
    fn trees() {
        let all: Vec<String> = ["academic.cs", "academic.math", "academics", "notes"].map(String::from).to_vec();
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(expand(&all, Some(&s(&["academic"])), None), s(&["academic.cs", "academic.math"]));
        assert_eq!(expand(&all, None, Some(&s(&["academic.cs"]))), s(&["academic.math", "academics", "notes"]));
        assert_eq!(expand(&all, Some(&s(&["Bad Name"])), None), Vec::<String>::new());
    }
}
