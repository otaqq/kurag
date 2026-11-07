//! Python's string rules, where the chunk text depends on them. Parts are compared with the state by sha256, and the
//! state may come from txtai-kura-sync (Python): a part made a little differently would be sent again. So lines are
//! split, stripped and counted the way Python's str does it.

/// `str.isspace()`: Unicode White_Space plus the separators U+001C..U+001F.
pub fn space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `s.strip()`, `s.lstrip()`, `s.rstrip()`.
pub fn strip(s: &str) -> &str {
    s.trim_matches(space)
}
pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(space)
}
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(space)
}

/// `s.lstrip(chars)`.
pub fn lstrip_chars<'a>(s: &'a str, chars: &str) -> &'a str {
    s.trim_start_matches(|c| chars.contains(c))
}

/// `len(s)`: characters (code points), not bytes.
pub fn len(s: &str) -> usize {
    s.chars().count()
}

/// `s.splitlines()`: \n, \r, \r\n, \v, \f, \x1c, \x1d, \x1e, \x85, U+2028 and U+2029 end a line; no empty last line.
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        match c {
            '\r' => {
                out.push(&s[start..i]);
                if let Some(&(_, '\n')) = it.peek() {
                    it.next();
                    start = i + 2;
                } else {
                    start = i + 1;
                }
            }
            '\n' | '\u{b}' | '\u{c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}' => {
                out.push(&s[start..i]);
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// What Python's text mode reads (universal newlines): \r\n and \r become \n.
pub fn universal_newlines(s: String) -> String {
    if s.contains('\r') { s.replace("\r\n", "\n").replace('\r', "\n") } else { s }
}

/// `fnmatch.fnmatch(name, pattern)` on POSIX: case-sensitive, `*` matches anything (also "/"), `?` one character,
/// `[...]` a set (`[!...]` not in it); a `[` without its `]` is itself.
pub fn fnmatch(name: &str, pattern: &str) -> bool {
    let n: Vec<char> = name.chars().collect();
    let p: Vec<char> = pattern.chars().collect();
    // classic wildcard matching with one backtrack point for the last *
    let (mut i, mut j) = (0usize, 0usize);
    let (mut star, mut mark) = (None::<usize>, 0usize);
    while i < n.len() {
        if j < p.len() {
            match p[j] {
                '*' => {
                    star = Some(j);
                    mark = i;
                    j += 1;
                    continue;
                }
                '?' => {
                    i += 1;
                    j += 1;
                    continue;
                }
                '[' => {
                    if let Some((matched, next)) = set(&p, j, n[i]) {
                        if matched {
                            i += 1;
                            j = next;
                            continue;
                        }
                    } else if n[i] == '[' {
                        i += 1;
                        j += 1;
                        continue;
                    }
                }
                c => {
                    if c == n[i] {
                        i += 1;
                        j += 1;
                        continue;
                    }
                }
            }
        }
        match star {
            Some(s) => {
                j = s + 1;
                mark += 1;
                i = mark;
            }
            None => return false,
        }
    }
    p[j..].iter().all(|&c| c == '*')
}

/// The set starting at p[j] == '[': (whether c is in it, the index after it), or None when it has no closing ].
fn set(p: &[char], j: usize, c: char) -> Option<(bool, usize)> {
    let mut k = j + 1;
    let negate = k < p.len() && p[k] == '!';
    if negate {
        k += 1;
    }
    let first = k;
    let mut hit = false;
    while k < p.len() {
        if p[k] == ']' && k > first {
            return Some((hit != negate, k + 1));
        }
        if k + 2 < p.len() && p[k + 1] == '-' && p[k + 2] != ']' {
            if p[k] <= c && c <= p[k + 2] {
                hit = true;
            }
            k += 3;
        } else {
            if p[k] == c {
                hit = true;
            }
            k += 1;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines_like_python() {
        assert_eq!(splitlines(""), Vec::<&str>::new());
        assert_eq!(splitlines("\n"), vec![""]);
        assert_eq!(splitlines("a\r\nb\rc\x0cd\u{2028}e\n"), vec!["a", "b", "c", "d", "e"]);
        assert_eq!(strip("\u{1f} a\u{3000}"), "a");
        assert_eq!(lstrip_chars("## # x", "# "), "x");
        assert_eq!(len("日本"), 2);
        assert_eq!(universal_newlines("a\r\nb\rc".into()), "a\nb\nc");
    }

    #[test]
    fn globs_like_fnmatch() {
        assert!(fnmatch("a/b.md", "*.md"));
        assert!(fnmatch("x/@eaDir/y", "**/@eaDir/**"));
        assert!(!fnmatch("a.MD", "*.md"));
        assert!(fnmatch(".obsidian/x", ".*/**"));
        assert!(fnmatch("a1", "a[0-9]"));
        assert!(!fnmatch("a1", "a[!0-9]"));
        assert!(fnmatch("#recycle/x", "#recycle/**"));
        assert!(fnmatch("a[", "a["));
        assert!(!fnmatch("ab", "a"));
    }
}
