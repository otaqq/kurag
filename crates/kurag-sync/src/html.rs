//! The text of an EPUB chapter (XHTML), as txtai-kura-sync makes it with Python's html.parser (3.14): text outside
//! `head`, `script` and `style`, a line break at both ends of block elements, character references unescaped.
//! The parser follows html.parser.HTMLParser.goahead() fed once and never closed, so what it leaves unparsed at the
//! end (an unclosed tag, a raw-text element without its end tag) is left out here too.

use crate::entities::{HTML5, INVALID_CHARREFS, INVALID_CODEPOINTS};

const BLOCK: [&str; 14] = [
    "p",
    "div",
    "br",
    "li",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "section",
    "blockquote",
    "pre",
];
const SKIP: [&str; 3] = ["script", "style", "head"];
const RAWTEXT: [&str; 6] = ["script", "style", "xmp", "iframe", "noembed", "noframes"];
const RCDATA: [&str; 2] = ["textarea", "title"];

/// The text handed to handle_data, joined (what txtai-kura-sync's _Text collects in out).
pub fn text(doc: &str) -> String {
    let mut p = Parser {
        s: doc.chars().collect(),
        out: String::new(),
        skip: 0,
        cdata: None,
    };
    p.goahead();
    p.out
}

struct Parser {
    s: Vec<char>,
    out: String,
    skip: i64,
    /// raw-text or escapable raw-text element we are in: (name, escapable)
    cdata: Option<(String, bool)>,
}

fn ws(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | '\u{c}' | ' ')
}

impl Parser {
    fn data(&mut self, d: &str) {
        if self.skip == 0 {
            self.out.push_str(d);
        }
    }

    fn start(&mut self, tag: &str) {
        if SKIP.contains(&tag) {
            self.skip += 1;
        } else if BLOCK.contains(&tag) {
            self.data("\n");
        }
    }

    fn end(&mut self, tag: &str) {
        if SKIP.contains(&tag) {
            self.skip -= 1;
        } else if BLOCK.contains(&tag) {
            self.data("\n");
        }
    }

    fn at(&self, i: usize) -> Option<char> {
        self.s.get(i).copied()
    }

    fn starts(&self, i: usize, lit: &str) -> bool {
        (i..).zip(lit.chars()).all(|(k, c)| self.at(k) == Some(c))
    }

    fn find(&self, c: char, from: usize) -> Option<usize> {
        (from..self.s.len()).find(|&k| self.s[k] == c)
    }

    fn slice(&self, a: usize, b: usize) -> String {
        self.s[a..b].iter().collect()
    }

    fn goahead(&mut self) {
        let n = self.s.len();
        let mut i = 0;
        while i < n {
            let j;
            let escapable;
            match self.cdata.clone() {
                None => {
                    escapable = true;
                    match self.find('<', i) {
                        Some(k) => j = k,
                        None => {
                            // text at the end: kept unless a reference may be cut in half
                            let from = i.max(n.saturating_sub(34));
                            if let Some(amp) = (from..n).rev().find(|&k| self.s[k] == '&')
                                && !(amp..n).any(|k| matches!(self.s[k], '\t' | '\n' | '\r' | '\u{c}' | ' ' | ';'))
                            {
                                break;
                            }
                            j = n;
                        }
                    }
                }
                Some((elem, esc)) => {
                    escapable = esc;
                    if elem == "plaintext" {
                        j = n;
                    } else {
                        match self.find_end_of(&elem, i) {
                            Some(k) => j = k,
                            None => break,
                        }
                    }
                }
            }
            if i < j {
                let raw = self.slice(i, j);
                let d = if escapable { unescape(&raw) } else { raw };
                self.data(&d);
            }
            i = j;
            if i == n {
                break;
            }
            // here s[i] == '<'
            let k: Option<usize> = if self.at(i + 1).is_some_and(|c| c.is_ascii_alphabetic()) {
                self.parse_starttag(i)
            } else if self.starts(i, "</") {
                self.parse_endtag(i)
            } else if self.starts(i, "<!--") {
                self.parse_comment(i)
            } else if self.starts(i, "<?") {
                self.find('>', i + 2).map(|g| g + 1)
            } else if self.starts(i, "<!") {
                self.parse_html_declaration(i)
            } else if i + 1 < n {
                self.data("<");
                Some(i + 1)
            } else {
                break;
            };
            match k {
                Some(k) => i = k,
                None => break,
            }
        }
    }

    /// `</elem` followed by whitespace, "/" or ">" (ASCII case-insensitive).
    fn find_end_of(&self, elem: &str, from: usize) -> Option<usize> {
        let e: Vec<char> = elem.chars().collect();
        let n = self.s.len();
        (from..n).find(|&p| {
            self.s[p] == '<'
                && self.at(p + 1) == Some('/')
                && p + 2 + e.len() < n
                && e.iter()
                    .enumerate()
                    .all(|(x, c)| self.s[p + 2 + x].eq_ignore_ascii_case(c))
                && matches!(self.s[p + 2 + e.len()], '\t' | '\n' | '\r' | '\u{c}' | ' ' | '/' | '>')
        })
    }

    /// The `locatetagend` regex from p (a letter): the tag name and its attributes, and a ">" if there is one.
    fn locatetagend(&self, p: usize) -> usize {
        let n = self.s.len();
        let mut q = p + 1;
        while q < n && !matches!(self.s[q], '\t' | '\n' | '\r' | '\u{c}' | ' ' | '/' | '>') {
            q += 1;
        }
        while q < n && matches!(self.s[q], '\t' | '\n' | '\r' | '\u{c}' | ' ' | '/') {
            q += 1;
        }
        loop {
            if q >= n || !self.after_attr_gap(q) || matches!(self.s[q], '\t' | '\n' | '\r' | '\u{c}' | ' ' | '/' | '>')
            {
                break;
            }
            q = self.attr_name_end(q);
            q = self.attr_value_end(q).unwrap_or(q);
            while q < n && matches!(self.s[q], '\t' | '\n' | '\r' | '\u{c}' | ' ' | '/') {
                q += 1;
            }
        }
        if q < n && self.s[q] == '>' {
            q += 1;
        }
        q
    }

    /// The lookbehind `(?<=['"\t\n\r\f /])` before an attribute name.
    fn after_attr_gap(&self, q: usize) -> bool {
        q > 0 && matches!(self.s[q - 1], '\'' | '"' | '\t' | '\n' | '\r' | '\u{c}' | ' ' | '/')
    }

    /// From the first character of an attribute name (any but whitespace, "/" and ">"), to its end.
    fn attr_name_end(&self, q: usize) -> usize {
        let mut q = q + 1;
        while q < self.s.len() && !matches!(self.s[q], '\t' | '\n' | '\r' | '\u{c}' | ' ' | '/' | '=' | '>') {
            q += 1;
        }
        q
    }

    /// `[ws]*=[ws]*('...'|"..."|bare)` from q, or None when there is no value (or its quote is not closed).
    fn attr_value_end(&self, q: usize) -> Option<usize> {
        let n = self.s.len();
        let mut r = q;
        while r < n && ws(self.s[r]) {
            r += 1;
        }
        if r >= n || self.s[r] != '=' {
            return None;
        }
        r += 1;
        while r < n && ws(self.s[r]) {
            r += 1;
        }
        match self.at(r) {
            Some(quote @ ('\'' | '"')) => self.find(quote, r + 1).map(|e| e + 1),
            _ => {
                while r < n && !matches!(self.s[r], '>' | '\t' | '\n' | '\r' | '\u{c}' | ' ') {
                    r += 1;
                }
                Some(r)
            }
        }
    }

    /// `tagfind_tolerant` from p: (lower-case tag name, the index after it and the gap after it).
    fn tagfind(&self, p: usize) -> (String, usize) {
        let n = self.s.len();
        let mut q = p + 1;
        while q < n && !matches!(self.s[q], '\t' | '\n' | '\r' | '\u{c}' | ' ' | '/' | '>') {
            q += 1;
        }
        let name = self.slice(p, q).to_lowercase();
        q = self.gap(q);
        (name, q)
    }

    /// `(?:[\t\n\r\f ]|/(?!>))*`
    fn gap(&self, mut q: usize) -> usize {
        let n = self.s.len();
        while q < n && (ws(self.s[q]) || (self.s[q] == '/' && self.at(q + 1) != Some('>'))) {
            q += 1;
        }
        q
    }

    fn parse_starttag(&mut self, i: usize) -> Option<usize> {
        let endpos = self.locatetagend(i + 1);
        if self.s[endpos - 1] != '>' {
            return None;
        }
        let (tag, mut k) = self.tagfind(i + 1);
        // attributes (attrfind_tolerant), only to see where they end
        while k < endpos {
            if !self.after_attr_gap(k) || matches!(self.s[k], '\t' | '\n' | '\r' | '\u{c}' | ' ' | '/' | '>') {
                break;
            }
            let mut q = self.attr_name_end(k);
            q = self.attr_value_end(q).unwrap_or(q);
            k = self.gap(q);
        }
        let rest = self.slice(k, endpos);
        let end = crate::pystr::strip(&rest);
        if end != ">" && end != "/>" {
            let raw = self.slice(i, endpos);
            self.data(&raw);
            return Some(endpos);
        }
        if end.ends_with("/>") {
            self.start(&tag);
            self.end(&tag);
        } else {
            self.start(&tag);
            if RAWTEXT.contains(&tag.as_str()) || tag == "plaintext" {
                self.cdata = Some((tag, false));
            } else if RCDATA.contains(&tag.as_str()) {
                self.cdata = Some((tag, true));
            }
        }
        Some(endpos)
    }

    fn parse_endtag(&mut self, i: usize) -> Option<usize> {
        self.find('>', i + 2)?;
        if !self.at(i + 2).is_some_and(|c| c.is_ascii_alphabetic()) {
            if self.at(i + 2) == Some('>') {
                return Some(i + 3);
            }
            return self.find('>', i + 2).map(|g| g + 1); // bogus comment
        }
        let j = self.locatetagend(i + 2);
        if self.s[j - 1] != '>' {
            return None;
        }
        let (tag, _) = self.tagfind(i + 2);
        self.end(&tag);
        self.cdata = None;
        Some(j)
    }

    fn parse_comment(&mut self, i: usize) -> Option<usize> {
        // an empty comment ends at the first ">" or "->"
        if self.at(i + 4) == Some('>') {
            return Some(i + 5);
        }
        if self.starts(i + 4, "->") {
            return Some(i + 6);
        }
        let n = self.s.len();
        (i + 4..n).find_map(|p| {
            if self.starts(p, "-->") {
                Some(p + 3)
            } else if self.starts(p, "--!>") {
                Some(p + 4)
            } else {
                None
            }
        })
    }

    fn parse_html_declaration(&mut self, i: usize) -> Option<usize> {
        if self.starts(i, "<![CDATA[") {
            let n = self.s.len();
            return (i + 9..n).find(|&p| self.starts(p, "]]>")).map(|p| p + 3);
        }
        if self.slice(i, (i + 9).min(self.s.len())).to_lowercase() == "<!doctype" {
            return self.find('>', i + 9).map(|g| g + 1);
        }
        self.find('>', i + 2).map(|g| g + 1) // bogus comment
    }
}

/// Python's `html.unescape`.
pub fn unescape(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let c: Vec<char> = s.chars().collect();
    let n = c.len();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < n {
        if c[i] != '&' {
            out.push(c[i]);
            i += 1;
            continue;
        }
        // &(#[0-9]+;?|#[xX][0-9a-fA-F]+;?|[^\t\n\f <&#;]{1,32};?)
        if i + 1 < n && c[i + 1] == '#' {
            let hex = i + 2 < n && (c[i + 2] == 'x' || c[i + 2] == 'X');
            let from = if hex { i + 3 } else { i + 2 };
            let mut k = from;
            while k < n
                && (if hex {
                    c[k].is_ascii_hexdigit()
                } else {
                    c[k].is_ascii_digit()
                })
            {
                k += 1;
            }
            if k > from {
                let digits: String = c[from..k].iter().collect();
                if k < n && c[k] == ';' {
                    k += 1;
                }
                out.push_str(&numeric(&digits, if hex { 16 } else { 10 }));
                i = k;
                continue;
            }
            out.push('&');
            i += 1;
            continue;
        }
        let mut k = i + 1;
        while k < n && k - (i + 1) < 32 && !matches!(c[k], '\t' | '\n' | '\u{c}' | ' ' | '<' | '&' | '#' | ';') {
            k += 1;
        }
        if k == i + 1 {
            out.push('&');
            i += 1;
            continue;
        }
        if k < n && c[k] == ';' {
            k += 1;
        }
        let name: Vec<char> = c[i + 1..k].to_vec();
        out.push_str(&named(&name));
        i = k;
    }
    out
}

fn numeric(digits: &str, radix: u32) -> String {
    let num = u32::from_str_radix(digits, radix).unwrap_or(u32::MAX); // too big: past U+10FFFF either way
    if let Ok(x) = INVALID_CHARREFS.binary_search_by_key(&num, |&(k, _)| k) {
        return INVALID_CHARREFS[x].1.to_string();
    }
    if (0xD800..=0xDFFF).contains(&num) || num > 0x10FFFF {
        return "\u{FFFD}".to_string();
    }
    if INVALID_CODEPOINTS.binary_search(&num).is_ok() {
        return String::new();
    }
    char::from_u32(num)
        .map(String::from)
        .unwrap_or_else(|| "\u{FFFD}".to_string())
}

fn lookup(name: &str) -> Option<&'static str> {
    HTML5.binary_search_by(|&(k, _)| k.cmp(name)).ok().map(|x| HTML5[x].1)
}

fn named(name: &[char]) -> String {
    let s: String = name.iter().collect();
    if let Some(v) = lookup(&s) {
        return v.to_string();
    }
    // the longest name at its start (at least 2 characters), the rest as it is
    for x in (2..name.len()).rev() {
        let head: String = name[..x].iter().collect();
        if let Some(v) = lookup(&head) {
            let tail: String = name[x..].iter().collect();
            return format!("{v}{tail}");
        }
    }
    format!("&{s}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unescape_like_python() {
        assert_eq!(
            unescape("a &amp; b &lt;c&gt; &#x41;&#66; &#150; &copy &notit; &#0; &nosuch;"),
            "a & b <c> AB \u{2013} \u{a9} \u{ac}it; \u{FFFD} &nosuch;"
        );
        assert_eq!(unescape("&#xD800;&#1114112;"), "\u{FFFD}\u{FFFD}");
        assert_eq!(unescape("& &# &x"), "& &# &x");
    }

    #[test]
    fn chapter_text() {
        let doc = "<?xml version=\"1.0\"?><!DOCTYPE html><html><head><title>T</title><style>p{}</style></head>\
                   <body><h1>Title</h1><p class=\"a\">One &amp; two<br/>three</p><!-- c --><script>if (a<b) x()</script>\
                   <div>x &lt; y</div>tail</body></html>";
        assert_eq!(text(doc), "\nTitle\n\nOne & two\n\nthree\n\nx < y\ntail");
        // an unclosed tag at the end is left out, as the parser waits for more
        assert_eq!(text("<p>a</p><span class=\"x"), "\na\n");
        // a broken start tag is text
        assert_eq!(text("a<b c=\"1\"d>e"), "ae");
    }
}
