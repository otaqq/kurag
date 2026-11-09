//! Files into parts, parts into chunks. A part is what the state tracks (a Markdown "## " section, a PDF page, an EPUB
//! chapter, a whole other file); its chunks get ids `<part key>#<n>`. The text is made exactly as txtai-kura-sync
//! makes it, so a part that has not changed has the same sha256 and is not sent again.

use crate::pystr;

pub const CHUNK: usize = 1500; // characters; Japanese is about 1-1.5 characters per token
/// A PDF page or EPUB chapter with fewer letters (spaces not counted) is left out: scanned pages without a text
/// layer give only page numbers or nothing, which would fill the index with empty chunks.
pub const MIN_PAGE_CHARS: usize = 50;

/// Words put into the parts (`[text]` in the sources file). Changing them changes every part.
#[derive(Clone)]
pub struct Words {
    /// the name of the text before a Markdown file's first "## " section
    pub intro: String,
    /// the line saying where a part comes from; `{where}` is the file (and page or chapter)
    pub source: String,
}

impl Default for Words {
    fn default() -> Words {
        Words {
            intro: "(intro)".into(),
            source: "Source: {where}".into(),
        }
    }
}

/// Where a file comes from: "<label>/<path>", or the path alone for a shelf source (its paths are absolute).
pub fn place(label: &str, rel: &str) -> String {
    if label.is_empty() {
        rel.to_string()
    } else {
        format!("{label}/{rel}")
    }
}

fn source_line(w: &Words, label: &str, rel: &str, at: &str) -> String {
    let p = place(label, rel);
    let whole = if at.is_empty() { p } else { format!("{p} {at}") };
    w.source.replace("{where}", &whole)
}

/// A page with its line ends trimmed, or None when it has fewer than MIN_PAGE_CHARS letters (spaces not counted).
pub fn readable(page: &str) -> Option<String> {
    let lines: Vec<&str> = pystr::splitlines(page).into_iter().map(pystr::rstrip).collect();
    let joined = lines.join("\n");
    let page = pystr::strip(&joined);
    (page.chars().filter(|&c| !pystr::space(c)).count() >= MIN_PAGE_CHARS).then(|| page.to_string())
}

/// A Markdown file's "## " sections: [(section name, text)], each starting with the file's title and where it comes
/// from. The text before the first section is named Words.intro. "## " inside code fences does not split. A part is
/// named after its heading, so adding a section does not rename (and re-send) the others; a repeated heading gets
/// " (2)", " (3)" ...
pub fn split_markdown(text: &str, w: &Words, label: &str, rel: &str) -> Vec<(String, String)> {
    let lines = pystr::splitlines(text);
    let title = lines
        .iter()
        .find(|l| l.starts_with("# "))
        .map(|l| pystr::strip(&l[2..]).to_string())
        .unwrap_or_else(|| rel.to_string());
    let intro = w.intro.as_str();
    let mut parts: Vec<(String, Vec<&str>)> = Vec::new();
    let (mut name, mut buf, mut fence) = (intro.to_string(), Vec::new(), false);
    for line in &lines {
        if pystr::lstrip(line).starts_with("```") {
            fence = !fence;
        }
        if !fence && line.starts_with("## ") {
            parts.push((
                std::mem::replace(&mut name, pystr::strip(&line[3..]).to_string()),
                std::mem::take(&mut buf),
            ));
        }
        buf.push(line);
    }
    parts.push((name, buf));
    let mut out = Vec::new();
    let mut used: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for (name, buf) in parts {
        let joined = buf.join("\n");
        let body = pystr::strip(&joined);
        if body.is_empty() || (name == intro && pystr::strip(pystr::lstrip_chars(body, "# ")) == title) {
            continue;
        }
        let count = used.entry(name.clone()).or_insert(0);
        *count += 1;
        let name = if *count > 1 { format!("{name} ({count})") } else { name };
        let src = source_line(w, label, rel, "");
        let text = if name != intro {
            format!("# {title}\n\n{src}\n\n{body}\n")
        } else {
            format!("{src}\n\n{body}\n")
        };
        out.push((name, text));
    }
    out
}

/// A PDF's pages or an EPUB's chapters (form feeds): [("<unit>.<n>", text)], each with the file name, page or
/// chapter and where it comes from. Pages with too little text are left out.
pub fn split_pages(text: &str, w: &Words, label: &str, rel: &str, unit: &str) -> Vec<(String, String)> {
    let title = std::path::Path::new(rel)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut out = Vec::new();
    for (k, page) in text.split('\u{c}').enumerate() {
        if let Some(page) = readable(page) {
            let at = format!("{unit}.{}", k + 1);
            let src = source_line(w, label, rel, &at);
            out.push((at.clone(), format!("# {title}\n\n## {at}\n\n{src}\n\n{page}\n")));
        }
    }
    out
}

/// Whether a Markdown file's YAML front matter says `rag: false`. Only that top-level line is looked at, so no YAML
/// parser is needed. A front matter that is never closed is not one.
fn left_out(text: &str) -> bool {
    let mut lines = text.trim_start_matches('\u{feff}').lines();
    if lines.next().map(str::trim_end) != Some("---") {
        return false;
    }
    let mut found = false;
    for line in lines {
        let line = line.trim_end();
        if line == "---" || line == "..." {
            return found;
        }
        if let Some(value) = line.strip_prefix("rag:") {
            found = value
                .split(" #")
                .next()
                .unwrap_or("")
                .trim()
                .eq_ignore_ascii_case("false");
        }
    }
    false
}

/// {part key: text} of one file: Markdown by section ("<path>#<section>"), PDF by page ("<path>#p.<n>"), EPUB by
/// chapter ("<path>#ch.<n>"), other files whole ("<path>").
pub fn parts_of(rel: &str, text: &str, w: &Words, label: &str) -> std::collections::BTreeMap<String, String> {
    let low = rel.to_lowercase();
    let mut out = std::collections::BTreeMap::new();
    if low.ends_with(".pdf") || low.ends_with(".epub") {
        let unit = if low.ends_with(".pdf") { "p" } else { "ch" };
        for (name, part) in split_pages(text, w, label, rel, unit) {
            out.insert(format!("{rel}#{name}"), part);
        }
    } else if low.ends_with(".md") {
        if left_out(text) {
            return out;
        }
        for (name, part) in split_markdown(text, w, label, rel) {
            out.insert(format!("{rel}#{name}"), part);
        }
    } else {
        out.insert(rel.to_string(), text.to_string());
    }
    out
}

/// A part cut into chunks of at most CHUNK characters at paragraph breaks; each but the first starts with
/// "title › section".
pub fn chunks(text: &str) -> Vec<String> {
    let lines = pystr::splitlines(text);
    let title = match lines.first() {
        Some(l) if l.starts_with("# ") => pystr::strip(pystr::lstrip_chars(l, "# ")).to_string(),
        _ => String::new(),
    };
    let section = lines
        .iter()
        .find(|l| l.starts_with("## "))
        .map(|l| pystr::strip(&l[3..]).to_string())
        .unwrap_or_default();
    let head = [title, section]
        .into_iter()
        .filter(|x| !x.is_empty())
        .collect::<Vec<_>>()
        .join(" › ");
    let mut out: Vec<String> = Vec::new();
    let mut buf = String::new();
    let mut buf_len = 0usize; // characters in buf
    for para in text.split("\n\n") {
        let para_len = pystr::len(para);
        if !buf.is_empty() && buf_len + para_len > CHUNK {
            out.push(std::mem::take(&mut buf));
            buf_len = 0;
        }
        if buf.is_empty() {
            buf.push_str(para);
            buf_len = para_len;
        } else {
            buf.push_str("\n\n");
            buf.push_str(para);
            buf_len += 2 + para_len;
        }
        while buf_len > CHUNK {
            // one very long paragraph (a big table)
            let cut = buf.char_indices().nth(CHUNK).map(|(b, _)| b).unwrap_or(buf.len());
            let rest = buf.split_off(cut);
            out.push(std::mem::replace(&mut buf, rest));
            buf_len -= CHUNK;
        }
    }
    if !pystr::strip(&buf).is_empty() {
        out.push(buf);
    }
    out.into_iter()
        .enumerate()
        .map(|(i, c)| {
            if i == 0 || head.is_empty() {
                c
            } else {
                format!("{head}\n{c}")
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn japanese() -> Words {
        Words {
            intro: "(冒頭)".into(),
            source: "（出典: {where}）".into(),
        }
    }

    #[test]
    fn front_matter_leaves_a_note_out() {
        for md in [
            "---\nrag: false\n---\n# T\n",
            "---\ntitle: x\nrag: False # draft\ntags: [a]\n---\n",
            "\u{feff}---\r\nrag: false\r\n---\r\n",
        ] {
            assert!(left_out(md), "{md:?}");
        }
        for md in [
            "# T\n\nrag: false\n",
            "---\nrag: true\n---\n",
            "---\ntitle: x\n---\nrag: false\n",
            "---\nrag: false\nnever closed\n",
            "---\n  rag: false\n---\n",
            "\n---\nrag: false\n---\n",
        ] {
            assert!(!left_out(md), "{md:?}");
        }
        let w = Words::default();
        assert!(parts_of("a.md", "---\nrag: false\n---\n## A\n\nbody\n", &w, "x").is_empty());
        assert_eq!(
            parts_of("a.md", "---\nrag: true\n---\n\n## A\n\nbody\n", &w, "x").len(),
            2
        );
        assert_eq!(parts_of("a.txt", "---\nrag: false\n---\n", &w, "x").len(), 1);
    }

    #[test]
    fn markdown_sections() {
        let md = "# T\n\nintro\n\n## A\n\na\n```\n## not\n```\n## A\n\nb\n";
        let p = split_markdown(md, &japanese(), "git/x", "d/r.md");
        assert_eq!(
            p.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>(),
            ["(冒頭)", "A", "A (2)"]
        );
        assert_eq!(p[0].1, "（出典: git/x/d/r.md）\n\n# T\n\nintro\n");
        assert_eq!(p[1].1, "# T\n\n（出典: git/x/d/r.md）\n\n## A\n\na\n```\n## not\n```\n");
        // a file that is only its title has no intro part
        assert!(split_markdown("# T\n", &japanese(), "", "a.md").is_empty());
    }

    #[test]
    fn pages_and_chunks() {
        let text = format!("short\u{c}{}", "x".repeat(60));
        let p = split_pages(&text, &Words::default(), "", "mnt/b/Book.pdf", "p");
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].0, "p.2");
        assert!(p[0].1.starts_with("# Book\n\n## p.2\n\nSource: mnt/b/Book.pdf p.2\n\n"));
        let long = format!("# T\n\n## S\n\n{}\n\n{}", "あ".repeat(1000), "い".repeat(3500));
        let c = chunks(&long);
        assert_eq!(c.len(), 4);
        assert!(c[1].starts_with("T › S\n"));
        assert_eq!(pystr::len(&c[2]), CHUNK + "T › S\n".chars().count());
    }
}
