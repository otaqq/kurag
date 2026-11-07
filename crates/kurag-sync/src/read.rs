//! Reading files into text: Markdown and text as they are, PDF through `pdftotext` (poppler), EPUB chapter by
//! chapter. PDF and EPUB text is cached (`<data>/cache/text/<hash>.json`, the same file txtai-kura-sync writes)
//! until the file's size or time changes. Pages and chapters are separated by form feeds.

use std::io::Read;
use std::path::Path;

use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::extract::log;
use crate::{html, pystr, split};

/// The text of a file (PDF and EPUB: extracted, cached); None when it could not be read (tried again next run).
pub fn read_text(path: &Path, cache_dir: &Path) -> Option<String> {
    let low = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if low.ends_with(".pdf") {
        cached(path, cache_dir, pdf_text)
    } else if low.ends_with(".epub") {
        cached(path, cache_dir, epub_text)
    } else {
        match std::fs::read(path) {
            Ok(b) => Some(pystr::universal_newlines(String::from_utf8_lossy(&b).into_owned())),
            Err(e) => {
                log(&format!("{}: {e}", path.display()));
                None
            }
        }
    }
}

/// The text of a file as run reads it, without the cache (`kurag-sync text`): to see what a file gives, or to
/// compare with another reader.
pub fn fresh_text(path: &Path) -> Option<String> {
    let low = path
        .file_name()
        .map(|n| n.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    if low.ends_with(".pdf") {
        pdf_text(path)
    } else if low.ends_with(".epub") {
        epub_text(path)
    } else {
        std::fs::read(path)
            .ok()
            .map(|b| pystr::universal_newlines(String::from_utf8_lossy(&b).into_owned()))
    }
}

pub fn sha256_hex(s: &str) -> String {
    Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The file's modification time in seconds as a float, the way Python's `os.stat().st_mtime` makes it.
pub fn mtime_float(meta: &std::fs::Metadata) -> f64 {
    use std::os::unix::fs::MetadataExt;
    meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9
}

fn cached(path: &Path, cache_dir: &Path, read: fn(&Path) -> Option<String>) -> Option<String> {
    let meta = std::fs::metadata(path).ok()?;
    let absolute = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let name = sha256_hex(&absolute.to_string_lossy())[..32].to_string() + ".json";
    let cache = cache_dir.join("text").join(name);
    let (size, mtime) = (meta.len(), mtime_float(&meta));
    if let Ok(raw) = std::fs::read_to_string(&cache)
        && let Ok(c) = serde_json::from_str::<Value>(&raw)
        && c["size"].as_u64() == Some(size)
        && c["mtime"].as_f64() == Some(mtime)
        && let Some(t) = c["text"].as_str()
    {
        return Some(t.to_string());
    }
    let text = read(path)?;
    let pages: Vec<&str> = text.split('\u{c}').collect();
    let short = pages.iter().filter(|p| split::readable(p).is_none()).count();
    if short == pages.len() {
        log(&format!(
            "{}: no text (scanned pages need OCR), skipped",
            path.display()
        ));
    } else if short > 0 {
        log(&format!(
            "{}: {short} of {} page(s) without text (fewer than {} letters), skipped",
            path.display(),
            pages.len(),
            split::MIN_PAGE_CHARS
        ));
    }
    if let Some(dir) = cache.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&cache, json!({"size": size, "mtime": mtime, "text": text}).to_string());
    Some(text)
}

static WARNED_PDFTOTEXT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// The text of a PDF by pdftotext, pages separated by form feeds. Without pdftotext, None (one warning).
fn pdf_text(path: &Path) -> Option<String> {
    let out = match std::process::Command::new("pdftotext")
        .args(["-enc", "UTF-8"])
        .arg(path)
        .arg("-")
        .output()
    {
        Ok(o) => o,
        Err(_) => {
            if !WARNED_PDFTOTEXT.swap(true, std::sync::atomic::Ordering::Relaxed) {
                log("pdftotext (poppler) is not installed: PDFs are skipped");
            }
            return None;
        }
    };
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        log(&format!(
            "{}: pdftotext failed: {}",
            path.display(),
            pystr::strip(&err).chars().take(200).collect::<String>()
        ));
        return None;
    }
    Some(pystr::universal_newlines(
        String::from_utf8_lossy(&out.stdout).into_owned(),
    ))
}

/// The text of an EPUB: the spine's items in reading order, one chapter each, separated by form feeds. The package
/// file is read with plain string search, as txtai-kura-sync does, so the same chapters come out.
fn epub_text(path: &Path) -> Option<String> {
    let fail = |e: String| {
        log(&format!("{}: not a readable EPUB: {e}", path.display()));
        None
    };
    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => return fail(e.to_string()),
    };
    let mut z = match zip::ZipArchive::new(file) {
        Ok(z) => z,
        Err(e) => return fail(e.to_string()),
    };
    let mut entry = |name: &str| -> Result<Vec<u8>, String> {
        let mut f = z.by_name(name).map_err(|e| format!("{name}: {e}"))?;
        let mut b = Vec::new();
        f.read_to_end(&mut b).map_err(|e| format!("{name}: {e}"))?;
        Ok(b)
    };
    let container = match entry("META-INF/container.xml") {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) => return fail(e),
    };
    let Some(opf_path) = after(&container, "full-path=\"") else {
        return fail("no full-path in container.xml".into());
    };
    let opf = match entry(&opf_path) {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) => return fail(e),
    };
    let base = match opf_path.rfind('/') {
        Some(k) => opf_path[..k + 1].to_string(),
        None => String::new(),
    };
    let mut items = std::collections::HashMap::new();
    for tag in opf.split("<item ").skip(1) {
        let attr = |k: &str| after(tag, &format!("{k}=\"")).unwrap_or_default();
        items.insert(attr("id"), attr("href"));
    }
    let spine: Vec<String> = opf
        .split("<itemref ")
        .skip(1)
        .filter_map(|t| after(t, "idref=\""))
        .collect();
    let mut chapters = Vec::new();
    for idref in spine {
        let href = url_path(items.get(&idref).map(String::as_str).unwrap_or("")).replace('\\', "/");
        if href.is_empty() {
            continue;
        }
        let raw = match entry(&format!("{base}{href}")) {
            Ok(b) => String::from_utf8_lossy(&b).into_owned(),
            Err(e) => return fail(e),
        };
        let t = html::text(&raw);
        let lines: Vec<&str> = pystr::splitlines(&t)
            .into_iter()
            .map(pystr::strip)
            .filter(|l| !l.is_empty())
            .collect();
        let text = lines.join("\n");
        if !text.is_empty() {
            chapters.push(text);
        }
    }
    Some(chapters.join("\u{c}"))
}

/// What follows the first `key` in s, up to the next double quote (Python: s.split(key, 1)[1].split('"', 1)[0]).
fn after(s: &str, key: &str) -> Option<String> {
    let rest = &s[s.find(key)? + key.len()..];
    Some(rest.split('"').next().unwrap_or("").to_string())
}

/// `urllib.request.url2pathname` on POSIX for a relative href: without ?query and #fragment, %xx decoded.
fn url_path(href: &str) -> String {
    let path = href.split(['#', '?']).next().unwrap_or("");
    let b = path.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() && b[i + 1].is_ascii_hexdigit() && b[i + 2].is_ascii_hexdigit() {
            let hex = std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or("00");
            out.push(u8::from_str_radix(hex, 16).unwrap_or(0));
            i += 3;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hrefs() {
        assert_eq!(url_path("Text/ch%201.xhtml#x"), "Text/ch 1.xhtml");
        assert_eq!(url_path("a%E3%81%82.xhtml"), "aあ.xhtml");
        assert_eq!(url_path("100%"), "100%");
        assert_eq!(
            after("a full-path=\"OEBPS/c.opf\" x", "full-path=\""),
            Some("OEBPS/c.opf".into())
        );
        assert_eq!(
            sha256_hex("abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        // a cached mtime (written by Python's json) reads back as the same float as the file's (float_roundtrip)
        let v: Value = serde_json::from_str("{\"mtime\": 1579017409.4108493}").unwrap();
        assert_eq!(v["mtime"].as_f64(), Some(1579017409f64 + 410849300f64 * 1e-9));
    }
}
