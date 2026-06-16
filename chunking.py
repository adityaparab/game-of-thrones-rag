"""Turn the structured Markdown files (YAML front-matter + heading body) into
retrieval-ready chunks with metadata. Splits on ## / ### sections, with a sliding
window for long sections, and prefixes each chunk with its title+section for context."""
import os
import re
import glob
from typing import List, Dict, Iterator, Optional
import yaml

from config import settings

HEADING = re.compile(r"^(#{1,4})\s+(.*)$", re.M)
_TOKEN = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer used by the BM25 sparse index."""
    return _TOKEN.findall(text.lower())


def parse_md(path: str):
    """Return (metadata_dict, body_markdown)."""
    raw = open(path, encoding="utf-8").read()
    meta, body = {}, raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            try:
                meta = yaml.safe_load(parts[1]) or {}
            except yaml.YAMLError:
                meta = {}
            body = parts[2]
    return meta, body.strip()


def _sections(body: str):
    """Yield (section_path, text). Tracks H2/H3 nesting; H1 is the page title."""
    h2, h3, buf, start = None, None, [], 0
    last_path = "Overview"

    def flush(path, text):
        text = text.strip()
        if text:
            yield path, text

    matches = list(HEADING.finditer(body))
    if not matches:
        yield "Overview", body.strip()
        return

    # lead text before first heading
    lead = body[: matches[0].start()].strip()
    if lead:
        yield "Overview", lead

    for i, m in enumerate(matches):
        level, title = len(m.group(1)), m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[m.end():end].strip()
        if level <= 1:        # page title (#) — skip as a section header
            continue
        if level == 2:
            h2, h3 = title, None
        elif level == 3:
            h3 = title
        path = " > ".join(p for p in (h2, h3) if p) or title
        if text:
            yield path, text


def _window(text: str, max_chars: int, overlap: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    out, paras, buf = [], text.split("\n\n"), ""
    for para in paras:
        if len(buf) + len(para) + 2 > max_chars and buf:
            out.append(buf.strip())
            buf = buf[-overlap:] if overlap else ""   # carry overlap context
        buf += para + "\n\n"
    if buf.strip():
        out.append(buf.strip())
    return out


def chunk_file(path: str) -> List[Dict]:
    meta, body = parse_md(path)
    title = meta.get("title") or os.path.splitext(os.path.basename(path))[0]
    url = meta.get("url", "")
    infobox = {k: v for k, v in meta.items() if k not in ("title", "url")}

    chunks = []
    for section, text in _sections(body):
        for piece in _window(text, settings.chunk_max_chars, settings.chunk_overlap):
            chunks.append({
                "title": title,
                "section": section,
                "url": url,
                "infobox": infobox,
                # context-prefixed text is what gets embedded AND reranked
                "text": f"{title} — {section}\n\n{piece}",
            })
    return chunks


def iter_corpus(md_dir: Optional[str] = None) -> Iterator[Dict]:
    md_dir = md_dir or settings.md_dir
    for path in sorted(glob.glob(os.path.join(md_dir, "*.md"))):
        yield from chunk_file(path)
