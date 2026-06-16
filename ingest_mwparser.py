#!/usr/bin/env python3
"""
MediaWiki XML dump -> logically structured Markdown, using ONLY the Wikimedia stack.

  mwxml            -> stream <page>/<revision> out of the dump (memory-safe, namespace-aware)
  mwparserfromhell -> parse tree: extract infobox/template fields as metadata,
                      then render the body to clean, structured Markdown

Output: one .md file per article, with YAML front-matter (title, url, infobox fields)
        followed by a Markdown body whose section hierarchy mirrors the wiki page.

This is a SEPARATE, MANUALLY-TRIGGERED step. It is not part of the app image, the
Docker stack, or build_index.py — you run it by hand only when you need to (re)generate
the Markdown corpus from a fresh wiki dump. Its dependencies (mwxml, mwparserfromhell)
live in the optional `ingest` extra so they never ship with the deployed service.

Requirements (install the optional ingest extra; no pandoc, no other binaries):
  uv sync --extra ingest
  # or:  uv pip install mwparserfromhell mwxml

Usage:
  uv run --extra ingest python ingest_mwparser.py gameofthrones_pages_current.xml.bz2 out_md/

Then build the index from the generated Markdown:
  python build_index.py
"""
import sys, os, re, io, bz2
import mwxml
import mwparserfromhell as mw
from mwparserfromhell.nodes import (
    Text, Heading, Wikilink, ExternalLink, Tag, Template, Comment, HTMLEntity, Argument
)

BASE_URL = "https://gameofthrones.fandom.com/wiki/"

# Wikilink namespaces we never want as prose
SKIP_LINK_PREFIXES = ("file:", "image:", "category:", "media:")


# --------------------------------------------------------------------------- #
# 1. Render a sequence of wikitext nodes -> Markdown string                    #
# --------------------------------------------------------------------------- #
def render(nodes) -> str:
    out = []
    for node in nodes:
        if isinstance(node, Text):
            out.append(str(node))

        elif isinstance(node, Heading):
            title = render(node.title.nodes).strip()
            out.append(f"\n\n{'#' * node.level} {title}\n\n")

        elif isinstance(node, Wikilink):
            target = str(node.title).strip()
            if target.lower().startswith(SKIP_LINK_PREFIXES):
                continue  # drop media/category links entirely
            # keep only the visible text, not the link itself
            out.append(render(node.text.nodes) if node.text else target)

        elif isinstance(node, ExternalLink):
            # keep only the visible text; drop the URL entirely
            if node.title:
                out.append(render(node.title.nodes))

        elif isinstance(node, Tag):
            tag = str(node.tag)
            content = render(node.contents.nodes) if node.contents else ""
            if tag == "b":
                out.append(f"**{content}**")
            elif tag == "i":
                out.append(f"*{content}*")
            elif tag == "li":
                marker = "1. " if node.wiki_markup == "#" else "- "
                out.append(marker)
            elif tag in ("dt", "dd"):
                out.append("")  # definition lists -> flatten to text
            elif tag in ("ref", "references"):
                continue        # drop citation footnotes
            else:
                out.append(content)  # br, span, small, etc. -> keep inner text

        elif isinstance(node, HTMLEntity):
            out.append(node.normalize())

        # Template, Comment, Argument -> intentionally dropped here
    return "".join(out)


# --------------------------------------------------------------------------- #
# 2. Tidy whitespace / list spacing in the rendered Markdown                   #
# --------------------------------------------------------------------------- #
def tidy(md: str) -> str:
    md = re.sub(r"[ \t]+\n", "\n", md)                 # trailing spaces
    md = re.sub(r"^(\s*(?:- |\d+\. ))[ \t]+", r"\1", md, flags=re.M)  # marker spacing
    md = re.sub(r"\n{3,}", "\n\n", md)                 # collapse blank lines
    return md.strip()


# --------------------------------------------------------------------------- #
# 3. Pull infobox / template fields into a metadata dict                       #
# --------------------------------------------------------------------------- #
def extract_metadata(code) -> dict:
    meta = {}
    for tmpl in code.filter_templates():
        if "infobox" in str(tmpl.name).strip().lower():
            for param in tmpl.params:
                key = str(param.name).strip()
                val = mw.parse(str(param.value)).strip_code().strip()
                val = re.sub(r"\s+", " ", val)
                if key and val and not key.isdigit():
                    meta[key] = val
    return meta


def strip_templates(code):
    for tmpl in list(code.filter_templates()):
        try:
            code.remove(tmpl)
        except ValueError:
            pass
    return code


# --------------------------------------------------------------------------- #
# 4. Build one structured Markdown document for a page                         #
# --------------------------------------------------------------------------- #
def page_to_markdown(title: str, wikitext: str) -> str:
    code = mw.parse(wikitext)
    meta = extract_metadata(code)
    strip_templates(code)
    body = tidy(render(code.nodes))

    url = BASE_URL + title.replace(" ", "_")

    # YAML front-matter (logical, machine-readable metadata block)
    fm = ["---", f'title: "{title}"', f"url: {url}"]
    for k, v in meta.items():
        safe = v.replace('"', "'")
        fm.append(f'{k}: "{safe}"')
    fm.append("---")

    # Human-readable infobox summary line (handy for retrieval context)
    summary = ""
    keys = [k for k in ("house", "allegiance", "status", "culture", "born", "died") if k in meta]
    if keys:
        summary = "> " + " · ".join(f"**{k.title()}:** {meta[k]}" for k in keys) + "\n\n"

    return "\n".join(fm) + f"\n\n# {title}\n\n{summary}{body}\n"


# --------------------------------------------------------------------------- #
# 5. Driver: stream the dump, write one .md per article                        #
# --------------------------------------------------------------------------- #
def open_dump(path):
    if path.endswith(".bz2"):
        return io.TextIOWrapper(bz2.open(path, "rb"), encoding="utf-8")  # type: ignore[arg-type]
    return open(path, encoding="utf-8")


def safe_filename(title: str) -> str:
    name = re.sub(r"[^\w\-]+", "_", title).strip("_")[:120]
    return name or "untitled"

def cleanup_markdown(md_dir: str):
    if not os.path.exists(md_dir):
        os.removedirs(md_dir)
    
    os.makedirs(md_dir, exist_ok=True)

def main(dump_path: str, out_dir: str):
    cleanup_markdown(out_dir)
    dump = mwxml.Dump.from_file(open_dump(dump_path))
    n = 0
    for page in dump:
        if page.namespace != 0 or page.redirect:          # main namespace, no redirects
            continue
        wikitext = next((rev.text for rev in page if rev.text), None)
        if not wikitext:
            continue
        try:
            md = page_to_markdown(page.title, wikitext)
        except Exception as e:
            print(f"  skip {page.title!r}: {e}", file=sys.stderr)
            continue
        with open(os.path.join(out_dir, safe_filename(page.title) + ".md"),
                  "w", encoding="utf-8") as f:
            f.write(md)
        n += 1
        if n % 200 == 0:
            print(f"  wrote {n} pages", file=sys.stderr)
    print(f"Done: {n} Markdown files written to {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python ingest_mwparser.py <dump.xml[.bz2]> <out_dir>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
