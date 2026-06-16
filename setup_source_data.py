#!/usr/bin/env python3
"""
Download the Game of Thrones wiki dump, extract it, and generate Markdown files.

Usage:
  uv sync --extra ingest
  uv run --extra ingest python setup_source_data.py
"""
import shutil
import urllib.request
from pathlib import Path

import py7zr

from ingest_mwparser import main as ingest_main

WIKI_7Z_URL = "https://s3.amazonaws.com/wikia_xml_dumps/g/ga/gameofthrones_pages_current.xml.7z"
SOURCE_DIR = Path("source_data")
EXTRACTED_DIR = Path("extracted_data")
WIKI_7Z_PATH = SOURCE_DIR / "wiki.xml.7z"
WIKI_XML_PATH = SOURCE_DIR / "wiki.xml"


def setup_source_data() -> None:
    for folder in (SOURCE_DIR, EXTRACTED_DIR):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    print(f"Downloading {WIKI_7Z_URL} ...")
    with urllib.request.urlopen(WIKI_7Z_URL) as resp, open(WIKI_7Z_PATH, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"Saved to {WIKI_7Z_PATH}")

    print(f"Extracting {WIKI_7Z_PATH} ...")
    with py7zr.SevenZipFile(WIKI_7Z_PATH, mode="r") as archive:
        archive.extractall(path=SOURCE_DIR)

    if not WIKI_XML_PATH.exists():
        xml_files = [p for p in SOURCE_DIR.glob("*.xml") if p != WIKI_XML_PATH]
        if len(xml_files) == 1:
            xml_files[0].rename(WIKI_XML_PATH)
        else:
            raise FileNotFoundError(
                f"Expected {WIKI_XML_PATH} after extraction; found: {xml_files or 'none'}"
            )
    print(f"Extracted to {WIKI_XML_PATH}")

    ingest_main(str(WIKI_XML_PATH), str(EXTRACTED_DIR))


if __name__ == "__main__":
    setup_source_data()
