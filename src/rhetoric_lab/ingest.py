import os
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text as pdfminer_extract_text
from tqdm import tqdm

def _read_txt(path: Path) -> str:
    return path.read_text(errors="ignore")

def _read_html(path: Path) -> str:
    html = path.read_text(errors="ignore")
    # readability to extract main content
    try:
        doc = ReadabilityDocument(html)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "lxml")
        text = soup.get_text(separator=" ")
    except Exception:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator=" ")
    return text

def _read_pdf(path: Path) -> str:
    # Try PyMuPDF first (often cleaner); fallback to pdfminer
    try:
        with fitz.open(path) as doc:
            texts = []
            for page in doc:
                texts.append(page.get_text("text"))
        out = "\n".join(texts)
        if out.strip():
            return out
    except Exception:
        pass
    try:
        return pdfminer_extract_text(str(path))
    except Exception:
        return ""

def _detect_year(parts: List[str]) -> str:
    # attempt to parse a 4-digit year from folder names
    for p in parts[::-1]:
        m = re.search(r"(19|20)\d{2}", p)
        if m:
            return m.group(0)
    return "unknown"

def ingest_corpus(input_root: str, output_root: str) -> None:
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    docs_meta = []
    for path in tqdm(list(input_root.rglob("*")), desc="Scanning input"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in [".txt", ".html", ".htm", ".pdf"]:
            continue

        try:
            if ext == ".txt":
                text = _read_txt(path)
            elif ext in [".html", ".htm"]:
                text = _read_html(path)
            elif ext == ".pdf":
                text = _read_pdf(path)
            else:
                text = ""
        except Exception:
            text = ""

        rel = path.relative_to(input_root)
        parts = rel.parts
        source = parts[0] if len(parts) > 0 else "unknown"
        year = _detect_year(list(parts))
        out_dir = output_root / source / year
        out_dir.mkdir(parents=True, exist_ok=True)
        out_txt = out_dir / (path.stem + ".txt")
        out_meta = out_dir / (path.stem + ".meta.json")

        out_txt.write_text(text, errors="ignore")
        meta = {
            "source": source,
            "year": year,
            "orig_path": str(path),
            "rel_path": str(rel),
            "chars": len(text),
            "tokens": None
        }
        out_meta.write_text(json.dumps(meta, indent=2))
        docs_meta.append(meta)

    (output_root / "index.json").write_text(json.dumps(docs_meta, indent=2))
    print(f"Ingested {len(docs_meta)} documents into {output_root}")
