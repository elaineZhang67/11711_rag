from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pdfplumber
from bs4 import BeautifulSoup
from pypdf import PdfReader

from compact.io_utils import read_jsonl, write_jsonl
from compact.text_utils import (
    clean_text_basic,
    dedupe_texts_exact,
    normalize_whitespace,
    simple_sentence_split,
    word_count,
)
from rag_hw2.types import Chunk, Document


TEXT_EXTS = {".txt", ".md"}
HTML_EXTS = {".html", ".htm"}
PDF_EXTS = {".pdf"}
SKIP_FILE_NAMES = {"load.php"}

REMOVAL_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "form",
    "button",
    "footer",
    "nav",
    "aside",
}


def _compute_doc_id(source_key: str) -> str:
    return hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]


def _chunk_id(doc_id: str, idx: int, text: str) -> str:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{doc_id}_{idx}_{h}"


def _find_span(text: str, chunk_text: str, start_hint: int = 0) -> tuple[int | None, int | None]:
    pos = text.find(chunk_text, start_hint)
    if pos < 0:
        return None, None
    return pos, pos + len(chunk_text)


def _load_crawl_manifest(raw_dirs: Iterable[str | Path]) -> dict[str, str]:
    url_by_file: dict[str, str] = {}
    for raw_dir in raw_dirs:
        manifest_path = Path(raw_dir) / "manifest.jsonl"
        if not manifest_path.exists():
            continue
        for rec in read_jsonl(manifest_path):
            fp = rec.get("file_path")
            url = rec.get("url")
            if fp and url and rec.get("status") == "ok":
                url_by_file[str(Path(fp).resolve())] = str(url)
    return url_by_file


def iter_source_files(input_dirs: Iterable[str | Path]) -> Iterable[Path]:
    for input_dir in input_dirs:
        root = Path(input_dir)
        if not root.exists():
            continue
        if root.is_file():
            yield root
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part.endswith("_files") for part in p.parts):
                continue
            if p.name in SKIP_FILE_NAMES:
                continue
            if p.suffix.lower() in HTML_EXTS | PDF_EXTS | TEXT_EXTS:
                yield p


def _remove_noise(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(REMOVAL_TAGS):
        tag.decompose()
    for tag in soup.find_all(attrs={"aria-hidden": "true"}):
        tag.decompose()
    for cls in ["reference", "mw-editsection", "navbox", "sidebar", "infobox", "toc", "metadata"]:
        for tag in soup.select(f".{cls}"):
            tag.decompose()


def _best_content_root(soup: BeautifulSoup):
    for sel in ["main", "article", "#mw-content-text", ".mw-parser-output", "#content", ".content"]:
        node = soup.select_one(sel)
        if node is not None:
            return node
    return soup.body or soup


def extract_html_document(path: str | Path, source_url: str | None = None) -> dict:
    p = Path(path)
    raw_html = p.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw_html, "html.parser")
    _remove_noise(soup)
    root = _best_content_root(soup)

    title = None
    if soup.title and soup.title.string:
        title = normalize_whitespace(soup.title.string)
    if not title:
        h1 = root.find(["h1", "h2"]) if root else None
        title = normalize_whitespace(h1.get_text(" ", strip=True)) if h1 else p.stem

    lines: list[str] = []
    for node in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "th", "td"]):
        text = normalize_whitespace(node.get_text(" ", strip=True))
        if text and len(text) > 1:
            lines.append(text)
    text = root.get_text("\n", strip=True) if len(lines) < 5 else "\n".join(lines)
    text = clean_text_basic(text)
    return {
        "title": title,
        "text": text,
        "source_path": str(p),
        "source_url": source_url,
        "doc_type": "html",
        "metadata": {"parser": "bs4"},
    }


def extract_pdf_document(path: str | Path) -> dict:
    p = Path(path)
    parser_used = None
    text = ""
    try:
        parser_used = "pdfplumber"
        pages: list[str] = []
        with pdfplumber.open(str(p)) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text() or "")
        text = "\n\n".join(pages)
    except Exception:
        try:
            parser_used = "pypdf"
            reader = PdfReader(str(p))
            pages = [(page.extract_text() or "") for page in reader.pages]
            text = "\n\n".join(pages)
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF {p}: {e}") from e
    text = clean_text_basic(text)
    return {
        "title": p.stem,
        "text": text,
        "source_path": str(p),
        "source_url": None,
        "doc_type": "pdf",
        "metadata": {"parser": parser_used},
    }


def _extract_doc(path: Path, source_url: str | None = None) -> dict | None:
    ext = path.suffix.lower()
    if ext in HTML_EXTS:
        return extract_html_document(path, source_url=source_url)
    if ext in PDF_EXTS:
        return extract_pdf_document(path)
    if ext in TEXT_EXTS:
        text = clean_text_basic(path.read_text(encoding="utf-8", errors="ignore"))
        return {
            "title": path.stem,
            "text": text,
            "source_path": str(path),
            "source_url": source_url,
            "doc_type": "text",
            "metadata": {},
        }
    return None


def build_corpus(
    input_dirs: list[str | Path],
    output_jsonl: str | Path,
    min_chars: int = 100,
    dedupe_exact: bool = True,
    verbose: bool = False,
) -> list[Document]:
    url_by_file = _load_crawl_manifest(input_dirs)
    docs: list[Document] = []
    for p in iter_source_files(input_dirs):
        src_url = url_by_file.get(str(p.resolve()))
        try:
            rec = _extract_doc(p, source_url=src_url)
        except Exception as e:
            if verbose:
                print(f"[corpus] skip {p}: {e}")
            continue
        if not rec:
            continue
        text = rec["text"].strip()
        if len(text) < min_chars:
            continue
        if verbose:
            print(f"[corpus] {p} -> {len(text)} chars")
        source_key = rec.get("source_url") or rec.get("source_path") or str(p)
        docs.append(
            Document(
                doc_id=_compute_doc_id(str(source_key)),
                text=text,
                title=rec.get("title"),
                source_path=rec.get("source_path"),
                source_url=rec.get("source_url"),
                doc_type=rec.get("doc_type"),
                metadata=rec.get("metadata", {}),
            )
        )
    if dedupe_exact and docs:
        keep = set(dedupe_texts_exact([d.text for d in docs]))
        docs = [d for i, d in enumerate(docs) if i in keep]
    write_jsonl([d.to_dict() for d in docs], output_jsonl)
    return docs


class ChunkingConfig:
    def __init__(
        self,
        strategy: str = "fixed",
        chunk_size_words: int = 250,
        overlap_words: int = 50,
        min_chunk_words: int = 20,
    ) -> None:
        self.strategy = strategy  # fixed | sentence
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.min_chunk_words = min_chunk_words


def _fixed_word_chunks(text: str, cfg: ChunkingConfig) -> list[str]:
    words = text.split()
    if not words:
        return []
    size = max(1, cfg.chunk_size_words)
    overlap = max(0, min(cfg.overlap_words, size - 1))
    step = max(1, size - overlap)
    chunks: list[str] = []
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if not piece:
            continue
        if chunks and len(piece) < cfg.min_chunk_words:
            chunks[-1] = chunks[-1] + " " + " ".join(piece)
            break
        chunks.append(" ".join(piece))
        if start + size >= len(words):
            break
    return chunks


def _sentence_aware_chunks(text: str, cfg: ChunkingConfig) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    units: list[str] = []
    for p in paragraphs:
        if word_count(p) <= cfg.chunk_size_words:
            units.append(p)
        else:
            sents = simple_sentence_split(p)
            units.extend(sents if sents else _fixed_word_chunks(p, cfg))

    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for unit in units:
        uw = word_count(unit)
        if cur and cur_words + uw > cfg.chunk_size_words:
            chunks.append(" ".join(cur).strip())
            if cfg.overlap_words > 0:
                tail_words = " ".join(cur).split()[-cfg.overlap_words :]
                cur = [" ".join(tail_words)] if tail_words else []
                cur_words = len(tail_words)
            else:
                cur = []
                cur_words = 0
        cur.append(unit)
        cur_words += uw
    if cur:
        chunks.append(" ".join(cur).strip())
    if len(chunks) >= 2 and word_count(chunks[-1]) < cfg.min_chunk_words:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks.pop()
    return [c for c in chunks if c]


def chunk_document(doc: Document, cfg: ChunkingConfig) -> list[Chunk]:
    pieces = _sentence_aware_chunks(doc.text, cfg) if cfg.strategy == "sentence" else _fixed_word_chunks(doc.text, cfg)
    out: list[Chunk] = []
    cursor = 0
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        start, end = _find_span(doc.text, piece, cursor)
        if start is not None:
            cursor = start + 1
        out.append(
            Chunk(
                chunk_id=_chunk_id(doc.doc_id, i, piece),
                doc_id=doc.doc_id,
                text=piece,
                title=doc.title,
                source_path=doc.source_path,
                source_url=doc.source_url,
                start_char=start,
                end_char=end,
                chunk_index=i,
                metadata={"doc_type": doc.doc_type, **(doc.metadata or {})},
            )
        )
    return out


def build_chunks(
    documents_jsonl: str | Path,
    output_jsonl: str | Path,
    cfg: ChunkingConfig,
    verbose: bool = False,
) -> list[Chunk]:
    docs = [Document.from_dict(x) for x in read_jsonl(documents_jsonl)]
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(doc, cfg)
        all_chunks.extend(chunks)
        if verbose:
            print(f"[chunk] {doc.doc_id} -> {len(chunks)} chunks")
    write_jsonl([c.to_dict() for c in all_chunks], output_jsonl)
    return all_chunks
