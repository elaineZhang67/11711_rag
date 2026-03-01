from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable

from rag_hw2.preprocess.html_extract import extract_html_document
from rag_hw2.preprocess.pdf_extract import extract_pdf_document
from rag_hw2.types import Document


TEXT_EXTS = {".txt", ".md"}
HTML_EXTS = {".html", ".htm"}
PDF_EXTS = {".pdf"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
SKIP_FILE_NAMES = {"load.php"}


def _read_jsonl(path) :
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(records, path) :
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _normalize_whitespace(text) :
    return re.sub(r"\s+", " ", text).strip()


def _clean_text_basic(text) :
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _dedupe_texts_exact(texts) :
    seen = set()
    keep = []
    for i, t in enumerate(texts):
        key = _normalize_whitespace(t.lower())
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep


def _compute_doc_id(source_key) :
    return hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]


def _compile_patterns(values) :
    out = []
    for v in values or []:
        try:
            out.append(re.compile(str(v), re.IGNORECASE))
        except Exception:
            continue
    return out


def _match_any(text, patterns) :
    t = str(text or "")
    for p in patterns:
        if p.search(t):
            return True
    return False


def _repeat_line_ratio(text) :
    lines = [_normalize_whitespace(x) for x in (text or "").splitlines()]
    lines = [x for x in lines if x]
    if len(lines) < 4:
        return 0.0
    c = Counter(lines)
    return max(c.values()) / max(1, len(lines))


def _quality_filter_decision(
    rec,
    text,
    url_patterns,
    path_patterns,
    title_patterns,
    text_patterns,
    min_alpha_ratio= 0.45,
    max_digit_ratio= 0.45,
    max_repeat_line_ratio= 0.5,
) :
    source_url = rec.get("source_url") or ""
    source_path = rec.get("source_path") or ""
    title = rec.get("title") or ""

    if _match_any(source_url, url_patterns):
        return False, "url_pattern"
    if _match_any(source_path, path_patterns):
        return False, "path_pattern"
    if _match_any(title, title_patterns):
        return False, "title_pattern"
    if _match_any(text, text_patterns):
        return False, "text_pattern"

    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return False, "empty"
    alpha = sum(ch.isalpha() for ch in compact)
    digit = sum(ch.isdigit() for ch in compact)
    alpha_ratio = alpha / max(1, len(compact))
    digit_ratio = digit / max(1, len(compact))
    if alpha_ratio < float(min_alpha_ratio):
        return False, "low_alpha_ratio"
    if digit_ratio > float(max_digit_ratio):
        return False, "high_digit_ratio"
    if _repeat_line_ratio(text) > float(max_repeat_line_ratio):
        return False, "repeat_lines"
    return True, ""


def _load_crawl_manifest(raw_dirs) :
    url_by_file = {}
    for raw_dir in raw_dirs:
        manifest_path = Path(raw_dir) / "manifest.jsonl"
        if not manifest_path.exists():
            continue
        for rec in _read_jsonl(manifest_path):
            fp = rec.get("file_path")
            url = rec.get("url")
            if fp and url and rec.get("status") == "ok":
                url_by_file[str(Path(fp).resolve())] = str(url)
    return url_by_file


def _check_image_ocr_dependencies() :
    try:
        import PIL  # noqa: F401
        import pytesseract  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Image OCR requires Pillow and pytesseract. Install with: pip install Pillow pytesseract"
        ) from e
    try:
        import pytesseract
        _ = pytesseract.get_tesseract_version()
    except Exception as e:
        raise RuntimeError(
            "pytesseract is installed, but the Tesseract binary is missing. Install Tesseract on your system."
        ) from e


def _extract_image_document(path, source_url= None) :
    from PIL import Image
    import pytesseract

    p = Path(path)
    with Image.open(str(p)) as img:
        text = pytesseract.image_to_string(img)
    text = _clean_text_basic(text)
    return {
        "title": p.stem,
        "text": text,
        "source_path": str(p),
        "source_url": source_url,
        "doc_type": "image",
        "metadata": {"parser": "pytesseract"},
    }


def iter_source_files(input_dirs, include_images= False) :
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
            in_asset_dir = any(part.endswith("_files") for part in p.parts)
            if in_asset_dir and not (include_images and p.suffix.lower() in IMAGE_EXTS):
                continue
            if p.name in SKIP_FILE_NAMES:
                continue
            allowed_exts = HTML_EXTS | PDF_EXTS | TEXT_EXTS
            if include_images:
                allowed_exts = allowed_exts | IMAGE_EXTS
            if p.suffix.lower() in allowed_exts:
                yield p


def _extract_doc(path, source_url= None, include_images= False) :
    ext = path.suffix.lower()
    if ext in HTML_EXTS:
        return extract_html_document(path, source_url=source_url)
    if ext in PDF_EXTS:
        return extract_pdf_document(path)
    if include_images and ext in IMAGE_EXTS:
        return _extract_image_document(path, source_url=source_url)
    if ext in TEXT_EXTS:
        text = _clean_text_basic(path.read_text(encoding="utf-8", errors="ignore"))
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
    input_dirs,
    output_jsonl,
    min_chars= 100,
    dedupe_exact= True,
    include_images= False,
    quality_filter= False,
    drop_url_patterns= None,
    drop_path_patterns= None,
    drop_title_patterns= None,
    drop_text_patterns= None,
    min_alpha_ratio= 0.45,
    max_digit_ratio= 0.45,
    max_repeat_line_ratio= 0.5,
    verbose= False,
) :
    if include_images:
        _check_image_ocr_dependencies()
    url_patterns = _compile_patterns(drop_url_patterns)
    path_patterns = _compile_patterns(drop_path_patterns)
    title_patterns = _compile_patterns(drop_title_patterns)
    text_patterns = _compile_patterns(drop_text_patterns)
    url_by_file = _load_crawl_manifest(input_dirs)
    docs = []
    raw_recs = []

    for p in iter_source_files(input_dirs, include_images=include_images):
        src_url = url_by_file.get(str(p.resolve()))
        try:
            rec = _extract_doc(p, source_url=src_url, include_images=include_images)
        except Exception as e:
            if verbose:
                print(f"[corpus] skip {p}: {e}")
            continue
        if not rec:
            continue
        text = rec["text"].strip()
        if len(text) < min_chars:
            continue
        if quality_filter:
            ok, why = _quality_filter_decision(
                rec,
                text,
                url_patterns=url_patterns,
                path_patterns=path_patterns,
                title_patterns=title_patterns,
                text_patterns=text_patterns,
                min_alpha_ratio=min_alpha_ratio,
                max_digit_ratio=max_digit_ratio,
                max_repeat_line_ratio=max_repeat_line_ratio,
            )
            if not ok:
                if verbose:
                    print(f"[corpus] filtered {p}: {why}")
                continue
        if verbose:
            print(f"[corpus] {p} -> {len(text)} chars")
        source_key = rec.get("source_url") or rec.get("source_path") or str(p)
        doc = Document(
            doc_id=_compute_doc_id(str(source_key)),
            text=text,
            title=rec.get("title"),
            source_path=rec.get("source_path"),
            source_url=rec.get("source_url"),
            doc_type=rec.get("doc_type"),
            metadata=rec.get("metadata", {}),
        )
        docs.append(doc)
        raw_recs.append(doc.to_dict())

    if dedupe_exact and docs:
        keep = set(_dedupe_texts_exact([d.text for d in docs]))
        docs = [d for i, d in enumerate(docs) if i in keep]
        raw_recs = [d.to_dict() for d in docs]

    _write_jsonl(raw_recs, output_jsonl)
    return docs
