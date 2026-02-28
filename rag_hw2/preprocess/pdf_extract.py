from __future__ import annotations

from pathlib import Path
import re
import unicodedata
import pdfplumber
from pypdf import PdfReader # type: ignore


def _clean_text_basic(text) :
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pdf_document(path) :
    p = Path(path)
    text = ""
    parser_used = None

    try:
        parser_used = "pdfplumber"
        pages = []
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

    text = _clean_text_basic(text)
    return {
        "title": p.stem,
        "text": text,
        "source_path": str(p),
        "source_url": None,
        "doc_type": "pdf",
        "metadata": {"parser": parser_used},
    }
