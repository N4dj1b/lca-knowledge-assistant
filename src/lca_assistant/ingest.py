"""Load PDF, Markdown, and text files into traceable text chunks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import fitz

from .models import DocumentChunk

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def _chunk_text(text: str, max_chars: int = 1100, overlap: int = 140) -> list[str]:
    """Split text on paragraph/sentence boundaries with a small overlap."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    # Paragraph boundaries are preferred, but PDF extraction often collapses them.
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) > max_chars:
            start = 0
            while start < len(sentence):
                end = min(start + max_chars, len(sentence))
                chunks.append(sentence[start:end].strip())
                if end == len(sentence):
                    break
                start = max(0, end - overlap)
            current = ""
        else:
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sentence}".strip()
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _make_chunk(text: str, source: str, page: int | None, index: int) -> DocumentChunk:
    digest = hashlib.sha1(f"{source}:{page}:{index}:{text}".encode("utf-8")).hexdigest()[:12]
    return DocumentChunk(text=text, source=source, page=page, chunk_id=digest)


def load_file(path: str | Path) -> list[DocumentChunk]:
    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use PDF, TXT, or Markdown.")

    chunks: list[DocumentChunk] = []
    if path.suffix.lower() == ".pdf":
        with fitz.open(path) as pdf:
            for page_index, page in enumerate(pdf, start=1):
                for chunk_index, chunk_text in enumerate(_chunk_text(page.get_text("text"))):
                    chunks.append(_make_chunk(chunk_text, path.name, page_index, chunk_index))
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        for chunk_index, chunk_text in enumerate(_chunk_text(text)):
            chunks.append(_make_chunk(chunk_text, path.name, None, chunk_index))
    return chunks


def load_directory(directory: str | Path) -> list[DocumentChunk]:
    directory = Path(directory)
    if not directory.exists():
        return []
    chunks: list[DocumentChunk] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            chunks.extend(load_file(path))
    return chunks


def load_uploaded_files(files: Iterable) -> list[DocumentChunk]:
    """Load Streamlit-style uploaded files without trusting their original paths."""
    chunks: list[DocumentChunk] = []
    for uploaded in files:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        payload = uploaded.getvalue()
        if suffix == ".pdf":
            with fitz.open(stream=payload, filetype="pdf") as pdf:
                for page_index, page in enumerate(pdf, start=1):
                    for chunk_index, chunk_text in enumerate(_chunk_text(page.get_text("text"))):
                        chunks.append(_make_chunk(chunk_text, uploaded.name, page_index, chunk_index))
        else:
            text = payload.decode("utf-8", errors="replace")
            for chunk_index, chunk_text in enumerate(_chunk_text(text)):
                chunks.append(_make_chunk(chunk_text, uploaded.name, None, chunk_index))
    return chunks
