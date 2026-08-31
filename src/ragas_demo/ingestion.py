import hashlib
import re
from collections.abc import Iterable
from pathlib import Path

import tiktoken
from pypdf import PdfReader

from ragas_demo.models import DocumentChunk, PageText


class CorpusError(RuntimeError):
    """Raised when a source PDF cannot be safely indexed."""


def discover_pdfs(corpus_dir: Path) -> list[Path]:
    return sorted(
        (path for path in corpus_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf"),
        key=lambda path: path.name.casefold(),
    )


def corpus_fingerprint(
    paths: Iterable[Path],
    chunk_tokens: int,
    chunk_overlap: int,
    embedding_model: str,
) -> str:
    digest = hashlib.sha256(f"{chunk_tokens}:{chunk_overlap}:{embedding_model}".encode())
    for path in sorted(paths, key=lambda item: item.name.casefold()):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pages(paths: Iterable[Path]) -> list[PageText]:
    pages: list[PageText] = []
    for path in paths:
        try:
            reader = PdfReader(path)
        except Exception as exc:
            raise CorpusError(f"Unable to open {path.name}: {exc}") from exc
        if reader.is_encrypted:
            raise CorpusError(f"Encrypted PDF is not supported: {path.name}")
        extracted = 0
        for number, page in enumerate(reader.pages, start=1):
            try:
                text = normalize_text(page.extract_text() or "")
            except Exception as exc:
                raise CorpusError(f"Unable to extract {path.name} page {number}: {exc}") from exc
            if text:
                extracted += len(text)
                pages.append(PageText(source_file=path.name, page=number, text=text))
        if extracted == 0:
            raise CorpusError(f"No extractable text found in {path.name}")
    return pages


def chunk_pages(
    pages: Iterable[PageText],
    chunk_tokens: int,
    chunk_overlap: int,
    encoding_name: str = "cl100k_base",
) -> list[DocumentChunk]:
    if chunk_tokens < 2 or chunk_overlap < 0 or chunk_overlap >= chunk_tokens:
        raise ValueError("chunk overlap must be non-negative and smaller than chunk size")
    encoding = tiktoken.get_encoding(encoding_name)
    step = chunk_tokens - chunk_overlap
    chunks: list[DocumentChunk] = []
    for page in pages:
        normalized = normalize_text(page.text)
        if not normalized:
            continue
        tokens = encoding.encode(normalized)
        for chunk_index, start in enumerate(range(0, len(tokens), step)):
            token_slice = tokens[start : start + chunk_tokens]
            if not token_slice:
                break
            text = encoding.decode(token_slice).strip()
            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{page.source_file}:{page.page}:{chunk_index}:{content_hash}"
            chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=text,
                    source_file=page.source_file,
                    page=page.page,
                    chunk_index=chunk_index,
                )
            )
            if start + chunk_tokens >= len(tokens):
                break
    return chunks
