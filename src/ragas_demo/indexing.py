from collections.abc import Sequence
from typing import Protocol

from ragas_demo.config import Settings
from ragas_demo.ingestion import (
    CorpusError,
    chunk_pages,
    corpus_fingerprint,
    discover_pdfs,
    load_pdf_pages,
)
from ragas_demo.models import DocumentChunk, IndexSummary


class WritableIndex(Protocol):
    def matches(self, corpus_fingerprint: str) -> bool: ...

    def replace(
        self, chunks: Sequence[DocumentChunk], corpus_fingerprint: str
    ) -> None: ...


def build_index(settings: Settings, index: WritableIndex, rebuild: bool = False) -> IndexSummary:
    pdf_paths = discover_pdfs(settings.corpus_dir)
    if not pdf_paths:
        raise CorpusError(f"No PDF files found in {settings.corpus_dir}")
    fingerprint = corpus_fingerprint(
        pdf_paths,
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
        embedding_model=settings.embedding_model,
    )
    pages = load_pdf_pages(pdf_paths)
    chunks = chunk_pages(
        pages,
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise CorpusError("The corpus produced no indexable chunks")
    reused = index.matches(fingerprint) and not rebuild
    if not reused:
        index.replace(chunks, corpus_fingerprint=fingerprint)
    return IndexSummary(
        corpus_fingerprint=fingerprint,
        document_count=len(pdf_paths),
        page_count=len(pages),
        chunk_count=len(chunks),
        reused=reused,
    )
