from pathlib import Path

from ragas_demo.ingestion import chunk_pages, corpus_fingerprint
from ragas_demo.models import PageText


def test_corpus_fingerprint_changes_with_file_content(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"first")
    first = corpus_fingerprint(
        [pdf],
        chunk_tokens=20,
        chunk_overlap=5,
        embedding_model="text-embedding-3-small",
    )
    pdf.write_bytes(b"second")

    assert (
        corpus_fingerprint(
            [pdf],
            chunk_tokens=20,
            chunk_overlap=5,
            embedding_model="text-embedding-3-small",
        )
        != first
    )

def test_corpus_fingerprint_changes_with_embedding_model(tmp_path: Path) -> None:
    pdf = tmp_path / "sample.pdf"
    pdf.write_bytes(b"same corpus")

    first = corpus_fingerprint(
        [pdf],
        chunk_tokens=20,
        chunk_overlap=5,
        embedding_model="text-embedding-3-small",
    )

    assert (
        corpus_fingerprint(
            [pdf],
            chunk_tokens=20,
            chunk_overlap=5,
            embedding_model="text-embedding-3-large",
        )
        != first
    )


def test_chunk_pages_is_deterministic_and_preserves_metadata() -> None:
    pages = [PageText(source_file="guide.pdf", page=4, text="alpha beta gamma delta epsilon")]

    first = chunk_pages(pages, chunk_tokens=3, chunk_overlap=1)
    second = chunk_pages(pages, chunk_tokens=3, chunk_overlap=1)

    assert first == second
    assert len(first) >= 2
    assert first[0].source_file == "guide.pdf"
    assert first[0].page == 4
    assert first[0].chunk_index == 0
    assert first[0].id.startswith("guide.pdf:4:0:")


def test_chunk_pages_skips_blank_pages() -> None:
    pages = [PageText(source_file="blank.pdf", page=1, text="  \n ")]

    assert chunk_pages(pages, chunk_tokens=20, chunk_overlap=5) == []
