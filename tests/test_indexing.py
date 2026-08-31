from pathlib import Path

from ragas_demo.config import Settings
from ragas_demo.indexing import build_index
from ragas_demo.models import PageText


class FakeIndex:
    def __init__(self, matching: bool) -> None:
        self.matching = matching
        self.replaced = []

    def matches(self, fingerprint: str) -> bool:
        assert len(fingerprint) == 64
        return self.matching

    def replace(self, chunks, corpus_fingerprint: str) -> None:
        self.replaced.append((list(chunks), corpus_fingerprint))


def settings(tmp_path: Path) -> Settings:
    return Settings(
        OPENAI_API_KEY="test",
        corpus_dir=tmp_path,
        state_dir=tmp_path / "state",
        chunk_tokens=10,
        chunk_overlap=2,
    )


def test_build_index_reuses_matching_collection(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(
        "ragas_demo.indexing.load_pdf_pages",
        lambda paths: [PageText(source_file="one.pdf", page=1, text="some useful text")],
    )
    index = FakeIndex(matching=True)

    summary = build_index(settings(tmp_path), index)

    assert summary.reused
    assert summary.document_count == 1
    assert index.replaced == []


def test_build_index_replaces_when_rebuild_requested(tmp_path: Path, monkeypatch) -> None:
    pdf = tmp_path / "one.pdf"
    pdf.write_bytes(b"pdf")
    monkeypatch.setattr(
        "ragas_demo.indexing.load_pdf_pages",
        lambda paths: [PageText(source_file="one.pdf", page=1, text="some useful text")],
    )
    index = FakeIndex(matching=True)

    summary = build_index(settings(tmp_path), index, rebuild=True)

    assert not summary.reused
    assert summary.chunk_count == len(index.replaced[0][0])

