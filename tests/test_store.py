from pathlib import Path

from ragas_demo.models import DocumentChunk
from ragas_demo.store import VectorIndex


class FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "safety" in text.lower() else [0.0, 1.0] for text in texts]


def make_chunk(identifier: str, text: str, page: int) -> DocumentChunk:
    return DocumentChunk(
        id=identifier,
        text=text,
        source_file="guide.pdf",
        page=page,
        chunk_index=page - 1,
    )


def test_vector_index_replaces_and_queries_ranked_chunks(tmp_path: Path) -> None:
    index = VectorIndex(tmp_path / "chroma", FakeEmbedder())
    chunks = [
        make_chunk("safety", "Safety helmets are required.", 1),
        make_chunk("vehicles", "Vehicle inspection intervals.", 2),
    ]

    index.replace(chunks, corpus_fingerprint="abc")
    sources = index.query("What safety equipment is required?", top_k=2)

    assert index.matches("abc")
    assert [source.id for source in sources] == ["safety", "vehicles"]
    assert sources[0].score > sources[1].score
    assert sources[0].source_file == "guide.pdf"
    assert sources[0].page == 1


def test_vector_index_does_not_match_missing_fingerprint(tmp_path: Path) -> None:
    index = VectorIndex(tmp_path / "chroma", FakeEmbedder())

    assert not index.matches("missing")

