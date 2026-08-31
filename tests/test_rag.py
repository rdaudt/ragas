from ragas_demo.models import SourceChunk
from ragas_demo.rag import INSUFFICIENT_CONTEXT_MESSAGE, RagService


class FakeIndex:
    def __init__(self, sources: list[SourceChunk]) -> None:
        self.sources = sources

    def query(self, question: str, top_k: int) -> list[SourceChunk]:
        assert question
        assert top_k == 5
        return self.sources


class RecordingGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[SourceChunk]]] = []

    def generate(self, question: str, sources: list[SourceChunk]) -> str:
        self.calls.append((question, sources))
        return "Wear a helmet [guide.pdf, p. 3]."


def source() -> SourceChunk:
    return SourceChunk(
        id="guide:3:0",
        text="Workers must wear helmets.",
        source_file="guide.pdf",
        page=3,
        chunk_index=0,
        score=0.91,
    )


def test_answer_returns_generated_text_and_exact_retrieved_sources() -> None:
    generator = RecordingGenerator()
    service = RagService(FakeIndex([source()]), generator, top_k=5)

    result = service.answer("What protective equipment is required?")

    assert result.answer == "Wear a helmet [guide.pdf, p. 3]."
    assert result.sources == [source()]
    assert generator.calls == [(result.question, result.sources)]


def test_answer_does_not_call_model_without_retrieved_context() -> None:
    generator = RecordingGenerator()
    service = RagService(FakeIndex([]), generator, top_k=5)

    result = service.answer("What is not in the corpus?")

    assert result.answer == INSUFFICIENT_CONTEXT_MESSAGE
    assert result.sources == []
    assert generator.calls == []

