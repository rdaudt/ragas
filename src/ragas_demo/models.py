from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PageText:
    source_file: str
    page: int
    text: str


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: str
    text: str
    source_file: str
    page: int
    chunk_index: int


@dataclass(frozen=True, slots=True)
class SourceChunk(DocumentChunk):
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class AnswerResult:
    question: str
    answer: str
    sources: list[SourceChunk] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class IndexSummary:
    corpus_fingerprint: str
    document_count: int
    page_count: int
    chunk_count: int
    reused: bool

