from typing import Protocol

from openai import OpenAI

from ragas_demo.models import AnswerResult, SourceChunk

INSUFFICIENT_CONTEXT_MESSAGE = "I don't have enough information in the indexed documents."


class SearchIndex(Protocol):
    def query(self, question: str, top_k: int) -> list[SourceChunk]: ...


class AnswerGenerator(Protocol):
    def generate(self, question: str, sources: list[SourceChunk]) -> str: ...


class OpenAIAnswerGenerator:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def generate(self, question: str, sources: list[SourceChunk]) -> str:
        context = "\n\n".join(
            f"SOURCE: [{source.source_file}, p. {source.page}]\n{source.text}"
            for source in sources
        )
        response = self.client.responses.create(
            model=self.model,
            store=False,
            input=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the supplied sources. Cite factual statements with "
                        "[filename, p. N]. If the sources do not support an answer, say you do not "
                        "have enough information in the indexed documents."
                    ),
                },
                {"role": "user", "content": f"QUESTION:\n{question}\n\nSOURCES:\n{context}"},
            ],
        )
        return response.output_text.strip()


class RagService:
    def __init__(self, index: SearchIndex, generator: AnswerGenerator, top_k: int) -> None:
        self.index = index
        self.generator = generator
        self.top_k = top_k

    def answer(self, question: str) -> AnswerResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("question must not be blank")
        sources = self.index.query(normalized_question, self.top_k)
        if not sources:
            return AnswerResult(
                question=normalized_question,
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                sources=[],
            )
        answer = self.generator.generate(normalized_question, sources)
        return AnswerResult(question=normalized_question, answer=answer, sources=sources)

