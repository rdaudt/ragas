import asyncio
from pathlib import Path

import pytest

from ragas_demo.evaluation import collect_answers, load_jsonl, save_jsonl, score_answers
from ragas_demo.models import AnswerResult, SourceChunk


class FakeService:
    def __init__(self) -> None:
        self.questions: list[str] = []

    def answer(self, question: str) -> AnswerResult:
        self.questions.append(question)
        source = SourceChunk("id", "context", "guide.pdf", 2, 0, 0.9)
        return AnswerResult(question, f"answer: {question}", [source])


class FakeScorer:
    async def score(self, record: dict) -> dict[str, float]:
        return {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }


def cases() -> list[dict]:
    return [
        {"case_id": "a", "user_input": "first?", "reference": "first"},
        {"case_id": "b", "user_input": "second?", "reference": "second"},
    ]


def test_save_jsonl_refuses_unintended_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    save_jsonl(output, [{"case_id": "a"}])

    with pytest.raises(FileExistsError):
        save_jsonl(output, [{"case_id": "b"}])


def test_collect_answers_resumes_completed_cases(tmp_path: Path) -> None:
    output = tmp_path / "answers.jsonl"
    save_jsonl(
        output,
        [
            {
                "case_id": "a",
                "user_input": "first?",
                "reference": "first",
                "response": "existing",
                "retrieved_contexts": ["old"],
                "source_ids": ["old-id"],
            }
        ],
    )
    service = FakeService()

    collect_answers(cases(), service, output, resume=True)

    assert service.questions == ["second?"]
    assert [row["case_id"] for row in load_jsonl(output)] == ["a", "b"]


def test_score_answers_writes_all_four_numeric_metrics(tmp_path: Path) -> None:
    output = tmp_path / "scores.jsonl"
    answered = [
        {
            "case_id": "a",
            "user_input": "first?",
            "reference": "first",
            "response": "answer",
            "retrieved_contexts": ["context"],
            "source_ids": ["id"],
        }
    ]

    asyncio.run(score_answers(answered, FakeScorer(), output))

    row = load_jsonl(output)[0]
    assert row["faithfulness"] == 0.9
    assert row["answer_relevancy"] == 0.8
    assert row["context_precision"] == 0.7
    assert row["context_recall"] == 0.6

