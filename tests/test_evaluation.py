import asyncio
from pathlib import Path

import pytest

from ragas_demo.evaluation import (
    collect_answers,
    load_jsonl,
    save_jsonl,
    score_answers,
    validate_complete_records,
)
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


def test_collect_answers_resume_rejects_checkpoint_from_replaced_testset(
    tmp_path: Path,
) -> None:
    output = tmp_path / "answers.jsonl"
    save_jsonl(
        output,
        [
            {
                "case_id": "old",
                "user_input": "old?",
                "reference": "old",
                "response": "existing",
                "retrieved_contexts": ["old"],
                "source_ids": ["old-id"],
            }
        ],
    )
    service = FakeService()

    with pytest.raises(ValueError, match="does not belong"):
        collect_answers(cases(), service, output, resume=True)

    assert service.questions == []


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


def test_score_answers_resume_rejects_checkpoint_from_replaced_responses(
    tmp_path: Path,
) -> None:
    output = tmp_path / "scores.jsonl"
    save_jsonl(
        output,
        [
            {
                "case_id": "a",
                "user_input": "old?",
                "reference": "first",
                "response": "answer",
                "retrieved_contexts": ["context"],
                "source_ids": ["id"],
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
                "context_recall": 0.6,
            }
        ],
    )
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

    with pytest.raises(ValueError, match="does not match"):
        asyncio.run(score_answers(answered, FakeScorer(), output, resume=True))


def test_validate_complete_records_rejects_missing_response() -> None:
    responses = [
        {
            "case_id": "a",
            "user_input": "first?",
            "reference": "first",
            "response": "answer",
            "retrieved_contexts": ["context"],
            "source_ids": ["id"],
        }
    ]

    with pytest.raises(ValueError, match="missing"):
        validate_complete_records(
            cases(),
            responses,
            fields=("user_input", "reference"),
            expected_label="test set",
            actual_label="responses",
        )
