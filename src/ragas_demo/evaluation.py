import asyncio
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, Field
from ragas import RunConfig
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.testset import TestsetGenerator

from ragas_demo.config import Settings
from ragas_demo.models import DocumentChunk
from ragas_demo.rag import RagService

METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class CaseScorer(Protocol):
    async def score(self, record: dict) -> dict[str, float]: ...


class SyntheticGenerator(Protocol):
    def generate(self, chunk_texts: Sequence[str], size: int) -> list[dict]: ...


class SyntheticCase(BaseModel):
    question: str = Field(min_length=1)
    reference: str = Field(min_length=1)


class SyntheticCaseBatch(BaseModel):
    cases: list[SyntheticCase]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def save_jsonl(path: Path, records: Sequence[dict], force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; use --force or --resume")
    _write_jsonl(path, records)


def _records_by_case_id(records: Sequence[dict], label: str) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    for record in records:
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{label} contains a record without a case_id")
        if case_id in by_id:
            raise ValueError(f"{label} contains duplicate case_id {case_id}")
        by_id[case_id] = record
    return by_id


def validate_record_subset(
    expected: Sequence[dict],
    actual: Sequence[dict],
    fields: Sequence[str],
    expected_label: str,
    actual_label: str,
) -> None:
    expected_by_id = _records_by_case_id(expected, expected_label)
    actual_by_id = _records_by_case_id(actual, actual_label)
    for case_id, actual_record in actual_by_id.items():
        expected_record = expected_by_id.get(case_id)
        if expected_record is None:
            raise ValueError(
                f"{actual_label} case {case_id} does not belong to current {expected_label}"
            )
        for field in fields:
            if actual_record.get(field) != expected_record.get(field):
                raise ValueError(
                    f"{actual_label} case {case_id} does not match current "
                    f"{expected_label} field {field}"
                )


def validate_complete_records(
    expected: Sequence[dict],
    actual: Sequence[dict],
    fields: Sequence[str],
    expected_label: str,
    actual_label: str,
) -> None:
    validate_record_subset(expected, actual, fields, expected_label, actual_label)
    expected_ids = set(_records_by_case_id(expected, expected_label))
    actual_ids = set(_records_by_case_id(actual, actual_label))
    missing = sorted(expected_ids - actual_ids)
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else ", ..."
        raise ValueError(
            f"{actual_label} missing {len(missing)} {expected_label} case(s): "
            f"{preview}{suffix}"
        )


def collect_answers(
    cases: Sequence[dict],
    service: RagService,
    output_path: Path,
    resume: bool = False,
) -> list[dict]:
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {output_path}; use --resume")
    completed = load_jsonl(output_path) if resume else []
    validate_record_subset(
        cases,
        completed,
        fields=("user_input", "reference"),
        expected_label="test set",
        actual_label="response checkpoint",
    )
    completed_ids = {record["case_id"] for record in completed}
    for case in cases:
        if case["case_id"] in completed_ids:
            continue
        result = service.answer(case["user_input"])
        completed.append(
            {
                "case_id": case["case_id"],
                "user_input": case["user_input"],
                "reference": case["reference"],
                "response": result.answer,
                "retrieved_contexts": [source.text for source in result.sources],
                "source_ids": [source.id for source in result.sources],
            }
        )
        _write_jsonl(output_path, completed)
    return completed


async def score_answers(
    answered: Sequence[dict],
    scorer: CaseScorer,
    output_path: Path,
    resume: bool = False,
) -> list[dict]:
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {output_path}; use --resume")
    completed = load_jsonl(output_path) if resume else []
    validate_record_subset(
        answered,
        completed,
        fields=("user_input", "reference", "response", "retrieved_contexts", "source_ids"),
        expected_label="responses",
        actual_label="score checkpoint",
    )
    completed_ids = {record["case_id"] for record in completed}
    for record in answered:
        if record["case_id"] in completed_ids:
            continue
        metrics = await scorer.score(record)
        for name in METRIC_NAMES:
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Missing or invalid {name} for case {record['case_id']}")
        completed.append({**record, **metrics})
        _write_jsonl(output_path, completed)
    return completed


class RagasSyntheticGenerator:
    def __init__(self, settings: Settings) -> None:
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        llm = llm_factory(settings.eval_model, client=client)
        embeddings = embedding_factory(
            "openai",
            model=settings.embedding_model,
            client=client,
            interface="modern",
        )
        self.generator = TestsetGenerator(llm=llm, embedding_model=embeddings)
        self.run_config = RunConfig(
            max_workers=settings.eval_max_workers,
            max_retries=settings.eval_max_retries,
            max_wait=settings.eval_max_wait,
        )

    def generate(self, chunk_texts: Sequence[str], size: int) -> list[dict]:
        run_config = self.run_config or RunConfig(max_workers=1, max_retries=20, max_wait=90)
        testset = self.generator.generate_with_chunks(
            list(chunk_texts),
            testset_size=size,
            run_config=run_config,
        )
        cases = []
        for sample in testset.to_list():
            question = sample.get("user_input")
            reference = sample.get("reference")
            if not question or not reference:
                raise ValueError("RAGAS generated a case without user_input or reference")
            digest = hashlib.sha256(f"{question}\n{reference}".encode()).hexdigest()[:16]
            cases.append({"case_id": digest, "user_input": question, "reference": reference})
        if len(cases) != size:
            raise ValueError(f"RAGAS generated {len(cases)} cases; expected {size}")
        return cases


class OpenAISyntheticGenerator:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    @classmethod
    def from_settings(cls, settings: Settings) -> "OpenAISyntheticGenerator":
        client = OpenAI(api_key=settings.openai_api_key.get_secret_value())
        return cls(client=client, model=settings.eval_model)

    def generate(self, chunk_texts: Sequence[str], size: int) -> list[dict]:
        cases: list[dict] = []
        for chunk_text in chunk_texts:
            if len(cases) >= size:
                break
            response = self.client.responses.parse(
                model=self.model,
                instructions=(
                    "Generate one grounded RAG evaluation test case from the provided "
                    "source text. The question must be answerable from the source text. "
                    "The reference answer must be concise and factual."
                ),
                input=f"Source text:\n{chunk_text}",
                text_format=SyntheticCaseBatch,
                temperature=0.2,
                max_output_tokens=500,
            )
            parsed = response.output_parsed
            for synthetic_case in parsed.cases:
                question = synthetic_case.question.strip()
                reference = synthetic_case.reference.strip()
                if not question or not reference:
                    continue
                digest = hashlib.sha256(f"{question}\n{reference}".encode()).hexdigest()[:16]
                cases.append(
                    {"case_id": digest, "user_input": question, "reference": reference}
                )
                if len(cases) >= size:
                    break
        if len(cases) != size:
            raise ValueError(f"OpenAI generated {len(cases)} cases; expected {size}")
        return cases


def generate_testset(
    chunks: Sequence[DocumentChunk],
    generator: SyntheticGenerator,
    size: int,
    output_path: Path,
    force: bool = False,
    limit: int | None = None,
) -> list[dict]:
    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; use --force")
    source_chunks = chunks[:limit] if limit is not None else chunks
    cases = generator.generate([chunk.text for chunk in source_chunks], size)
    save_jsonl(output_path, cases, force=force)
    return cases


class RagasMetricScorer:
    def __init__(self, settings: Settings) -> None:
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        llm = llm_factory(settings.eval_model, client=client)
        embeddings = embedding_factory(
            "openai",
            model=settings.embedding_model,
            client=client,
            interface="modern",
        )
        self.metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
            "context_precision": ContextPrecision(llm=llm),
            "context_recall": ContextRecall(llm=llm),
        }

    async def _with_retries(self, operation, attempts: int = 3):
        for attempt in range(attempts):
            try:
                return await operation()
            except Exception:
                if attempt == attempts - 1:
                    raise
                await asyncio.sleep(2**attempt)
        raise RuntimeError("unreachable")

    async def score(self, record: dict) -> dict[str, float]:
        arguments = {
            "user_input": record["user_input"],
            "response": record["response"],
            "reference": record["reference"],
            "retrieved_contexts": record["retrieved_contexts"],
        }
        calls = {
            "faithfulness": lambda: self.metrics["faithfulness"].ascore(
                user_input=arguments["user_input"],
                response=arguments["response"],
                retrieved_contexts=arguments["retrieved_contexts"],
            ),
            "answer_relevancy": lambda: self.metrics["answer_relevancy"].ascore(
                user_input=arguments["user_input"], response=arguments["response"]
            ),
            "context_precision": lambda: self.metrics["context_precision"].ascore(
                user_input=arguments["user_input"],
                reference=arguments["reference"],
                retrieved_contexts=arguments["retrieved_contexts"],
            ),
            "context_recall": lambda: self.metrics["context_recall"].ascore(
                user_input=arguments["user_input"],
                reference=arguments["reference"],
                retrieved_contexts=arguments["retrieved_contexts"],
            ),
        }
        results = {}
        for name, call in calls.items():
            result = await self._with_retries(call)
            results[name] = float(result.value)
        return results
