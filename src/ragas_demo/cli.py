import asyncio

import typer

from ragas_demo.config import Settings
from ragas_demo.evaluation import (
    RagasMetricScorer,
    RagasSyntheticGenerator,
    collect_answers,
    generate_testset,
    load_jsonl,
    score_answers,
)
from ragas_demo.indexing import build_index
from ragas_demo.ingestion import chunk_pages, discover_pdfs, load_pdf_pages
from ragas_demo.reporting import write_report_artifacts
from ragas_demo.runtime import create_rag_service, create_vector_index

app = typer.Typer(help="Local RAG chatbot and staged RAGAS evaluation workflow.")


def _settings() -> Settings:
    return Settings()


def _chunks(settings: Settings):
    paths = discover_pdfs(settings.corpus_dir)
    pages = load_pdf_pages(paths)
    return chunk_pages(pages, settings.chunk_tokens, settings.chunk_overlap)


@app.command()
def ingest(rebuild: bool = typer.Option(False, help="Replace an existing matching index.")) -> None:
    """Extract, chunk, embed, and persist the PDF corpus."""
    settings = _settings()
    summary = build_index(settings, create_vector_index(settings), rebuild=rebuild)
    action = "reused" if summary.reused else "built"
    typer.echo(
        f"Index {action}: {summary.document_count} documents, {summary.page_count} pages, "
        f"{summary.chunk_count} chunks."
    )


@app.command("generate-testset")
def generate_testset_command(
    size: int = typer.Option(12, min=1, help="Number of synthetic cases."),
    force: bool = typer.Option(False, help="Replace the existing finalized test set."),
) -> None:
    """Generate synthetic question/reference pairs with RAGAS."""
    settings = _settings()
    output = settings.results_dir / "testset.jsonl"
    cases = generate_testset(
        _chunks(settings),
        RagasSyntheticGenerator(settings),
        size=size,
        output_path=output,
        force=force,
    )
    typer.echo(f"Generated {len(cases)} cases at {output}. Spot-check before continuing.")


@app.command("collect-answers")
def collect_answers_command(
    resume: bool = typer.Option(False, help="Continue from the existing response checkpoint."),
) -> None:
    """Run finalized synthetic questions through the shared chatbot service."""
    settings = _settings()
    cases = load_jsonl(settings.results_dir / "testset.jsonl")
    if not cases:
        raise typer.BadParameter("Generate and review results/testset.jsonl first")
    output = settings.results_dir / "responses.jsonl"
    records = collect_answers(cases, create_rag_service(settings), output, resume=resume)
    typer.echo(f"Collected {len(records)} responses at {output}.")


@app.command()
def score(
    resume: bool = typer.Option(False, help="Continue from the existing score checkpoint."),
) -> None:
    """Score collected responses using the four required RAGAS metrics."""
    settings = _settings()
    answered = load_jsonl(settings.results_dir / "responses.jsonl")
    if not answered:
        raise typer.BadParameter("Collect responses before scoring")
    output = settings.state_dir / "checkpoints" / "scores.jsonl"
    records = asyncio.run(
        score_answers(answered, RagasMetricScorer(settings), output, resume=resume)
    )
    typer.echo(f"Scored {len(records)} cases at {output}.")


@app.command()
def report() -> None:
    """Create finalized CSV, JSON, and Markdown results artifacts."""
    settings = _settings()
    scores = load_jsonl(settings.state_dir / "checkpoints" / "scores.jsonl")
    if not scores:
        raise typer.BadParameter("Score responses before creating the report")
    write_report_artifacts(scores, settings.results_dir)
    typer.echo(f"Wrote finalized results to {settings.results_dir}.")


@app.command()
def smoke(
    question: str = typer.Option(
        "What safety measures are recommended?",
        help="Single paid query used to validate credentials and the built index.",
    ),
) -> None:
    """Run one explicitly requested live OpenAI query."""
    result = create_rag_service(_settings()).answer(question)
    typer.echo(result.answer)
    for source in result.sources:
        typer.echo(f"- {source.source_file}, page {source.page} ({source.score:.3f})")


if __name__ == "__main__":
    app()
