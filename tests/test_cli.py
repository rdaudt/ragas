from typer.testing import CliRunner

from ragas_demo.cli import _generator_for_backend, app
from ragas_demo.config import Settings
from ragas_demo.evaluation import OpenAISyntheticGenerator, RagasSyntheticGenerator


def test_cli_exposes_staged_workflow_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "ingest",
        "generate-testset",
        "collect-answers",
        "score",
        "report",
        "smoke",
    ):
        assert command in result.stdout


def test_generator_backend_defaults_to_openai(tmp_path) -> None:
    settings = Settings(OPENAI_API_KEY="test", state_dir=tmp_path)

    assert isinstance(_generator_for_backend(settings, None), OpenAISyntheticGenerator)


def test_generator_backend_can_select_ragas(tmp_path) -> None:
    settings = Settings(OPENAI_API_KEY="test", state_dir=tmp_path)

    assert isinstance(_generator_for_backend(settings, "ragas"), RagasSyntheticGenerator)
