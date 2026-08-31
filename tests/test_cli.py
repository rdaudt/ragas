from typer.testing import CliRunner

from ragas_demo.cli import app


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
