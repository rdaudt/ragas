from pathlib import Path

import pytest

from ragas_demo.config import Settings


def test_settings_load_api_key_and_overrides_from_dotenv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=from-dotenv\n"
        "RAGAS_ANSWER_MODEL=answer-test\n"
        "RAGAS_TOP_K=3\n",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.openai_api_key.get_secret_value() == "from-dotenv"
    assert settings.answer_model == "answer-test"
    assert settings.top_k == 3


def test_settings_default_ragas_generation_is_low_burst(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=test\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.testset_generator == "openai"
    assert settings.eval_model == "gpt-4.1-mini"
    assert settings.eval_max_workers == 1
    assert settings.eval_max_retries >= 10
    assert settings.eval_max_wait >= 60


def test_settings_reject_invalid_chunk_overlap(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=test\nRAGAS_CHUNK_TOKENS=100\nRAGAS_CHUNK_OVERLAP=100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlap"):
        Settings(_env_file=env_file)


def test_dotenv_api_key_overrides_inherited_process_environment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "organizational-key")
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=personal-dotenv-key\n", encoding="utf-8")

    settings = Settings(_env_file=env_file)

    assert settings.openai_api_key.get_secret_value() == "personal-dotenv-key"
