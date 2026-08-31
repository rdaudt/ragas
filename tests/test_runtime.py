from pathlib import Path

from ragas_demo.config import Settings
from ragas_demo.runtime import create_openai_client


def test_openai_client_receives_key_from_settings(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, str] = {}

    class FakeOpenAI:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key

    monkeypatch.setattr("ragas_demo.runtime.OpenAI", FakeOpenAI)
    settings = Settings(OPENAI_API_KEY="dotenv-secret", state_dir=tmp_path)

    client = create_openai_client(settings)

    assert isinstance(client, FakeOpenAI)
    assert captured == {"api_key": "dotenv-secret"}

