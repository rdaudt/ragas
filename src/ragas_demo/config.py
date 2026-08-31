from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from the project-local .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: SecretStr = Field(validation_alias=AliasChoices("OPENAI_API_KEY"))
    answer_model: str = Field("gpt-4.1-mini", validation_alias="RAGAS_ANSWER_MODEL")
    eval_model: str = Field("gpt-4.1", validation_alias="RAGAS_EVAL_MODEL")
    embedding_model: str = Field(
        "text-embedding-3-small", validation_alias="RAGAS_EMBEDDING_MODEL"
    )
    top_k: int = Field(5, ge=1, validation_alias="RAGAS_TOP_K")
    chunk_tokens: int = Field(800, ge=2, validation_alias="RAGAS_CHUNK_TOKENS")
    chunk_overlap: int = Field(120, ge=0, validation_alias="RAGAS_CHUNK_OVERLAP")
    corpus_dir: Path = Path("grounding-documents")
    state_dir: Path = Path(".ragas-demo")
    results_dir: Path = Path("results")

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self

