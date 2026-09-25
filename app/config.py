"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Small, explicit settings object for the teaching project."""

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "sample_docs"
    docs_path: Path = BASE_DIR / "sample_docs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    answer_model: str = "google/flan-t5-small"
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 3
    similarity_threshold: float = 0.35

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
