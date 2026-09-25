"""Request and response models for the minimal API."""

from typing import Any

from pydantic import BaseModel, Field


class IndexRequest(BaseModel):
    docs_path: str | None = Field(
        default=None,
        description="Optional path to the folder of local documents to index.",
    )


class IndexedChunk(BaseModel):
    id: str
    source: str
    chunk_index: int
    page: int | None = None
    text_preview: str


class IndexResponse(BaseModel):
    collection_name: str
    documents_loaded: int
    chunks_indexed: int
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    indexed_chunks: list[IndexedChunk]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(
        default=None,
        ge=1,
        description="Optional override for how many chunks to retrieve.",
    )


class RetrievedChunk(BaseModel):
    score: float
    text: str
    metadata: dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    prompt_used: str
    retrieved_chunks: list[RetrievedChunk]


class HealthResponse(BaseModel):
    status: str
