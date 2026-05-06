from typing import Literal

from pydantic import BaseModel, Field


StoreType = Literal["chroma", "faiss"]


class IngestRequest(BaseModel):
    docs_dir: str | None = None
    store_type: StoreType = "chroma"
    chunk_size: int = Field(default=800, ge=200, le=2000)
    chunk_overlap: int = Field(default=120, ge=0, le=400)


class IngestResponse(BaseModel):
    message: str
    store_type: StoreType
    docs_count: int
    chunks_count: int


class IngestAsyncResponse(BaseModel):
    task_id: str
    status: str
    message: str


class IngestTaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: int
    stage: str
    detail: str
    result: IngestResponse | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=2)
    store_type: StoreType = "chroma"
    top_k: int = Field(default=2, ge=1, le=10)
    max_iterations: int = Field(default=1, ge=1, le=3)


class IterationTrace(BaseModel):
    iteration: int
    query: str
    relevance_decision: str
    relevance_reason: str


class AskResponse(BaseModel):
    question: str
    final_query: str
    answer: str
    iterations_used: int
    trace: list[IterationTrace]
    sources: list[dict]
