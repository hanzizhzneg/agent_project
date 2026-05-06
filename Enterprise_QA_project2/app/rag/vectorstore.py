from pathlib import Path
from typing import Literal
from uuid import uuid4

from langchain_community.vectorstores import Chroma, FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.config import settings

StoreType = Literal["chroma", "faiss"]


def _embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.ollama_embed_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.ollama_timeout_sec},
    )


def _store_path(base_dir: Path, store_type: StoreType) -> Path:
    path = base_dir / store_type
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_incremental_writer(
    store_type: StoreType,
    base_dir: Path | None = None,
):
    base_dir = base_dir or settings.vector_db_path
    store_path = _store_path(base_dir, store_type)
    emb = _embeddings()

    if store_type == "chroma":
        vector_store = Chroma(
            persist_directory=str(store_path),
            embedding_function=emb,
            collection_name="enterprise_qa",
        )

        def add_docs(docs: list[Document]):
            if not docs:
                return
            texts = [d.page_content for d in docs]
            metadatas = [d.metadata for d in docs]
            ids = [str(uuid4()) for _ in docs]
            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

        def finalize():
            return vector_store

        return add_docs, finalize

    vector_store: FAISS | None = None

    def add_docs(docs: list[Document]):
        nonlocal vector_store
        if not docs:
            return
        if vector_store is None:
            vector_store = FAISS.from_documents(documents=docs, embedding=emb)
            return
        vector_store.add_documents(docs)

    def finalize():
        if vector_store is None:
            raise ValueError("No documents were added to FAISS store.")
        vector_store.save_local(str(store_path))
        return vector_store

    return add_docs, finalize


def build_vector_store(
    docs: list[Document],
    store_type: StoreType,
    base_dir: Path | None = None,
):
    base_dir = base_dir or settings.vector_db_path
    store_path = _store_path(base_dir, store_type)
    emb = _embeddings()

    batch_size = max(1, settings.ingest_batch_size)

    if store_type == "chroma":
        vector_store = Chroma(
            persist_directory=str(store_path),
            embedding_function=emb,
            collection_name="enterprise_qa",
        )
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [d.page_content for d in batch]
            metadatas = [d.metadata for d in batch]
            ids = [str(uuid4()) for _ in batch]
            vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        return vector_store

    first_batch = docs[:batch_size]
    vector_store = FAISS.from_documents(documents=first_batch, embedding=emb)
    for i in range(batch_size, len(docs), batch_size):
        vector_store.add_documents(docs[i : i + batch_size])
    vector_store.save_local(str(store_path))
    return vector_store


def load_vector_store(store_type: StoreType, base_dir: Path | None = None):
    base_dir = base_dir or settings.vector_db_path
    store_path = _store_path(base_dir, store_type)
    emb = _embeddings()

    if store_type == "chroma":
        return Chroma(
            persist_directory=str(store_path),
            embedding_function=emb,
            collection_name="enterprise_qa",
        )

    if not (store_path / "index.faiss").exists():
        raise FileNotFoundError("FAISS index not found. Run /ingest first.")
    return FAISS.load_local(
        str(store_path),
        embeddings=emb,
        allow_dangerous_deserialization=False,
    )
