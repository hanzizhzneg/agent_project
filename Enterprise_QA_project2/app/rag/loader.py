from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".doc"}


def list_supported_files(docs_dir: Path) -> list[Path]:
    files: list[Path] = []
    for file_path in docs_dir.glob("**/*"):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(file_path)
    return files


def load_document_file(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(file_path)).load()
    if suffix in {".docx", ".doc"}:
        return Docx2txtLoader(str(file_path)).load()
    return []


def load_documents(docs_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for file_path in list_supported_files(docs_dir):
        documents.extend(load_document_file(file_path))
    return documents


def split_documents(
    docs: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", " ", ""],
    )
    return splitter.split_documents(docs)
