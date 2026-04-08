from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader
)

def get_loader_for_file(file_path: str) -> any:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PyPDFLoader(file_path)
    elif suffix == ".docx":
        return UnstructuredWordDocumentLoader(file_path)
    elif suffix == ".md":
        return TextLoader(file_path, encoding="utf-8")
    elif suffix == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def load_document(file_path: str) -> List[Document]:
    loader = get_loader_for_file(file_path)
    return loader.load()
