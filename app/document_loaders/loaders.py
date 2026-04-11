from pathlib import Path # 导入Path类，用于处理文件路径
from typing import List # 导入List类型，用于返回文档列表
from langchain_core.documents import Document # 导入Document类，用于表示文档对象
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
