from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.core.config import settings

embedding_fn = SentenceTransformerEmbeddings(
    model_name=settings.EMBEDDING_MODEL
)

def get_vector_store():
    return Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=embedding_fn
    )

def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len
    )
    return text_splitter.split_documents(documents)

def add_documents_to_vector_store(documents: List[Document], file_id: str = None, filename: str = None) -> int:
    chunks = split_documents(documents)
    vector_store = get_vector_store()
    
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    metadatas = []
    for i, doc in enumerate(chunks):
        meta = dict(doc.metadata) if doc.metadata else {}
        if file_id:
            meta["file_id"] = file_id
        if filename:
            meta["filename"] = filename
        metadatas.append(meta)
    
    texts = [doc.page_content for doc in chunks]
    
    vector_store.add_texts(
        texts=texts,
        ids=ids,
        metadatas=metadatas
    )
    return len(chunks)

def delete_all_documents():
    vector_store = get_vector_store()
    vector_store.delete_collection()

def get_retriever():
    vector_store = get_vector_store()
    return vector_store.as_retriever()
