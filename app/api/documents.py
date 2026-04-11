import os
import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.document_loaders.loaders import load_document
from app.storage.vector_store import add_documents_to_vector_store, get_vector_store

router = APIRouter(prefix="/api/documents", tags=["文档管理"])

os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True) # 确保文档目录存在

class DocumentInfo(BaseModel):
    id: str
    filename: str
    chunk_count: int

# 文档上传接口
@router.post("/upload", response_model=DocumentInfo) #返回上传后的文档信息，包括id、filename、chunk_count
async def upload_document(file: UploadFile = File(...)): # 接收上传的文件
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    file_ext = Path(file.filename).suffix.lower() # 获取文件扩展名并转换为小写
    allowed_exts = [".pdf", ".docx", ".md", ".txt"] # 允许的文件扩展名
    if file_ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file_ext}")

    file_id = str(uuid.uuid4()) # 生成唯一的文件ID
    safe_filename = f"{file_id}{file_ext}" # 生成安全的文件名
    file_path = os.path.join(settings.DOCUMENTS_DIR, safe_filename) # 构建文件路径

    try:
        content = await file.read() 
        with open(file_path, "wb") as f:
            f.write(content)

        documents = load_document(file_path)
        
        for doc in documents:
            doc.metadata["filename"] = file.filename
            doc.metadata["file_id"] = file_id
        
        chunk_count = add_documents_to_vector_store(documents, file_id, file.filename)

        return DocumentInfo(
            id=file_id,
            filename=file.filename,
            chunk_count=chunk_count
        )
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"处理文档失败: {str(e)}")

@router.get("", response_model=List[DocumentInfo])
async def list_documents():
    vector_store = get_vector_store()
    results = vector_store.get(include=["metadatas"])

    doc_info_dict = {}
    for id_, metadata in zip(results["ids"], results.get("metadatas", [])):
        if metadata:
            file_id = metadata.get("file_id")
            filename = metadata.get("filename")
            if file_id and filename:
                if file_id not in doc_info_dict:
                    doc_info_dict[file_id] = {"id": file_id, "filename": filename, "chunk_count": 0}
                doc_info_dict[file_id]["chunk_count"] += 1

    return list(doc_info_dict.values())

@router.delete("/{file_id}")
async def delete_document(file_id: str):
    try:
        vector_store = get_vector_store()
        results = vector_store.get(where={"file_id": file_id})
        if results["ids"]:
            vector_store.delete(ids=results["ids"])

        # 同步删除本地源文件，文件命名格式为 {file_id}.{ext}
        documents_dir = Path(settings.DOCUMENTS_DIR)
        for file_path in documents_dir.glob(f"{file_id}.*"):
            if file_path.is_file():
                file_path.unlink()

        return {"message": "删除成功"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")
