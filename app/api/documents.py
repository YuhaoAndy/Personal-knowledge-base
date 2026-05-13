import os
import uuid
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.core.config import settings
from app.document_loaders.loaders import load_document
from app.schemas.document import DocumentInfo
from app.storage.vector_store import add_documents_to_vector_store, get_vector_store

router = APIRouter(prefix="/api/documents", tags=["文档管理"])

os.makedirs(settings.DOCUMENTS_DIR, exist_ok=True) 
# 确保documents目录存在,如果不存在则创建

# 文档上传接口
@router.post("/upload", response_model=DocumentInfo,status_code=status.HTTP_201_CREATED)
#返回上传后的文档信息，包括id、filename、chunk_count  响应状态码201表示资源创建成功

async def upload_document(file: UploadFile = File(...)): # 接收上传的文件
#upload_document函数接收一个UploadFile类型的参数file，
# 这个参数是通过FastAPI的File函数声明的，表示这是一个文件上传字段
# 客户端必须提供这个字段（因为File(...)中的...表示必填）。
# 函数将返回一个包含文档信息的DocumentInfo对象。

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="文件名不能为空")

    file_ext = Path(file.filename).suffix.lower() # 获取文件扩展名并转换为小写
    allowed_exts = [".pdf", ".docx", ".md", ".txt"] # 允许的文件扩展名
    if file_ext not in allowed_exts:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"不支持的文件类型: {file_ext}")

    file_id = str(uuid.uuid4()) # 生成唯一的文件ID，如 3c56f0e9
    safe_filename = f"{file_id}{file_ext}" # 生成安全的文件名，如 3c56f0e9.pdf
    file_path = os.path.join(settings.DOCUMENTS_DIR, safe_filename) # 构建完整文件路径

    try:
        content = await file.read()  # 异步读取上传的文件内容
        with open(file_path, "wb") as f:  # 将文件内容以二进制写入磁盘
            f.write(content)   # 将上传的文件内容写入到指定路径的文件中

        documents = load_document(file_path)
        #返回了一个Document对象列表，每个Document对象包含文档的内容和元数据
        
        for doc in documents:
            doc.metadata["filename"] = file.filename 
            doc.metadata["file_id"] = file_id  
            # 将原始文件名和生成的文件ID添加到文档的元数据中
        
        chunk_count = add_documents_to_vector_store(documents, file_id, file.filename)
        # 将文档分块并添加到向量数据库中，返回分块的数量

        return DocumentInfo(
            id=file_id,
            filename=file.filename,
            chunk_count=chunk_count
        )
    except Exception as e: # 处理文档上传和存储过程中可能发生的异常
         # 如果发生异常，尝试删除已经保存的文件以避免垃圾文件残留
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
