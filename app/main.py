import os 
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"    # 配置 HuggingFace 镜像

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  #跨域中间件
from app.core.config import settings   #导入配置文件的settings类
from app.api import chat,documents   #导入chat和documents模块的路由
import fastapi_cdn_host

app = FastAPI(
    title="个人知识库管理系统",
    description="基于 FastAPI + LangChain + RAG 的个人知识库问答系统",
    version="1.0.0"
)

fastapi_cdn_host.patch_docs(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/health")
async def health():
    return {"status": "healthy"}
