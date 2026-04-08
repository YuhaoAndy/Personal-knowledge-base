import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.documents import router as documents_router
from app.api.chat import router as chat_router
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

app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/health")
async def health():
    return {"status": "healthy"}
