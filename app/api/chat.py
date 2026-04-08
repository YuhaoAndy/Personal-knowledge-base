from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableWithMessageHistory

from app.chains.rag_chain import llm, get_conversation_history, create_rag_chain
from app.storage.vector_store import get_retriever

router = APIRouter(prefix="/api/chat", tags=["对话"])

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

_session_histories = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_histories:
        _session_histories[session_id] = get_conversation_history()
    return _session_histories[session_id]

@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    try:
        retriever = get_retriever()
        
        docs = retriever.invoke(request.message)
        
        sources = []
        for doc in docs:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "filename": doc.metadata.get("filename", "未知")
            })
        
        context = "\n\n".join(doc.page_content for doc in docs)
        
        prompt = f"""你是一个知识库问答助手。请根据以下参考文档回答用户的问题。

参考文档：
{context}

用户问题：{request.message}

请给出回答，如果参考文档中没有相关信息，请说明"根据当前知识库无法回答该问题"。"""
        
        history = get_session_history(request.session_id)
        
        messages = history.messages + [HumanMessage(content=prompt)]
        
        response = llm.invoke(messages)
        
        history.add_user_message(request.message)
        history.add_ai_message(response.content)
        
        return ChatResponse(
            answer=response.content,
            sources=sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")

@router.get("/history")
async def get_history(session_id: str = "default"):
    history = get_session_history(session_id)
    messages = []
    for msg in history.messages:
        if isinstance(msg, HumanMessage):
            messages.append({"type": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"type": "ai", "content": msg.content})
    return {"session_id": session_id, "messages": messages}

@router.delete("/clear")
async def clear_history(session_id: str = "default"):
    if session_id in _session_histories:
        _session_histories[session_id].clear()
    return {"message": "对话历史已清除"}
