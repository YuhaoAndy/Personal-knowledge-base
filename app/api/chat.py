import os
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import FileChatMessageHistory

from app.chains.rag_chain import answer_with_rag
from app.core.config import settings

router = APIRouter(prefix="/api/chat", tags=["对话"])


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    os.makedirs(settings.CHAT_HISTORY_DIR, exist_ok=True)
    file_path = os.path.join(settings.CHAT_HISTORY_DIR, f"{session_id}.json")
    return FileChatMessageHistory(file_path)


@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    try:
        history = get_session_history(request.session_id)
        answer, sources = answer_with_rag(request.message, history.messages)

        history.add_user_message(request.message)
        history.add_ai_message(answer)

        return ChatResponse(
            answer=answer,
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
    file_path = os.path.join(settings.CHAT_HISTORY_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"message": "对话历史已清除"}


@router.get("/sessions")
async def list_sessions():
    os.makedirs(settings.CHAT_HISTORY_DIR, exist_ok=True)
    files = os.listdir(settings.CHAT_HISTORY_DIR)
    sessions = [f.replace('.json', '') for f in files if f.endswith('.json')]
    return {"sessions": sessions}
