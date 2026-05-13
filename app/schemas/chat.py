from typing import List, Optional

from pydantic import BaseModel


class ChatRequest(BaseModel): 
    message: str
    session_id: Optional[str] = "default" # 默认使用 "default" 作为 session_id


class ChatResponse(BaseModel): 

    answer: str
    sources: List[dict]
