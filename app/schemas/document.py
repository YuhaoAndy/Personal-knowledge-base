from pydantic import BaseModel


class DocumentInfo(BaseModel):
    id: str
    filename: str 
    chunk_count: int  # 文档被分成了多少块
