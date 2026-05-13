import os
from pathlib import Path # 导入Path类用于路径操作
from pydantic_settings import BaseSettings

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


BASE_DIR = Path(__file__).resolve().parent.parent.parent 
#解释：
# path(__file__)表示当前文件的路径
# resolve()方法将其转换为绝对路径
# parent.parent.parent则获取上三级目录，即项目根目录。



class Settings(BaseSettings):
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    HF_ENDPOINT: str = "https://hf-mirror.com"
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma")
    DOCUMENTS_DIR: str = str(BASE_DIR / "data" / "documents")
    CHAT_HISTORY_DIR: str = str(BASE_DIR / "data" / "chat_history")

    CHUNK_SIZE: int = 500 # 文档分块大小 每块包含500个字符
    CHUNK_OVERLAP: int = 50 # 文档分块重叠部分，避免信息丢失 每块与前一块重叠50个字符

    # 会话历史裁剪配置：限制发送给LLM的历史长度，避免无限增长
    MAX_HISTORY_MESSAGES_FOR_LLM: int = 12
    MAX_HISTORY_CHARS_FOR_LLM: int = 4000

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
