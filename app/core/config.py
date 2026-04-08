import os
from pathlib import Path
from pydantic_settings import BaseSettings

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    HF_ENDPOINT: str = "https://hf-mirror.com"
    EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma")
    DOCUMENTS_DIR: str = str(BASE_DIR / "data" / "documents")

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
