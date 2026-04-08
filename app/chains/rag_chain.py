from typing import List
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from app.core.config import settings
from app.storage.vector_store import get_retriever

llm = ChatOpenAI(
    model=settings.DEEPSEEK_MODEL,
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    temperature=0.7
)

def get_conversation_history() -> ChatMessageHistory:
    return ChatMessageHistory()

def create_rag_chain():
    retriever = get_retriever()

    prompt = ChatPromptTemplate.from_template(
        """你是一个知识库问答助手。请根据以下参考文档回答用户的问题。
        
参考文档：
{context}

用户问题：{question}

请给出回答，如果参考文档中没有相关信息，请说明"根据当前知识库无法回答该问题"。"""
    )

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

rag_chain = create_rag_chain()
