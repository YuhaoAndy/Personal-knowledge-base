from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.storage.vector_store import get_retriever

MAX_HISTORY_QUESTIONS_FOR_RETRIEVAL = 3

llm = ChatOpenAI(
    model=settings.DEEPSEEK_MODEL,
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    temperature=0.7
)

PROMPT_TEMPLATE = """你是一个知识库问答助手。请根据以下参考文档回答用户的问题。

参考文档：
{context}

用户问题：{question}

回答要求：
1. 如果参考文档中有相关信息，请基于文档内容回答"
2. 如果参考文档中没有相关信息，请使用你的通用知识回答"
3. 回答要简洁、准确"""


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_prompt(question: str, context: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=question)


def get_recent_user_questions(history_messages: Optional[List[BaseMessage]], limit: int = MAX_HISTORY_QUESTIONS_FOR_RETRIEVAL) -> List[str]:
    if not history_messages:
        return []

    questions: List[str] = []
    for msg in reversed(history_messages):
        if isinstance(msg, HumanMessage):
            text = str(msg.content).strip()
            if text:
                questions.append(text)
            if len(questions) >= limit:
                break
    questions.reverse()
    return questions


def build_retrieval_query(question: str, history_messages: Optional[List[BaseMessage]]) -> str:
    current_question = question.strip()
    recent_questions = get_recent_user_questions(history_messages)

    if not recent_questions:
        return current_question

    history_block = "\n".join(f"- {q}" for q in recent_questions)
    return (
        "最近对话中的用户问题：\n"
        f"{history_block}\n"
        "当前追问：\n"
        f"{current_question}"
    )


def retrieve_documents(question: str, history_messages: Optional[List[BaseMessage]] = None) -> List[Document]:
    retriever = get_retriever()
    retrieval_query = build_retrieval_query(question, history_messages)
    return retriever.invoke(retrieval_query)


def build_sources(docs: List[Document]) -> List[dict]:
    sources = []
    for doc in docs:
        snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        sources.append({
            "content": snippet,
            "filename": doc.metadata.get("filename", "未知")
        })
    return sources


def trim_history_for_llm(history_messages: Optional[List[BaseMessage]]) -> List[BaseMessage]:
    if not history_messages:
        return []

    max_messages = max(0, settings.MAX_HISTORY_MESSAGES_FOR_LLM)
    max_chars = max(0, settings.MAX_HISTORY_CHARS_FOR_LLM)

    # 先按消息数量做窗口裁剪
    windowed = list(history_messages[-max_messages:]) if max_messages > 0 else []

    # 再按字符预算从新到旧保留，避免超长会话拖慢请求
    if max_chars <= 0:
        return windowed

    selected_reversed: List[BaseMessage] = []
    used_chars = 0
    for msg in reversed(windowed):
        text = str(msg.content)
        text_len = len(text)

        if selected_reversed and used_chars + text_len > max_chars:
            break

        selected_reversed.append(msg)
        used_chars += text_len

    selected_reversed.reverse()
    return selected_reversed


def answer_with_rag(question: str, history_messages: Optional[List[BaseMessage]] = None) -> Tuple[str, List[dict]]:
    docs = retrieve_documents(question, history_messages)
    context = format_docs(docs)
    prompt = build_prompt(question, context)

    messages: List[BaseMessage] = trim_history_for_llm(history_messages)
    messages.append(HumanMessage(content=prompt))

    response = llm.invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return answer, build_sources(docs)
