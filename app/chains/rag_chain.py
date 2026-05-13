from typing import Any, Callable, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.storage.vector_store import get_retriever

# 执行顺序（先看这 5 行）：
# 1) 用户问题进入 rag_processing_chain
# 2) prepare_chain_payload: 检索文档并构建 context
# 3) render_prompt_messages: 渲染 system/history/human 消息
# 4) generate_answer_with_sources: 调用 LLM，产出 answer + sources
# 5) build_rag_with_history_runnable: 负责会话历史自动注入与回写

MAX_HISTORY_QUESTIONS_FOR_RETRIEVAL = 3
# 在构建检索查询时，最多包含最近 3 个用户问题，平衡上下文与检索效率。

llm = ChatOpenAI(
    model=settings.DEEPSEEK_MODEL,
    api_key=settings.DEEPSEEK_API_KEY,
    base_url=settings.DEEPSEEK_BASE_URL,
    temperature=0.7
)

SYSTEM_PROMPT = (
    "你是一个知识库问答助手。请根据以下参考文档回答用户的问题。\n\n"
    "参考文档：\n{context}\n\n"
    "回答要求：\n"
    "1. 如果参考文档中有相关信息，请基于文档内容回答\n"
    "2. 如果参考文档中没有相关信息，请使用你的通用知识回答\n"
    "3. 回答要简洁、准确"
)


rag_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        SYSTEM_PROMPT,
    ),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}"),
])


"""检索相关函数"""
def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


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


#到向量库寻找相关片段
# 构建检索查询时，包含当前问题和最近的几个用户问题
# 帮助向量库理解检索意图，提升相关性。
def retrieve_documents(question: str, history_messages: Optional[List[BaseMessage]] = None) -> List[Document]:
    retriever = get_retriever()
    retrieval_query = build_retrieval_query(question, history_messages)
    return retriever.invoke(retrieval_query)


"""输出构建与历史裁剪函数"""
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


"""主链路 1/3 准备检索与提示词上下文"""
def prepare_chain_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    question = str(payload.get("question", "")).strip()
    history_messages = payload.get("chat_history", [])

    docs = retrieve_documents(question, history_messages)
    return {
        "question": question,
        "chat_history": trim_history_for_llm(history_messages),
        "docs": docs,
        "context": format_docs(docs),
    }


"""主链路 2/3 渲染发给模型的消息列表"""
def render_prompt_messages(payload: Dict[str, Any]) -> Dict[str, Any]:
    messages: List[BaseMessage] = rag_prompt_template.format_messages(
        context=payload["context"],
        question=payload["question"],
        chat_history=payload["chat_history"],
    )
    return {
        "messages": messages,
        "docs": payload["docs"],
    }


"""主链路 3/3 调用模型并构建标准输出"""
def generate_answer_with_sources(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = llm.invoke(payload["messages"])
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return {
        "answer": answer,
        "sources": build_sources(payload["docs"]),
    }


rag_processing_chain = (
    RunnableLambda(prepare_chain_payload)  
    # 输入原始问题和会话历史，输出包含检索到的文档和构建好的提示词上下文的payload
    | RunnableLambda(render_prompt_messages)
    | RunnableLambda(generate_answer_with_sources)
)


def build_rag_with_history_runnable(
    get_session_history: Callable[[str], BaseChatMessageHistory]
) -> RunnableWithMessageHistory:
    return RunnableWithMessageHistory(
        rag_processing_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
