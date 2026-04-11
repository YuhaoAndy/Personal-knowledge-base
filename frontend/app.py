from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st

from api_client import ApiClient


def build_client() -> ApiClient:
    base_url = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
    return ApiClient(base_url=base_url)


def render_header() -> None:
    st.title("个人知识库")
    st.caption("文档管理与知识问答")


def show_documents_page(client: ApiClient) -> None:
    st.subheader("文档管理")

    upload_col, help_col = st.columns([2, 1])
    with upload_col:
        uploaded_file = st.file_uploader(
            "上传文档（PDF / DOCX / MD / TXT）",
            type=["pdf", "docx", "md", "txt"],
            help="上传后会自动解析、切分并写入向量库。",
        )

    with help_col:
        st.info("建议单文件大小先控制在 10MB 以内，便于本地快速处理。")

    if st.button("上传文档", use_container_width=False, type="primary"):
        if not uploaded_file:
            st.warning("请先选择文件")
        else:
            try:
                result = client.upload_document(
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
                st.success(
                    f"上传成功：{result['filename']}（切分块数：{result['chunk_count']}）"
                )
                st.rerun()
            except requests.HTTPError as exc:
                st.error(f"上传失败：{exc.response.text}")
            except Exception as exc:
                st.error(f"上传失败：{exc}")

    st.divider()
    st.markdown("### 已入库文档")

    try:
        docs = client.list_documents()
    except Exception as exc:
        st.error(f"获取文档列表失败: {exc}")
        return

    if not docs:
        st.info("当前没有文档")
        return

    st.dataframe(
        [
            {
                "文件名": doc.get("filename", ""),
                "文档ID": doc.get("id", ""),
                "切分块数": doc.get("chunk_count", 0),
            }
            for doc in docs
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 删除文档")
    for doc in docs:
        with st.expander(f"{doc.get('filename', '')} | {doc.get('id', '')}"):
            st.write(f"切分块数：{doc.get('chunk_count', 0)}")
            if st.button("删除该文档", key=f"del_{doc.get('id')}"):
                try:
                    client.delete_document(doc["id"])
                    st.success(f"已删除：{doc.get('filename')}")
                    st.rerun()
                except requests.HTTPError as exc:
                    st.error(f"删除失败：{exc.response.text}")
                except Exception as exc:
                    st.error(f"删除失败：{exc}")


def show_chat_page(client: ApiClient) -> None:
    st.subheader("知识库对话")

    session_id = st.text_input(
        "会话 ID",
        value=st.session_state.get("session_id", "default"),
        help="不同会话 ID 对应不同对话历史。",
    )
    st.session_state["session_id"] = session_id.strip() or "default"

    try:
        sessions = client.list_sessions()
        if sessions:
            with st.expander("📋 会话列表"):
                for sid in sessions:
                    col_sid, col_btn = st.columns([3, 1])
                    with col_sid:
                        st.write(f"**{sid}**")
                    with col_btn:
                        if st.button("切换", key=f"switch_{sid}"):
                            st.session_state["session_id"] = sid
                            st.rerun()
    except Exception:
        pass

    col1, col2 = st.columns([1, 1])
    with col2:
        if st.button("清空历史", use_container_width=True):
            try:
                client.clear_history(st.session_state["session_id"])
                st.success("会话历史已清空")
                st.rerun()
            except Exception as exc:
                st.error(f"清空失败：{exc}")

    with col1:
        if st.button("刷新历史", use_container_width=True):
            st.rerun()

    try:
        history_resp = client.get_history(st.session_state["session_id"])
        messages: List[Dict[str, Any]] = history_resp.get("messages", [])
    except Exception as exc:
        st.error(f"读取历史失败：{exc}")
        messages = []

    if not messages:
        st.info("当前会话暂无历史消息，可以直接开始提问。")

    for msg in messages:
        role = "user" if msg.get("type") == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.get("content", ""))

    user_input = st.chat_input("请输入问题")
    if not user_input:
        return

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.info("正在检索并生成答案...")
        try:
            response = client.send_message(user_input, st.session_state["session_id"])
            answer = response.get("answer", "")
            sources = response.get("sources", [])

            placeholder.write(answer)
        except requests.HTTPError as exc:
            placeholder.error(f"请求失败：{exc.response.text}")
        except Exception as exc:
            placeholder.error(f"请求失败：{exc}")


def main() -> None:
    st.set_page_config(
        page_title="个人知识库",
        page_icon="📚",
        layout="wide",
    )

    client = build_client()
    render_header()

    page = st.sidebar.radio("导航", ["知识库对话", "文档管理"], index=0)
    st.sidebar.caption("后端地址")
    st.sidebar.code(os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000"))

    try:
        requests.get(f"{os.getenv('BACKEND_BASE_URL', 'http://127.0.0.1:8000').rstrip('/')}/health", timeout=5)
        st.sidebar.success("后端连接正常")
    except Exception:
        st.sidebar.error("后端不可达，请先启动 FastAPI")

    if page == "文档管理":
        show_documents_page(client)
    else:
        show_chat_page(client)


if __name__ == "__main__":
    main()
