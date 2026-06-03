# 个人知识库管理系统

基于 **FastAPI + LangChain + RAG** 技术栈构建的个人知识库问答系统。支持上传本地文档，通过自然语言与知识库进行对话。

## 功能特性

- **文档管理** — 支持 PDF、Word（.docx）、Markdown、TXT 格式文档的上传、解析、分块与向量化存储
- **RAG 智能问答** — 基于检索增强生成（Retrieval-Augmented Generation），结合向量检索与 LLM 生成精准回答
- **连续对话** — 支持多轮对话，AI 能记住上下文，支持会话历史查看与清除
- **查询改写** — 自动将追问中的指代消解（如"它的原理是什么？" → "Transformer 的原理是什么？"），提升检索准确率
- **引用来源** — 回答中附带参考文档片段，方便溯源验证
- **双端交互** — FastAPI 后端提供 RESTful API，Streamlit 前端提供 Web UI

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit 前端                      │
│              (文档管理 + 知识库对话 UI)                │
└───────────────────────┬─────────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────────┐
│                   FastAPI 后端                        │
│  ┌─────────────────┐  ┌──────────────────────────┐  │
│  │   文档管理 API    │  │     对话 API (带记忆)     │  │
│  │  /api/documents  │  │      /api/chat/*         │  │
│  └────────┬────────┘  └───────────┬──────────────┘  │
└───────────┼───────────────────────┼──────────────────┘
            │                       │
┌───────────▼───────────────────────▼──────────────────┐
│                   LangChain 层                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Document │  │  Text    │  │   RunnableWith   │   │
│  │ Loaders  │  │ Splitter │  │ MessageHistory   │   │
│  └──────────┘  └──────────┘  └──────────────────┘   │
└───────────────────┬──────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────┐
│                    数据层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   ChromaDB   │  │  DeepSeek    │  │  bge-small │ │
│  │  (向量存储)   │  │  (LLM)       │  │  (Embedding)│ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
└──────────────────────────────────────────────────────┘
```

### 核心流程

```
用户提问 → 查询改写(消解指代) → 向量检索(ChromaDB) → 构建上下文
    → 拼接提示词(含历史消息) → LLM生成回答 → 返回答案+引用来源
```

## 项目结构

```
personal-knowledge-base/
├── app/
│   ├── api/
│   │   ├── chat.py              # 对话 API（带会话历史）
│   │   └── documents.py         # 文档管理 API
│   ├── chains/
│   │   └── rag_chain.py         # RAG 处理链路（检索→改写→生成）
│   ├── core/
│   │   └── config.py            # 全局配置（环境变量管理）
│   ├── document_loaders/
│   │   └── loaders.py           # 文档解析加载器（PDF/DOCX/MD/TXT）
│   ├── schemas/
│   │   ├── chat.py              # 对话请求/响应模型
│   │   └── document.py          # 文档信息模型
│   ├── storage/
│   │   └── vector_store.py      # ChromaDB 向量存储与检索
│   └── main.py                  # FastAPI 应用入口
├── frontend/
│   ├── app.py                   # Streamlit 前端页面
│   └── api_client.py            # 后端 API 客户端封装
├── data/                        # 运行时数据（不提交到 Git）
│   ├── chroma/                  # ChromaDB 持久化数据
│   ├── documents/               # 上传的原始文档
│   └── chat_history/            # 会话历史 JSON 文件
├── .env.example                 # 环境变量模板
├── requirements.txt             # Python 依赖
└── PROJECT_PLAN.md              # 项目规划文档
```

## 技术选型

| 组件 | 技术方案 |
|------|----------|
| Web 框架 | FastAPI |
| 前端 UI | Streamlit |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| Embedding 模型 | BAAI/bge-small-zh-v1.5 |
| LLM | DeepSeek API（deepseek-chat） |
| 文档解析 | PyPDFLoader / UnstructuredWordDocumentLoader / TextLoader |
| 文本分块 | RecursiveCharacterTextSplitter（chunk_size=500, overlap=50） |

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repo-url>
cd personal-knowledge-base

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key：

```bash
cp .env.example .env
```

编辑 `.env` 文件：

```
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 启动服务

**启动后端（FastAPI）：**

```bash
uvicorn app.main:app --reload --port 8000
```

后端默认运行在 `http://127.0.0.1:8000`，API 文档访问 `http://127.0.0.1:8000/docs`。

**启动前端（Streamlit）：**

```bash
streamlit run frontend/app.py
```

前端默认运行在 `http://127.0.0.1:8501`。

### 4. 使用

1. 打开浏览器访问 Streamlit 前端
2. 在"文档管理"页面上传 PDF/DOCX/MD/TXT 文档
3. 切换到"知识库对话"页面，开始提问

## API 接口

### 文档管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents/upload` | 上传文档 |
| GET | `/api/documents` | 获取文档列表 |
| DELETE | `/api/documents/{file_id}` | 删除文档 |

### 对话

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat/send` | 发送消息（带会话历史） |
| GET | `/api/chat/history` | 获取会话历史 |
| DELETE | `/api/chat/clear` | 清除会话历史 |
| GET | `/api/chat/sessions` | 获取会话列表 |

### 健康检查

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |

## 配置说明

所有配置项位于 `app/core/config.py`，可通过 `.env` 文件覆盖：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | — | DeepSeek API 密钥 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 模型名称 |
| `EMBEDDING_MODEL` | `BAAI/bge-small-zh-v1.5` | Embedding 模型 |
| `CHUNK_SIZE` | `500` | 文档分块大小（字符数） |
| `CHUNK_OVERLAP` | `50` | 分块重叠字符数 |
| `REWRITE_ENABLED` | `True` | 是否启用查询改写 |
| `MAX_HISTORY_MESSAGES_FOR_LLM` | `12` | 发送给 LLM 的最大历史消息数 |
| `MAX_HISTORY_CHARS_FOR_LLM` | `4000` | 发送给 LLM 的最大历史字符数 |

## 开发计划

- [x] Phase 1: 基础框架搭建（FastAPI + 配置管理）
- [x] Phase 2: 文档管理核心（解析 + 分块 + 向量存储）
- [x] Phase 3: RAG 对话实现（检索 + 生成 + 会话记忆）
- [ ] Phase 4: 完善与优化（错误处理、日志、测试）
