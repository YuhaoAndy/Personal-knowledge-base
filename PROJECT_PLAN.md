# 个人知识库管理系统 - 项目规划

## 1. 项目概述

基于 FastAPI + LangChain + RAG 技术栈，构建一个支持本地文档管理的个人知识库问答系统。

### 技术选型
| 组件 | 技术方案 |
|------|----------|
| Web 框架 | FastAPI |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB |
| Embedding 模型 | BAAI/bge-small-zh-v1.5 |
| LLM | DeepSeek API |

---

## 2. 核心功能模块

### 2.1 文档管理模块
- **文档上传**：支持 PDF、Word(.docx)、Markdown、TXT 等格式
- **文档解析**：提取文本内容并分块处理
- **向量存储**：文档内容向量化并存入 ChromaDB
- **文档管理**：查看已导入的文档列表、删除文档

### 2.2 RAG 对话模块
- **知识检索**：基于用户问题检索相关文档片段
- **答案生成**：结合检索结果和 LLM 生成回答
- **引用来源**：显示回答引用的文档来源
- **连续对话**：AI 记住之前的对话内容，理解上下文

---

## 3. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                     前端层 (待定)                        │
└───────────────────────┬─────────────────────────────────┘
                        │ HTTP/WebSocket
┌───────────────────────▼─────────────────────────────────┐
│                      FastAPI Server                      │
│  ┌─────────────┐  ┌───────────────────────┐  │
│  │ 文档管理API │  │ 对话API (带记忆)       │  │
│  └─────────────┘  └───────────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                     LangChain Layer                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ DocumentLoa │  │  TextSplitt │  │  Retriever     │  │
│  │ ders        │  │ er          │  │                 │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                     数据层                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ ChromaDB    │  │ DeepSeek    │  │ bge-small-zh   │  │
│  │ (向量存储)  │  │ (LLM)       │  │ (Embedding)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 4. API 设计

### 4.1 文档管理
| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/documents/upload` | 上传文档 |
| GET | `/api/documents` | 获取文档列表 |
| DELETE | `/api/documents/{doc_id}` | 删除文档 |

### 4.2 对话（带记忆）
| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/chat/send` | 发送对话消息（AI 有记忆） |
| GET | `/api/chat/history` | 获取对话历史 |
| DELETE | `/api/chat/clear` | 清除对话历史 |

> 实现方式：使用 `RunnableWithMessageHistory` + `BaseChatMessageHistory`

---

## 5. 开发计划

### Phase 1: 基础框架搭建 ✅ 已完成
- [x] 项目初始化，目录结构设计
- [x] 环境配置与依赖安装
- [x] FastAPI 基础服务搭建
- [x] 配置管理（环境变量）

### Phase 2: 文档管理核心 ✅ 已完成
- [x] 文档解析加载器
- [x] 文本分块处理
- [x] Embedding 模型
- [x] ChromaDB 向量存储
- [x] 文档 API

### Phase 3: RAG 对话实现 ✅ 已完成
- [x] LangChain RAG 链构建
- [x] DeepSeek API 集成
- [x] 对话 API 实现（带记忆）
- [x] 引用来源功能

---

## 6. API 接口文档

### 6.1 文档管理

| 序号 | 模块 | 功能 | 关键组件 |
|------|------|------|----------|
| 1 | 文档解析加载器 | 解析 PDF/Word/MD/TXT | PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, MarkdownLoader |
| 2 | 文本分块处理 | 拆分长文档为小块 | RecursiveCharacterTextSplitter |
| 3 | Embedding 模型 | 文本向量化 | SentenceTransformerEmbeddingFunction |
| 4 | ChromaDB 向量存储 | 存储和检索向量 | Chroma |
| 5 | 文档 API | 上传/删除文档 | FastAPI |

#### 文档上传流程

```
用户上传文件
    │
    ▼
1. FastAPI 接收文件
   POST /api/documents/upload
   → 保存到 data/documents/ 目录
    │
    ▼
2. 文档解析（根据文件类型选择 Loader）
   ├── .pdf  → PyPDFLoader
   ├── .docx → UnstructuredWordDocumentLoader
   ├── .md   → MarkdownLoader
   └── .txt  → TextLoader
   → 读取文本内容，返回 Document 对象
    │
    ▼
3. 文本分块
   RecursiveCharacterTextSplitter
   → 按 CHUNK_SIZE=500 拆分，每块重叠 CHUNK_OVERLAP=50
   → 返回多个 Document 对象
    │
    ▼
4. 向量存储
   Chroma.from_documents(
       documents=chunks,
       embedding=embedding_fn,
       collection_name="documents"
   )
   → 文本向量化，存入 ChromaDB
```

#### 实现细节：

**① 文档加载器 (`app/document_loaders/loaders.py`)**
```python
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import MarkdownLoader
```

**② 文本分块 (`RecursiveCharacterTextSplitter`)**
- `CHUNK_SIZE = 500` - 每块字数
- `CHUNK_OVERLAP = 50` - 块之间重叠字数

**③ Embedding 模型**
```python
from chromadb.utils import embedding_functions

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)
```
> 注意：需要设置 `HF_ENDPOINT=https://hf-mirror.com`

**④ ChromaDB 向量存储 (`app/storage/vector_store.py`)**
- `Chroma.from_documents()` - 文档转向量并存储
- `Chroma.as_retriever()` - 创建检索器
- `Chroma.delete_collection()` - 删除集合

**⑤ 预计新增依赖**
```
langchain
langchain-community
chromadb
sentence-transformers
pypdf
python-docx
markdown
```

---

### Phase 3: RAG 对话实现
- [ ] LangChain RAG 链构建
- [ ] DeepSeek API 集成
- [ ] 对话 API 实现（带记忆）
- [ ] 引用来源功能

### Phase 4: 完善与优化
- [ ] 错误处理与日志
- [ ] 性能优化
- [ ] 单元测试

---

## 6. 目录结构

```
personal-knowledge-base/
├── app/
│   ├── api/
│   │   ├── documents.py    # 文档管理 API
│   │   └── chat.py         # 对话 API
│   ├── core/
│   │   └── config.py       # 配置管理
│   ├── chains/
│   │   ├── rag_chain.py    # RAG 链
│   │   └── memory.py       # 记忆模块
│   ├── document_loaders/
│   │   └── loaders.py      # 文档加载器
│   ├── storage/
│   │   └── vector_store.py # ChromaDB 向量存储
│   ├── schemas/
│   │   └── ...              # Pydantic 模型
│   ├── utils/
│   │   └── ...              # 工具函数
│   └── main.py              # 应用入口
├── data/
│   ├── chroma/              # ChromaDB 数据
│   └── documents/          # 上传文档
├── tests/                   # 测试
├── .env                     # 环境变量
├── requirements.txt         # 依赖
├── PROJECT_PLAN.md          # 项目规划
└── 开发问题记录.md            # 开发问题记录
```

