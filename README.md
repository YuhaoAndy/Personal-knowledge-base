# 个人知识库管理系统

基于 **FastAPI + LangChain + RAG** 技术栈构建的个人知识库问答系统。支持上传本地文档，通过自然语言与知识库进行智能对话。

## 功能特性

- **文档管理**：支持 PDF、Word（.docx）、Markdown、TXT 格式文档的上传、解析与删除
- **智能问答**：基于 RAG（检索增强生成）技术，结合文档内容与 LLM 回答问题
- **多轮对话**：AI 记住对话上下文，支持连续追问
- **引用溯源**：每个回答附带引用的文档片段和来源文件名
- **多会话管理**：支持创建多个独立会话，不同话题互不干扰

## 技术栈

| 组件 | 技术方案 |
|------|----------|
| Web 框架 | FastAPI |
| RAG 框架 | LangChain |
| 向量数据库 | ChromaDB（本地持久化） |
| Embedding 模型 | BAAI/bge-small-zh-v1.5 |
| LLM | DeepSeek API |
| 前端 | Streamlit |
| 文档解析 | PyPDF / Unstructured / TextLoader |

## 快速开始

### 前置要求

- Python 3.10+
- DeepSeek API Key

### 安装

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd personal-knowledge-base

# 2. 创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 配置

```bash
# 复制环境变量模板
cp .env.example .env

# 编辑 .env 文件，填入你的 DeepSeek API Key
DEEPSEEK_API_KEY=your_api_key_here
```

### 启动

```bash
# 1. 启动后端服务
uvicorn app.main:app --reload

# 2. 新开终端，启动前端界面
streamlit run frontend/app.py
```

启动后：
- 后端 API：http://localhost:8000
- API 文档：http://localhost:8000/docs
- 前端界面：http://localhost:8501

## 项目结构

```
personal-knowledge-base/
├── app/                          # 后端核心代码
│   ├── main.py                   # FastAPI 入口，路由注册
│   ├── api/
│   │   ├── chat.py               # 对话 API（带记忆）
│   │   └── documents.py          # 文档管理 API
│   ├── chains/
│   │   └── rag_chain.py          # RAG 核心链路（检索→增强→生成）
│   ├── core/
│   │   └── config.py             # 全局配置
│   ├── document_loaders/
│   │   └── loaders.py            # 文档解析器
│   ├── schemas/
│   │   ├── chat.py               # 对话请求/响应模型
│   │   └── document.py           # 文档信息模型
│   └── storage/
│       └── vector_store.py       # 向量存储（ChromaDB 封装）
├── frontend/
│   ├── app.py                    # Streamlit 前端主程序
│   └── api_client.py             # 后端 API 客户端封装
├── data/                         # 运行时数据
│   ├── chroma/                   # ChromaDB 持久化目录
│   ├── documents/                # 上传的源文件
│   └── chat_history/             # 会话历史
├── .env.example                  # 环境变量模板
├── requirements.txt              # Python 依赖
└── README.md
```

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
| POST | `/api/chat/send` | 发送消息 |
| GET | `/api/chat/history` | 获取对话历史 |
| DELETE | `/api/chat/clear` | 清空对话历史 |
| GET | `/api/chat/sessions` | 列出所有会话 |

## 系统架构

```
用户上传文档 → 文档解析 → 文本分块 → 向量化 → ChromaDB 存储
                                                      ↓
用户提问 → 检索相关文档片段 → 组装 Prompt → LLM 生成回答 → 返回结果+引用来源
```

## 核心流程

### 文档上传流程

```
上传文件 → 保存到本地 → 按类型解析（PDF/DOCX/MD/TXT）
         → 文本分块（每块 500 字符，重叠 50 字符）
         → 向量化 → 存入 ChromaDB
```

### RAG 问答流程

```
用户问题 + 历史消息 → 构建检索查询（含最近 3 个历史问题）
                   → 从 ChromaDB 检索相关文档
                   → 裁剪历史消息（最多 12 条 / 4000 字符）
                   → 组装 Prompt（System + 历史 + 问题）
                   → 调用 DeepSeek API
                   → 返回回答 + 引用来源
```

## 配置说明

核心配置项（通过 `.env` 文件或环境变量设置）：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | - |
| `DEEPSEEK_BASE_URL` | API 地址 | https://api.deepseek.com |
| `DEEPSEEK_MODEL` | 模型名称 | deepseek-chat |
| `EMBEDDING_MODEL` | Embedding 模型 | BAAI/bge-small-zh-v1.5 |
| `CHUNK_SIZE` | 文档分块大小 | 500 |
| `CHUNK_OVERLAP` | 分块重叠字符数 | 50 |
| `MAX_HISTORY_MESSAGES_FOR_LLM` | 历史消息最大条数 | 12 |
| `MAX_HISTORY_CHARS_FOR_LLM` | 历史消息最大字符数 | 4000 |

## 开发计划

- [x] Phase 1：基础框架搭建
- [x] Phase 2：文档管理核心
- [x] Phase 3：RAG 对话实现
- [ ] Phase 4：检索质量优化（重排、过滤）
- [ ] Phase 5：工程化提升（测试、日志、安全）
