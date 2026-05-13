from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.core.config import settings

embedding_fn = SentenceTransformerEmbeddings(
    model_name=settings.EMBEDDING_MODEL
)

def get_vector_store():
    return Chroma(
        persist_directory=settings.CHROMA_PERSIST_DIR,
        embedding_function=embedding_fn
    )

def split_documents(documents: List[Document]) -> List[Document]:
    # 接收一个Document对象列表，
    # 并使用RecursiveCharacterTextSplitter将每个文档分割成更小的块。
    # 分块的大小和重叠部分由配置中的CHUNK_SIZE和CHUNK_OVERLAP参数控制。
    # 函数返回一个新的Document对象列表
    # 其中每个Document对象代表一个分块后的文本片段，保留了原始文档的元数据。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len 
        # 使用内置的len函数来计算文本长度，以便根据字符数进行分块
    )
    return text_splitter.split_documents(documents)

def add_documents_to_vector_store(documents: List[Document], file_id: str = None, filename: str = None) -> int:
   #传入一个Document对象列表，以及可选的文件ID和文件名参数，返回分块的数量

    chunks = split_documents(documents)
    #返回一个新的Document对象列表，其中每个Document对象代表一个分块后的文本片段，保留了原始文档的元数据。
    vector_store = get_vector_store()
    #获取一个Chroma向量数据库实例，用于存储分块后的文本数据和相关的元数据。
    
    #生成唯一的ID列表
    # 每个ID由文件ID和分块索引组成，确保每个分块在向量数据库中都有一个唯一标识符
    #如果提供了文件ID，则每个分块的ID格式为"{file_id}_{i}"，如31548_0、31548_1
    # 其中i是分块的索引；如果没有提供文件ID，则ID仅由分块索引组成。
    ids = [f"{file_id}_{i}" for i in range(len(chunks))]
    metadatas = [] 
    for i, doc in enumerate(chunks):
        meta = dict(doc.metadata) if doc.metadata else {}
        if file_id:
            meta["file_id"] = file_id
        if filename:
            meta["filename"] = filename
        metadatas.append(meta)
    
    texts = [doc.page_content for doc in chunks]
    
    vector_store.add_texts(  #把分块后的文本数据和元数据添加到向量数据库中
        texts=texts,    # 分块后的文本内容列表
        ids=ids,         # 分块的唯一ID列表
        metadatas=metadatas # 分块的元数据列表
    )
    return len(chunks)# 返回分块的数量

def delete_all_documents():
    vector_store = get_vector_store()
    vector_store.delete_collection()

def get_retriever():
    vector_store = get_vector_store()
    #拿到已经绑定了持久化目录和嵌入函数的Chroma向量数据库实例
    # 并将其转换为一个检索器对象。

    return vector_store.as_retriever()
 # 返回的检索器对象可以用于根据查询从向量数据库中检索相关的文本分块和元数据，

#相当于一个“按问题取相关文档”的接口
#它的输入是一个查询字符串，输出是与查询相关的文本分块和元数据列表。
#它的价值在于解耦：
# 上层代码不需要关心 Chroma 的底层细节（例如向量存储实现、搜索接口名）
# 只需要调用 retriever 做“按问题取相关文档”。
# 这让后续替换向量库或调整检索策略时，改动集中在存储层而不是业务层。