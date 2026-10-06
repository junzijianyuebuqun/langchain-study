# ============================================
# Day 13：RAG 第二步 —— Embedding 向量化 + 存入向量库
# 流程位置：文档 → 加载 → 切分 →【Embedding → 存入 Chroma】→ 检索
# ============================================
import os
# 国内网络用镜像站下载模型（首次运行下载约100MB，之后走本地缓存）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --------------------------------------------
# 1. 加载 + 切分（Day 12 的内容）
# --------------------------------------------
docs = TextLoader("week03/company_policy.txt", encoding="utf-8").load()
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
chunks = splitter.split_documents(docs)

# --------------------------------------------
# 2. Embedding：把文本变成向量（一串数字）
#    语义相近的文本 → 向量距离近 → 所以能"按意思搜索"
#    用本地免费的中文模型 bge-small-zh，不花 API 钱
# --------------------------------------------
print("加载 Embedding 模型（首次需下载，请稍等）...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# --------------------------------------------
# 3. 向量化所有片段，存入 Chroma 向量库（持久化到本地文件夹）
# --------------------------------------------
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="week03/chroma_db",
)
print(f"✅ 已把 {len(chunks)} 个片段存入向量库")

# --------------------------------------------
# 4. 检索测试：按"意思"找最相关的片段
# --------------------------------------------
questions = ["年假有几天？", "生病了怎么请假？", "工资怎么算？"]

for q in questions:
    print(f"\n❓ 问题：{q}")
    results = vectorstore.similarity_search(q, k=2)  # 取最相关的2片
    for i, r in enumerate(results):
        print(f"  📄 相关片段{i+1}：{r.page_content[:60]}...")

# 注意第3个问题："工资"这个词文档里其实没有直接对应标题
# 但向量检索能按语义找到"病假期间工资按80%发放"——这就是语义搜索的威力！
