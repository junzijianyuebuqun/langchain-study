# ============================================
# Day 12：RAG 第一步 —— 文档加载 + 切分
# 流程位置：文档 →【Loader 加载 → Splitter 切分】→ 向量化 → 入库
# ============================================
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------
# 1. 加载：把文件变成 Document 对象列表
#    Document = page_content(文本) + metadata(来源等元数据)
# --------------------------------------------
loader = TextLoader("week03/company_policy.txt", encoding="utf-8")
docs = loader.load()
print(f"加载了 {len(docs)} 个文档，第一个文档长度：{len(docs[0].page_content)} 字")

# --------------------------------------------
# 2. 切分：长文档切成小块（chunk）
#    为什么切？① 文档太长超上下文窗口 ② 片段越小检索越精准
# --------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # 每片约 200 字
    chunk_overlap=30,    # 相邻片重叠 30 字（防止一句话被切成两半）
)
chunks = splitter.split_documents(docs)

print(f"\n切成了 {len(chunks)} 片：")
for i, c in enumerate(chunks[:4]):  # 只看前4片
    print(f"\n--- 第{i}片（{len(c.page_content)}字）---")
    print(c.page_content[:80] + "...")

# --------------------------------------------
# 面试点：chunk_size 怎么选？
# 太大 → 检索不精准（一片里混着好几个主题）
# 太小 → 语义不完整（一句话被切断）
# 经验值：中文 200~500 字，overlap 取 chunk_size 的 10%~15%
# --------------------------------------------
