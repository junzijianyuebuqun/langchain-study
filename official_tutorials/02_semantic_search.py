# ============================================================
# 官网教程 02：语义搜索（Semantic search over a PDF）
# 官方地址：https://python.langchain.com/docs/tutorials/retrievers/
# 所属模块：LangChain（集成包: document_loaders + embeddings + vectorstores）
# 官方目标：文档加载器 + 嵌入模型 + 向量库 构建 PDF 语义搜索引擎
#
# 【与官网的对应关系】
#   官网加载 Nike 财报 PDF → 这里用 reportlab 生成中文 PDF（流程一致）
#   官网流程：Load → Split → Embed → Store → Query 逐步对应
# ============================================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 官网无此行，国内下载模型用

# --- 第0步：生成示例 PDF（代替官网下载 Nike 财报） ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

PDF_PATH = "official_tutorials/company_policy.pdf"
if not os.path.exists(PDF_PATH):
    # 注册中文字体（用系统自带的微软雅黑）
    pdfmetrics.registerFont(TTFont("msyh", "C:/Windows/Fonts/msyh.ttc"))
    c = canvas.Canvas(PDF_PATH, pagesize=A4)
    c.setFont("msyh", 11)
    with open("week03/company_policy.txt", encoding="utf-8") as f:
        y = 800
        for line in f.readlines():
            if y < 50:
                c.showPage(); c.setFont("msyh", 11); y = 800
            c.drawString(50, y, line.strip()); y -= 20
    c.save()
    print(f"已生成示例 PDF：{PDF_PATH}\n")

# --- 第1步 Load：官网用 PyPDFLoader 加载 PDF ---
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"【Load】加载了 {len(docs)} 页，共 {sum(len(d.page_content) for d in docs)} 字")

# --- 第2步 Split：官网用 RecursiveCharacterTextSplitter ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
all_splits = text_splitter.split_documents(docs)
print(f"【Split】切成 {len(all_splits)} 片")

# --- 第3步 Embed + Store：官网 Embedding + VectorStore ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_store = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory="official_tutorials/chroma_pdf_db",
)
print(f"【Store】已存入向量库")

# --- 第4步 Query：官网演示 similarity_search ---
print("\n【Query】语义搜索测试：")
results = vector_store.similarity_search("请假要提前几天申请？", k=2)
for i, doc in enumerate(results, 1):
    print(f"  结果{i}：{doc.page_content[:50]}...")

# --- 官网进阶：转成 Retriever（供链使用） ---
retriever = vector_store.as_retriever(search_kwargs={"k": 2})
print("\n【Retriever】", retriever.invoke("年假") and "retriever 工作正常 ✅")
