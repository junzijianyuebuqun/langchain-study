# ============================================
# Day 14：RAG 第三步 —— 组装完整 RAG 问答链（⭐ 面试必考）
# 流程：提问 → 向量库检索 → 片段拼进 Prompt → 模型生成答案
# ============================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------
# 1. 加载已建好的向量库（Day 13 存的本地的，不用再向量化）
# --------------------------------------------
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(
    persist_directory="week03/chroma_db",
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # 每次检索3片

# --------------------------------------------
# 2. RAG 专用 Prompt：限制模型只能根据资料回答（防幻觉的关键！）
# --------------------------------------------
prompt = ChatPromptTemplate.from_template("""
你是公司制度问答助手。只根据以下资料回答问题，如果资料里没有答案，就说"资料中没有提到"。

资料：
{context}

问题：{question}
""")

model = ChatDeepSeek(model="deepseek-chat")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# --------------------------------------------
# 3. 组装 RAG 链（画出数据流向，面试会让你手写）：
#    question 同时走两路：
#      一路给 retriever 检索 → format_docs 拼成文本 → 填入 {context}
#      一路 RunnablePassthrough 原样保留 → 填入 {question}
#    然后 → prompt → model → parser
# --------------------------------------------
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# --------------------------------------------
# 4. 测试三个问题
# --------------------------------------------
for q in ["年假有几天？", "请7天假需要什么流程？", "公司食堂几点开饭？"]:
    print(f"\n❓ {q}")
    print(f"💬 {rag_chain.invoke(q)}")
