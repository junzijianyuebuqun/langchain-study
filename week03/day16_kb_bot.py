# ============================================
# Day 16-17：本周项目 —— 知识库问答机器人（完整版）
# 功能：RAG 问答 + 标注引用来源 + 不知道就说不知道
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
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory="week03/chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
model = ChatDeepSeek(model="deepseek-chat")

prompt = ChatPromptTemplate.from_template("""
你是公司制度问答助手。只根据以下资料回答问题，资料里没有就说"资料中没有提到"。
回答末尾不需要自己编来源，系统会自动附上。

资料：
{context}

问题：{question}
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# RunnableParallel：同时输出"答案"和"检索到的原文片段"（用于标注来源）
rag_chain = RunnableParallel(
    answer=(
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | model | StrOutputParser()
    ),
    sources=retriever,  # 把检索到的原始片段也带出来
)

questions = ["年假有多少天？怎么休？", "突发疾病怎么请假？", "公司团建多久一次？"]

for q in questions:
    result = rag_chain.invoke(q)
    print(f"\n❓ {q}")
    print(f"💬 {result['answer']}")
    print("📚 引用来源：")
    for i, doc in enumerate(result["sources"], 1):
        first_line = doc.page_content.strip().split("\n")[0][:40]
        print(f"   [{i}] {first_line}...")
    print("-" * 50)
