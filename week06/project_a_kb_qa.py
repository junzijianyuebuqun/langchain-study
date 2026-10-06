# ============================================
# Day 32-38：⭐ 求职项目 A —— 企业知识库问答系统（命令行完整版）
# 技术栈：RAG + 多轮对话记忆 + 来源引用 + 防幻觉
# 简历描述见 Week 7 文档
# ============================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, AIMessage

# ---------- 1. 加载向量库 ----------
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = Chroma(persist_directory="week03/chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
model = ChatDeepSeek(model="deepseek-chat")

# ---------- 2. 带历史消息的 RAG Prompt（多轮对话的关键！） ----------
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是公司制度问答助手。规则：
1. 只根据【资料】回答，资料里没有就说"资料中没有提到"
2. 结合【对话历史】理解追问（如"那3天以上呢"指的还是病假）

资料：
{context}"""),
    MessagesPlaceholder("history"),      # ← 对话历史插槽
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

# ---------- 3. 组装链：答案 + 来源 同时输出 ----------
rag_chain = RunnableParallel(
    answer=(
        {"context": lambda x: format_docs(retriever.invoke(x["question"])),
         "question": lambda x: x["question"],
         "history": lambda x: x["history"]}
        | prompt | model | StrOutputParser()
    ),
    sources=lambda x: retriever.invoke(x["question"]),
)

# ---------- 4. 多轮问答（记忆 = history 列表累积） ----------
history = []
questions = [
    "病假需要什么证明？",      # 第1轮
    "那工资怎么发？",          # 第2轮：追问，考验记忆
    "请10天要谁审批？",        # 第3轮
]

for q in questions:
    result = rag_chain.invoke({"question": q, "history": history})
    print(f"\n❓ {q}")
    print(f"💬 {result['answer']}")
    print(f"📚 来源：{result['sources'][0].page_content.strip().split(chr(10))[0][:30]}...")
    # 把本轮对话记入历史
    history += [HumanMessage(content=q), AIMessage(content=result["answer"])]
    print("-" * 50)
