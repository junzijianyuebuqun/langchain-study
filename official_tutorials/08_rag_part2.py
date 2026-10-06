# ============================================================
# 官网教程 08：RAG 第二部分 —— 带记忆的对话式 RAG（RAG Part 2）
# 官方地址：https://python.langchain.com/docs/tutorials/qa_chat_history/
# 所属模块：LangChain（agents）+ LangGraph（checkpointer 记忆）
# 官方目标：RAG + 对话记忆 + 多步检索
#
# 【与官网的对应关系】
#   官网核心问题1：追问"那呢？"检索会失败 → 需要把问题"上下文化"
#   官网核心问题2：记忆管理 → checkpointer + thread_id
#   官网解法：Agentic RAG（模型自己生成检索词，自动带上上下文）
# ============================================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_store = Chroma(persist_directory="official_tutorials/chroma_rag_db",
                      embedding_function=embeddings)
model = init_chat_model("deepseek-chat", model_provider="deepseek")

# --- 官网演示的坑：追问时代理检索会带上上下文 ---
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """检索公司制度资料，帮助回答问题。"""
    docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(f"[资料] {d.page_content}" for d in docs)
    return serialized, docs

agent = create_agent(
    model, tools=[retrieve_context],
    system_prompt="你是公司制度问答助手。需要时调用检索工具；"
                  "追问时要结合对话历史生成完整的检索词；"
                  "资料没有就说不知道。",
    checkpointer=MemorySaver(),   # ← 官网的记忆方案
)
config = {"configurable": {"thread_id": "rag_conv_1"}}

# 官网经典测试场景：连续追问
conversation = [
    "病假需要什么证明？",
    "那3天以上呢？",              # ← 追问：检索词必须结合上文
    "工资怎么发？",               # ← 再次追问
    "请半个月找谁批？",           # ← 需要多步理解
]

for q in conversation:
    result = agent.invoke({"messages": [{"role": "user", "content": q}]}, config=config)
    print(f"❓ {q}")
    print(f"💬 {result['messages'][-1].content[:150]}")
    print("-" * 50)
