# ============================================
# Day 28-30：本周项目 —— 自动写作 Agent（LangGraph 综合实战）
# 流程：生成大纲 → 写正文 → 自我审核 → 不达标重写(条件循环) → 人工确认 → 输出终稿
# 覆盖考点：State/Node/条件边/循环/interrupt/checkpointer
# ============================================
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

model = ChatDeepSeek(model="deepseek-chat", temperature=0.7)

class State(TypedDict):
    topic: str
    outline: str
    article: str
    score: int
    attempts: int       # 重写次数（防止死循环！）
    approved: bool

# ---------- 节点 ----------
def outline_node(state: State):
    r = model.invoke(f"为主题「{state['topic']}」列一个3点式文章大纲，每点一句话")
    print(f"📋 大纲：{r.content[:60]}...")
    return {"outline": r.content}

def write_node(state: State):
    r = model.invoke(
        f"根据大纲写一段200字以内的短文：\n{state['outline']}\n要求：生动有趣"
    )
    print(f"✍️  第{state['attempts']+1}次写作完成（{len(r.content)}字）")
    return {"article": r.content, "attempts": state["attempts"] + 1}

def review_node(state: State):
    r = model.invoke(
        f"给文章打分（1-10），第一行只写数字，之后写一句话理由：\n{state['article']}"
    )
    score = int("".join(ch for ch in r.content.split("\n")[0] if ch.isdigit()) or "0")
    print(f"🔍 审核：{score}分 —— {r.content.split(chr(10))[-1][:40]}")
    return {"score": score}

def human_node(state: State):
    decision = interrupt({"msg": "文章已达标，请人工终审", "article": state["article"]})
    return {"approved": decision["approved"]}

# ---------- 条件路由：达标且没超次数→人工终审；否则重写；超3次→强制送审 ----------
def route_after_review(state: State) -> str:
    if state["score"] >= 8 or state["attempts"] >= 3:
        return "human"
    return "write"

# ---------- 组装 ----------
builder = StateGraph(State)
builder.add_node("outline", outline_node)
builder.add_node("write", write_node)
builder.add_node("review", review_node)
builder.add_node("human", human_node)
builder.add_edge(START, "outline")
builder.add_edge("outline", "write")
builder.add_edge("write", "review")
builder.add_conditional_edges("review", route_after_review, {"write": "write", "human": "human"})
builder.add_edge("human", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "writing_001"}}

print("=== 第一阶段：AI 自动写作+审核循环 ===")
graph.invoke({"topic": "为什么要学 LangChain", "outline": "", "article": "",
              "score": 0, "attempts": 0, "approved": False}, config=config)

print("\n=== 第二阶段：人工终审通过，输出终稿 ===")
final = graph.invoke(Command(resume={"approved": True}), config=config)
print(f"\n📄 终稿（共写了{final['attempts']}稿，审核{final['score']}分）：")
print(final["article"])
