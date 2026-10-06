# ============================================
# Day 27：Human-in-the-loop 人工介入（演示版）
# 场景：AI 写好文案 → 暂停 → 人工审批 → 通过才发布
# 真实场景用 input() 等待人输入，这里用演示模式自动审批
# ============================================
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

model = ChatDeepSeek(model="deepseek-chat")

class State(TypedDict):
    topic: str
    draft: str
    approved: bool

def write_node(state: State):
    r = model.invoke(f"为「{state['topic']}」写一句宣传语")
    print(f"✍️  AI 写好了：{r.content}")
    return {"draft": r.content}

def human_review_node(state: State):
    """人工审核节点：interrupt 会暂停整个流程，等待外部输入"""
    decision = interrupt({
        "message": "请审核这句宣传语",
        "draft": state["draft"],
    })
    # 流程恢复后，decision = 外部传入的值
    return {"approved": decision["approved"]}

def publish_node(state: State):
    if state["approved"]:
        print(f"🚀 已发布：{state['draft']}")
    else:
        print(f"❌ 被驳回，不发布：{state['draft']}")
    return {}

builder = StateGraph(State)
builder.add_node("write", write_node)
builder.add_node("review", human_review_node)
builder.add_node("publish", publish_node)
builder.add_edge(START, "write")
builder.add_edge("write", "review")
builder.add_edge("review", "publish")
builder.add_edge("publish", END)

# 人工介入必须配 checkpointer（暂停时要把状态存下来）
graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "review_001"}}

# 第一次运行：跑到 review 节点会暂停
print("=== 第一次运行（会在人工审核处暂停）===")
graph.invoke({"topic": "LangGraph 框架", "draft": "", "approved": False}, config=config)

# 模拟人工审批：批准（真实场景这里是等用户在前端点"通过"按钮）
print("\n=== 人工点击了【通过】，流程继续 ===")
final = graph.invoke(Command(resume={"approved": True}), config=config)
print("最终状态：", final)
