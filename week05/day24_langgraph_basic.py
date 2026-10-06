# ============================================
# Day 24-25：LangGraph 核心三要素 —— State / Node / Edge
# 面试点：LangGraph 和 Chain/Agent 的区别？
#   Chain：流程写死；Agent：模型全权决定；LangGraph：你画流程图，模型在节点里干活
# ============================================
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langgraph.graph import StateGraph, START, END

model = ChatDeepSeek(model="deepseek-chat")

# --------------------------------------------
# 要素1：State —— 贯穿全流程的共享数据（ TypedDict 定义结构）
# --------------------------------------------
class State(TypedDict):
    topic: str      # 用户给的主题
    draft: str      # 初稿
    review: str     # 审核意见
    approved: bool  # 是否通过

# --------------------------------------------
# 要素2：Node —— 函数，读 State → 干活 → 返回要更新的字段
# --------------------------------------------
def write_node(state: State):
    """写作节点：根据主题写初稿"""
    r = model.invoke(f"用两句话介绍{state['topic']}，风格活泼")
    print(f"✍️  写作节点产出：{r.content[:50]}...")
    return {"draft": r.content}

def review_node(state: State):
    """审核节点：给初稿打分，>=8分通过"""
    r = model.invoke(
        f"给这段文案打分（1-10）并说明理由，第一行只写数字：\n{state['draft']}"
    )
    score_text = r.content.strip().split("\n")[0]
    score = int("".join(ch for ch in score_text if ch.isdigit()) or "0")
    print(f"🔍 审核节点打分：{score} 分")
    return {"review": r.content, "approved": score >= 8}

# --------------------------------------------
# 要素3：Edge —— 连线。条件边 = 根据 State 决定走哪条路
# --------------------------------------------
def should_continue(state: State) -> str:
    """条件路由：通过→结束；不通过→回去重写（循环！）"""
    return "end" if state["approved"] else "write"

# --------------------------------------------
# 组装图
# --------------------------------------------
builder = StateGraph(State)
builder.add_node("write", write_node)
builder.add_node("review", review_node)
builder.add_edge(START, "write")                 # 开始 → 写作
builder.add_edge("write", "review")              # 写作 → 审核
builder.add_conditional_edges(                   # 审核 → 条件分支
    "review", should_continue,
    {"write": "write", "end": END},              # 不达标重写，达标结束
)
graph = builder.compile()

# --------------------------------------------
# 运行：如果第一次打分 < 8，会自动循环重写！
# --------------------------------------------
result = graph.invoke({"topic": "LangGraph", "draft": "", "review": "", "approved": False})
print("\n📄 最终成稿：", result["draft"])
print("✅ 是否通过：", result["approved"])
