# ============================================
# Day 26：持久化 Checkpointer（面试加分项）
# 作用：把每一步的 State 存下来 → 断点续跑 / 多会话隔离(thread_id)
# ============================================
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    count: int
    log: list[str]

def step_a(state: State):
    return {"count": state["count"] + 1, "log": state["log"] + [f"A完成, count={state['count']+1}"]}

def step_b(state: State):
    return {"count": state["count"] * 10, "log": state["log"] + [f"B完成, count={state['count']*10}"]}

builder = StateGraph(State)
builder.add_node("a", step_a)
builder.add_node("b", step_b)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("b", END)

# 关键：编译时挂上 checkpointer
graph = builder.compile(checkpointer=MemorySaver())

# thread_id = 会话ID：同一个 id 状态共享，不同 id 完全隔离
config = {"configurable": {"thread_id": "user_001"}}
result = graph.invoke({"count": 1, "log": []}, config=config)
print("最终结果：", result)

# 查看存档：每一步的 State 都被记录下来了
print("\n📦 检查点历史（时间旅行）：")
for snapshot in graph.get_state_history(config):
    print(f"  节点={snapshot.next or ['结束']}, count={snapshot.values.get('count')}")

# 多会话隔离演示
config2 = {"configurable": {"thread_id": "user_002"}}
result2 = graph.invoke({"count": 100, "log": []}, config=config2)
print("\n另一个会话 user_002 的结果（互不影响）：", result2["count"])
