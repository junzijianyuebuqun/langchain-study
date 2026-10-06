# ============================================
# Day 32-38：⭐ 求职项目 B —— 多工具研究型 Agent（LangGraph 编排）
# 流程：生成搜索词 → 搜索 → 总结 → 反思 →【条件边】补充搜索/出报告
# 覆盖：状态机 + 循环 + 工具 + 结构化输出 + 防死循环
# ============================================
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END

model = ChatDeepSeek(model="deepseek-chat")

# ---------- 模拟搜索工具（真实项目换成 Tavily/SerpAPI） ----------
@tool
def web_search(query: str) -> str:
    """搜索互联网获取信息。输入搜索关键词。"""
    fake_db = {
        "LangChain 1.0": "LangChain 1.0 于2025年发布，核心是 LCEL 编排和 create_agent 智能体入口，废弃了旧的 Chain API。",
        "LangGraph": "LangGraph 是 LangChain 团队的编排框架，用 StateGraph 实现状态机，支持持久化、人工介入和多Agent。",
        "RAG": "RAG 检索增强生成：文档切分向量化入库，提问时检索片段拼入prompt，解决知识过期和幻觉问题。",
    }
    for k, v in fake_db.items():
        if k.lower() in query.lower():
            return v
    return f"关于「{query}」的搜索结果：这是一个AI应用开发相关的热门技术方向。"

class State(TypedDict):
    topic: str
    queries: list[str]
    findings: list[str]
    summary: str
    need_more: bool
    rounds: int

# ---------- 节点 ----------
def plan_node(state: State):
    r = model.invoke(f"为研究主题「{state['topic']}」生成2个搜索关键词，每行一个，不要序号")
    queries = [q.strip() for q in r.content.strip().split("\n") if q.strip()][:2]
    print(f"🔎 生成搜索词：{queries}")
    return {"queries": queries}

def search_node(state: State):
    findings = list(state["findings"])
    for q in state["queries"]:
        result = web_search.invoke({"query": q})
        findings.append(result)
        print(f"🌐 搜索「{q}」→ 获得{len(result)}字资料")
    return {"findings": findings, "rounds": state["rounds"] + 1}

def summarize_node(state: State):
    r = model.invoke(f"汇总以下资料，为「{state['topic']}」写100字总结：\n" + "\n".join(state["findings"]))
    print(f"📝 总结完成")
    return {"summary": r.content}

def reflect_node(state: State):
    r = model.invoke(
        f"这个总结是否完整？只回答 YES 或 NO：\n{state['summary']}\n"
        f"标准：需涵盖定义、核心价值、应用场景"
    )
    need = "NO" in r.content.upper()
    print(f"🤔 反思：{'资料不足，需要补充搜索' if need else '资料充分'}")
    return {"need_more": need}

def route(state: State) -> str:
    # 资料不足且没超2轮 → 再搜一轮；否则结束（防死循环）
    return "plan" if state["need_more"] and state["rounds"] < 2 else "end"

# ---------- 组装 ----------
builder = StateGraph(State)
builder.add_node("plan", plan_node)
builder.add_node("search", search_node)
builder.add_node("summarize", summarize_node)
builder.add_node("reflect", reflect_node)
builder.add_edge(START, "plan")
builder.add_edge("plan", "search")
builder.add_edge("search", "summarize")
builder.add_edge("summarize", "reflect")
builder.add_conditional_edges("reflect", route, {"plan": "plan", "end": END})
graph = builder.compile()

print("=== 研究型 Agent 启动 ===")
final = graph.invoke({"topic": "LangChain 1.0", "queries": [], "findings": [],
                      "summary": "", "need_more": False, "rounds": 0})
print(f"\n📄 最终研究报告（共搜索{final['rounds']}轮）：\n{final['summary']}")
