# ============================================================
# 官网教程 06：智能体（Agents）
# 官方地址：https://python.langchain.com/docs/tutorials/agents/
# 所属模块：LangChain（agents，底层运行于 LangGraph）
# 官方目标：构建能与外部工具交互的智能体
#
# 【与官网的对应关系】
#   官网用 create_agent + 搜索工具(Tavily) → 这里换成自定义工具（免注册）
#   官网要点：端到端 Agent / 自定义 prompt / 加记忆，全部保留
# ============================================================
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# --- 第1步：定义工具（对应官网的搜索工具，这里用模拟数据避免额外注册） ---
@tool
def search_web(query: str) -> str:
    """搜索互联网获取实时信息。输入搜索关键词。"""
    fake_db = {
        "天气": "北京今天晴，气温25°C，微风",
        "金价": "今日黄金价格约780元/克",
    }
    for k, v in fake_db.items():
        if k in query:
            return v
    return f"关于「{query}」的搜索结果：暂未找到相关信息"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式，输入如 '3 * (4 + 5)'。"""
    return str(eval(expression))

# --- 第2步：创建 Agent（官网核心代码，一行搞定） ---
model = init_chat_model("deepseek-chat", model_provider="deepseek")

agent = create_agent(
    model=model,
    tools=[search_web, calculator],
    system_prompt="你是一个有用的助手，需要时调用工具",  # 官网：可自定义 prompt
)

# --- 第3步：运行（官网演示 stream 观察每一步） ---
print("【Agent 运行过程】")
query = "北京今天天气怎么样？顺便算一下 1024 * 4"
for step in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    msg = step["messages"][-1]
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        print(f"  🤔 调工具：{msg.tool_calls[0]['name']}({msg.tool_calls[0]['args']})")
    elif msg.__class__.__name__ == "ToolMessage":
        print(f"  🔧 工具返回：{msg.content[:40]}")
    elif msg.__class__.__name__ == "AIMessage" and msg.content:
        print(f"  💬 回答：{msg.content}")

# --- 第4步：加记忆（官网进阶：checkpointer + thread_id） ---
print("\n【带记忆的 Agent】")
agent_with_memory = create_agent(
    model=model, tools=[search_web],
    system_prompt="你是助手", checkpointer=MemorySaver(),
)
config = {"configurable": {"thread_id": "demo_1"}}
agent_with_memory.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config=config)
r = agent_with_memory.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config=config)
print(f"  问'我叫什么' → {r['messages'][-1].content}")
