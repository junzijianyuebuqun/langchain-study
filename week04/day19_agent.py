# ============================================
# Day 19-20：create_agent 创建智能体（1.0 标准入口）
# 面试必考：ReAct 循环 = 思考 → 选工具 → 执行 → 观察 → 再思考 → 回答
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain.agents import create_agent

# --------------------------------------------
# 1. 准备工具（Day 18 的内容）
# --------------------------------------------
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入如 '3 * (4 + 5)'，返回计算结果。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错：{e}"

@tool
def get_weather(city: str) -> str:
    """查询指定中国城市今天的天气。输入城市名，如 '北京'。"""
    fake_data = {"北京": "晴，25°C", "上海": "多云，28°C", "广州": "雷阵雨，31°C"}
    return fake_data.get(city, f"{city}今天晴，26°C（模拟数据）")

# --------------------------------------------
# 2. 创建 Agent：模型 + 工具 + 系统提示
# --------------------------------------------
model = ChatDeepSeek(model="deepseek-chat")

agent = create_agent(
    model=model,
    tools=[calculator, get_weather],
    system_prompt="你是一个助手，需要用工具时主动调用工具，回答简洁",
)

# --------------------------------------------
# 3. 运行：一个需要"用两个工具"的复合问题
#    观察 ReAct 循环：模型自己决定先查天气、再算数
# --------------------------------------------
result = agent.invoke({
    "messages": [{"role": "user", "content": "北京今天天气怎么样？另外帮我算一下 123*456 等于多少"}]
})

# 打印完整的消息链：能看到 思考→调工具→拿结果→再调→最终回答 的全过程
for msg in result["messages"]:
    role = msg.__class__.__name__.replace("Message", "")
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"🤔 [{role}] 决定调用工具：{tc['name']}，参数：{tc['args']}")
    elif role == "Tool":
        print(f"🔧 [Tool] 工具返回：{content[:60]}")
    else:
        print(f"💬 [{role}] {content[:100]}")

print("\n✅ 最终回答：", result["messages"][-1].content)
