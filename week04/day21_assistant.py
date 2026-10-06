# ============================================
# Day 21：Agent 练习 —— 生活助手（4 个工具）
# 工具：计算器、查天气、记事本（读写本地文件）、当前时间
# ============================================
import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain.agents import create_agent

NOTE_FILE = "week04/notes.txt"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入如 '3 * (4 + 5)'。"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算出错：{e}"

@tool
def get_weather(city: str) -> str:
    """查询指定中国城市今天的天气。"""
    fake_data = {"北京": "晴，25°C", "上海": "多云，28°C"}
    return fake_data.get(city, f"{city}今天晴，26°C（模拟数据）")

@tool
def save_note(content: str) -> str:
    """把内容追加保存到记事本文件。输入要记录的文字内容。"""
    with open(NOTE_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%Y-%m-%d %H:%M}] {content}\n")
    return f"已记录：{content}"

@tool
def read_notes() -> str:
    """读取记事本里的所有记录。无需输入参数。"""
    if not os.path.exists(NOTE_FILE):
        return "记事本是空的"
    with open(NOTE_FILE, "r", encoding="utf-8") as f:
        return f.read()

model = ChatDeepSeek(model="deepseek-chat")
agent = create_agent(
    model=model,
    tools=[calculator, get_weather, save_note, read_notes],
    system_prompt="你是生活助手，可以查天气、算数、帮用户记笔记，回答简洁",
)

# 复合指令测试：需要连续调用 记笔记 + 查天气 两个工具
result = agent.invoke({
    "messages": [{"role": "user", "content": "帮我记一下：明天下午3点开产品会。另外上海天气怎么样？"}]
})
print("✅ 最终回答：", result["messages"][-1].content)

# 再验证：模型能通过工具读到刚才记的内容
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "看看我的记事本里有什么"}]
})
print("✅ 读取结果：", result2["messages"][-1].content)
