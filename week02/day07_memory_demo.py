# ============================================
# Day 7：记忆原理演示（非交互版，用于验证）
# 模拟 3 轮对话，证明"记忆"= 历史消息列表的累积
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatDeepSeek(model="deepseek-chat")

messages = [SystemMessage(content="你是一个友好的中文助手，回答简洁")]

# 模拟用户的三轮输入（真实场景来自 input()）
fake_user_inputs = [
    "我叫小明，养了一只叫豆豆的猫",
    "豆豆今年3岁了",
    "你还记得我的猫叫什么、几岁了吗？",  # ← 检验记忆
]

for user_input in fake_user_inputs:
    messages.append(HumanMessage(content=user_input))
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f"你：{user_input}")
    print(f"🤖：{response.content}")
    print("-" * 40)

# 观察：messages 列表越来越长，这就是"记忆"的全部真相
print(f"📒 记忆本里现在共 {len(messages)} 条消息（含system）")
