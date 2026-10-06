# ============================================
# Day 2 - Step 2.1：三种消息类型（面试高频！）
# 目标：理解 SystemMessage / HumanMessage / AIMessage
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatDeepSeek(model="deepseek-chat")

# --------------------------------------------
# 实验 1：基础用法 —— SystemMessage 设定角色
# --------------------------------------------
print("【实验1】严厉的面试官人设：")
messages = [
    SystemMessage(content="你是一个严厉的面试官，只回答技术问题，回答不超过50字"),
    HumanMessage(content="什么是 RAG？"),
]
response = model.invoke(messages)
print(response.content)

# --------------------------------------------
# 实验 2：改一下 SystemMessage，人设立刻变
# --------------------------------------------
print("\n【实验2】东北话人设（同一个问题，对比回答风格）：")
messages2 = [
    SystemMessage(content="你是一个东北人，必须用东北话回答，不超过50字"),
    HumanMessage(content="什么是 RAG？"),
]
response2 = model.invoke(messages2)
print(response2.content)

# --------------------------------------------
# 实验 3：AIMessage 的作用 —— 把历史对话一起发给模型
# 模型本身没有记忆！"记忆"= 每轮把之前的消息都带上
# --------------------------------------------
print("\n【实验3】带历史消息的对话（模型能记住你叫小明）：")
messages3 = [
    SystemMessage(content="你是一个友好的助手"),
    HumanMessage(content="我叫小明"),
    AIMessage(content="你好小明，很高兴认识你！有什么可以帮你的吗？"),  # 上一轮的回复
    HumanMessage(content="我叫什么名字？"),  # 测试模型是否"记得"
]
response3 = model.invoke(messages3)
print(response3.content)

# --------------------------------------------
# 观察返回的 response 到底是什么
# --------------------------------------------
print("\n【观察】response 对象的类型和内容：")
print(type(response3))          # <class 'langchain_core.messages.ai.AIMessage'>
print(response3.content)        # 文字内容
