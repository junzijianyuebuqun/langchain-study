# ============================================
# Day 7：对话记忆 Memory —— 命令行多轮聊天机器人
# 核心原理：模型没有记忆！把历史消息列表每轮一起发给模型
#
# 运行方式（交互式）：
#   .\python_3_12\python.exe -X utf8 week01\day07_chatbot.py
# 输入 exit 退出
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

model = ChatDeepSeek(model="deepseek-chat")

# 1. 消息列表 = "记忆本"，开局先放人设
messages = [
    SystemMessage(content="你是一个友好的中文助手，回答简洁，不超过100字")
]

print("🤖 聊天机器人已启动！输入 exit 退出")
print("-" * 40)

while True:
    user_input = input("你：")
    if user_input.strip().lower() == "exit":
        print("👋 再见！")
        break

    # 2. 把用户的话追加到"记忆本"
    messages.append(HumanMessage(content=user_input))

    # 3. 把【整个历史】发给模型 —— 这就是记忆的真相！
    response = model.invoke(messages)

    # 4. 模型的回答也追加进"记忆本"，下一轮才能记得
    messages.append(AIMessage(content=response.content))

    print(f"🤖：{response.content}")
    print("-" * 40)
