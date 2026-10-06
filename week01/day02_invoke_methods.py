# ============================================
# Day 2 - Step 2.2：四种调用方式（流式输出面试必问！）
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek

model = ChatDeepSeek(model="deepseek-chat")

# --------------------------------------------
# 方式 1：invoke —— 同步调用（等全部生成完一次性返回）
# --------------------------------------------
print("【方式1：invoke 同步调用】")
r = model.invoke("用一句话介绍北京")
print(r.content)

# --------------------------------------------
# 方式 2：batch —— 批量调用（一次传多个问题，并行处理）
# --------------------------------------------
print("\n【方式2：batch 批量调用】")
results = model.batch(["1+1=?", "2+2=?", "中国的首都是哪里？"])
for r in results:
    print("-", r.content)

# --------------------------------------------
# 方式 3：stream —— 流式输出（打字机效果，面试必问！）
# 原理：模型每生成一小段就立刻推送给你，不用等全部生成完
# 应用：所有聊天软件（ChatGPT、DeepSeek 网页版）的打字机效果都是它
# --------------------------------------------
print("\n【方式3：stream 流式输出（注意看文字是一个个字蹦出来的）】")
for chunk in model.stream("写一首关于程序员的四行短诗"):
    print(chunk.content, end="", flush=True)
print()  # 最后换行

# --------------------------------------------
# 方式 4：ainvoke —— 异步调用（不阻塞程序，高并发场景用）
# --------------------------------------------
print("\n【方式4：ainvoke 异步调用】")
import asyncio

async def main():
    r = await model.ainvoke("用一句话介绍上海")
    print(r.content)

asyncio.run(main())
