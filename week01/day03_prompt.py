# ============================================
# Day 3 - Step 3.1：ChatPromptTemplate 基础
# 目标：学会用 {占位符变量} 做可复用的提示词模板
# ============================================
from dotenv import load_dotenv
load_dotenv()

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

model = ChatDeepSeek(model="deepseek-chat")

# --------------------------------------------
# 没有模板时：每次都要手写完整的话，重复劳动
# --------------------------------------------
# model.invoke("你是一个专业的老师，用通俗易懂的语言解释什么是向量数据库")
# model.invoke("你是一个专业的老师，用通俗易懂的语言解释什么是RAG")
# ↑ 两句话只有"主题"不同，太浪费了

# --------------------------------------------
# 用模板：{xxx} 是占位符，调用时再填值
# --------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}，用通俗易懂的语言解释概念，100字以内"),
    ("human", "请解释什么是{topic}"),
])

# 把模板和模型用 | 连起来（这就是 LCEL 管道，Day 6 细讲）
chain = prompt | model

# 同一个模板，换不同的值反复使用！
print("【第1次】role=老师, topic=向量数据库：")
r1 = chain.invoke({"role": "老师", "topic": "向量数据库"})
print(r1.content)

print("\n【第2次】role=厨师, topic=区块链（换个角色，解释风格完全不同）：")
r2 = chain.invoke({"role": "厨师", "topic": "区块链"})
print(r2.content)

# --------------------------------------------
# 简写形式：("system", ...) 等价于 SystemMessage
# --------------------------------------------
# 这两种写法完全等价：
#   ChatPromptTemplate.from_messages([("system", "你是{role}")])
#   ChatPromptTemplate.from_messages([SystemMessage(content="你是{role}")])
